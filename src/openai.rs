use std::borrow::Cow;
use std::cmp::min;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{iter, mem, str};

use async_openai::types::*;
use either::Either;
use futures::channel::mpsc;
use futures::channel::oneshot as ones;
use futures::sink::SinkExt;
use llama_sys::sampling::{Logit, Output};
use llama_sys::{Context, ContextParams, Model, ModelParams, Token, Tokenizer};
use once_cell::sync::OnceCell;
use rocket::http::Status;
use rocket::response::stream::{Event, EventStream};
use rocket::serde::json::{Json, Value};
use rocket::serde::Serialize;
use rocket::tokio::task::{spawn_blocking, JoinHandle};
use rocket::{Route, State};

use super::prompt::Vocab;
use super::radix::{Radix, RadixKv};

pub fn routes() -> Vec<Route> {
    routes![chat_completions]
}

#[derive(Serialize, Default)]
struct ApiError {
    message: Cow<'static, str>,
    r#type: Option<Cow<'static, str>>,
    param: Option<Cow<'static, str>>,
    code: Option<Cow<'static, str>>,
}

type Fails = (Status, Json<ApiError>);

// TODO: make it better
fn fails<S: Into<Cow<'static, str>>>(status: Status, message: S) -> Fails {
    let e = ApiError {
        message: message.into(),
        ..Default::default()
    };

    (status, Json(e))
}

type ChatCompletionResponse =
    Either<Json<CreateChatCompletionResponse>, EventStream<mpsc::Receiver<Event>>>;

#[post("/v1/chat/completions", format = "json", data = "<req>")]
async fn chat_completions(
    api: &State<Api>,
    req: Json<CreateChatCompletionRequest>,
) -> Result<ChatCompletionResponse, Fails> {
    let r = req.into_inner();

    if r.messages.is_empty() {
        return Err(fails(Status::BadRequest, "no messages"));
    }

    let logit_bias = r.logit_bias.into_iter().flatten().map(|(k, v)| {
        let id = k
            .parse::<u32>()
            .map_err(|_| fails(Status::BadRequest, "invalid token in logit bias"))?;

        let Value::Number(n) = v else {
            return Err(fails(Status::BadRequest, "bias value is not a number"));
        };

        let bias = n
            .as_f64()
            .ok_or_else(|| fails(Status::BadRequest, "bias value"))?;

        if !(-100. ..=100.).contains(&bias) {
            return Err(fails(Status::BadRequest, "bias out of range"));
        }

        Ok((id as Token, bias as f32))
    });

    let logit_bias: Vec<_> = logit_bias.collect::<Result<_, Fails>>()?;

    let logprobs = r.logprobs.unwrap_or(false);

    if r.top_logprobs.is_some_and(|_| !logprobs) {
        return Err(fails(Status::BadRequest, "top_logprobs but logprobs=false"));
    }

    if r.top_logprobs.is_some_and(|n| n > 20) {
        return Err(fails(Status::BadRequest, "too many logprobs"));
    }

    let top_logprobs: usize = r.top_logprobs.unwrap_or(0).into();

    if r.temperature.is_some_and(|v| !(0. ..=2.).contains(&v)) {
        return Err(fails(Status::BadRequest, "temperature out of range"));
    }

    if r.top_p.is_some_and(|v| !(0. ..=1.).contains(&v)) {
        return Err(fails(Status::BadRequest, "top_p out of range"));
    }

    let api = api.inner();
    let mut span = vec![api.tk.token_bos()];

    let v = Vocab::Llama3;

    for msg in r.messages.iter() {
        match msg {
            ChatCompletionRequestMessage::System(m) => {
                api.feed(&mut span, v.system(), &m.content);
            }

            ChatCompletionRequestMessage::User(m) => match &m.content {
                ChatCompletionRequestUserMessageContent::Text(content) => {
                    api.feed(&mut span, v.user(), content);
                }
                ChatCompletionRequestUserMessageContent::Array(_) => {
                    unimplemented!();
                }
            },

            ChatCompletionRequestMessage::Assistant(m) => match (&m.content, &m.tool_calls) {
                (Some(content), None) => {
                    api.feed(&mut span, v.assistant(), content);
                }

                (None, Some(_tools)) => {
                    unimplemented!();
                }

                // TODO: handle errs
                _ => panic!(),
            },

            ChatCompletionRequestMessage::Tool(m) => {
                api.feed(&mut span, v.tool_out(), &m.content);
            }

            ChatCompletionRequestMessage::Function(m) => {
                unimplemented!();
            }
        }
    }

    // TODO(tools): we should not be adding this header, as we may want tool calls
    api.prep(&mut span, v.assistant());

    // token limits
    let n_max = min(api.n_train, api.n_cells);

    if span.len() > n_max {
        let msg = format!("too many prompt tokens: {} > {n_max}", span.len());
        return Err(fails(Status::InsufficientStorage, msg));
    }

    let k = api.n_cells - span.len();
    let n: usize = r.n.unwrap_or(1).into();
    let q = min(k / n, api.n_train - span.len());
    let n_infer = min(r.max_tokens.map(|v| v as usize).unwrap_or(q), q);

    let (tx, rx) = if r.stream.unwrap_or(false) {
        let (tx, rx) = mpsc::channel(32);
        (Either::Right(tx), Either::Right(rx))
    } else {
        let (tx, rx) = ones::channel();
        (Either::Left(Some(tx)), Either::Left(rx))
    };

    let mut p = Inflight {
        temperature: r.temperature.unwrap_or(1.),
        min_s: 1. - r.top_p.unwrap_or(1.),
        logit_bias,
        logprobs,
        top_logprobs,

        n_infill: span.len(),
        n_infer,
        n_actual: 0,
        span: vec![span; n],
        text: vec![vec![]; n],
        prob: vec![],

        finish: vec![(n_infer == 0).then_some(FinishReason::Length); n],
        tx,
    };

    if p.logprobs {
        p.prob.resize_with(n, Default::default);
    }

    api.schedule(p).await?;

    Ok(match rx {
        Either::Left(recv) => Either::Left(
            recv.await
                .map_err(|_| fails(Status::InternalServerError, "cancelled"))?,
        ),

        Either::Right(recv) => Either::Right(EventStream::from(recv)),
    })
}

// TODO(shift): support priveleged 'sliding window' type spans

// TODO(beam-search): modifications needed aren't huge:
// - decoding logic is unchanged
// - we already store space for beams (choices)
// can then further search for CoTs ala https://arxiv.org/pdf/2402.10200

struct Inflight {
    temperature: f32, // softmax temperature
    min_s: f32,       // min ratio of probability steps
    logit_bias: Vec<(Token, f32)>,
    logprobs: bool,
    top_logprobs: usize,

    n_infill: usize,       // how much of span is infill?
    n_infer: usize,        // max tokens to infer
    n_actual: usize,       // # tokens actually decoded (noncached)
    span: Vec<Vec<Token>>, // span tokens
    text: Vec<Vec<u8>>,    // detokenized
    prob: Vec<Vec<ChatCompletionTokenLogprob>>,

    finish: Vec<Option<FinishReason>>,
    tx: Either<Option<ones::Sender<Json<CreateChatCompletionResponse>>>, mpsc::Sender<Event>>,
}

impl Inflight {
    fn has_enabled(&self) -> bool {
        self.finish.iter().any(|f| f.is_none())
    }

    fn enabled(&self) -> impl Iterator<Item = &[Token]> {
        iter::zip(&self.span, &self.finish)
            .filter(|(_, e)| e.is_none())
            .map(|(s, _)| s.as_slice())
    }

    fn accept(&mut self, ctx: &Context, memo: &TokenCache, logits: Output<Logit>, i: usize) {
        assert!(self.finish[i].is_none(), "accept on disabled span?");

        let prob = logits
            .apply_logit_bias(self.logit_bias.iter().copied())
            .softmax(self.temperature);

        // if enabled, grab the top logprobs before truncating & converting the distribution
        let top_logprobs = self
            .logprobs
            .then(|| prob.logprobs().take(20).collect::<Vec<_>>());

        // j indexes into top_logprobs, but may be out of bounds
        let (j, tok) = prob
            // TODO: figure out how to pass min_p, or if we even need it
            .trunc_k(0., self.min_s, 1)
            .distribution()
            .sample();

        self.span[i].push(tok);
        ctx.detokenize(&mut self.text[i], tok, false);

        // finish computing logprobs now that we know which token was sampled
        if let Some(top_logprobs) = top_logprobs {
            let (token, bytes) = lookup_token(memo, ctx, tok);

            let tlp = ChatCompletionTokenLogprob {
                token,
                logprob: top_logprobs.get(j).map(|(_, lp)| *lp).unwrap_or(-9999.),
                bytes,
                top_logprobs: top_logprobs
                    .into_iter()
                    .take(self.top_logprobs)
                    .map(|(id, lp)| (lookup_token(memo, ctx, id), lp))
                    .map(|((token, bytes), logprob)| TopLogprobs { token, logprob, bytes })
                    .collect(),
            };

            self.prob[i].push(tlp);
        }

        // check stop conds
        if ctx.model().token_is_eog(tok) {
            self.finish[i] = Some(FinishReason::Stop);
        } else if self.span[i].len() - self.n_infill >= self.n_infer {
            self.finish[i] = Some(FinishReason::Length);
        }

        let done = !self.has_enabled();

        match &mut self.tx {
            Either::Left(tx) if done => {
                let mut logprobs = self
                    .prob
                    .drain(..)
                    .map(|content| ChatChoiceLogprobs { content: Some(content) });

                // TODO: tools
                let choices = iter::zip(&self.finish, &self.text)
                    .enumerate()
                    .map(|(i, (&finish_reason, text))| ChatChoice {
                        index: i as u32,
                        message: ChatCompletionResponseMessage {
                            content: Some(String::from_utf8_lossy(text).into_owned()),
                            tool_calls: None,
                            role: Role::Assistant,
                            function_call: None,
                        },
                        finish_reason,
                        logprobs: logprobs.next(),
                    })
                    .collect();

                let prompt_tokens = self.n_infill as u32;

                let usage = CompletionUsage {
                    prompt_tokens,
                    completion_tokens: self
                        .span
                        .iter()
                        .map(|s| s.len() as u32 - prompt_tokens)
                        .sum(),
                    total_tokens: self.n_actual as u32,
                };

                // TODO: id, timestamp, model, fingerprint
                let completion = CreateChatCompletionResponse {
                    id: "id".to_owned(),
                    choices,
                    created: 0,
                    model: "model".to_owned(),
                    system_fingerprint: None,
                    object: "chat.completion".to_owned(),
                    usage: Some(usage),
                };

                let _ = tx.take().unwrap().send(Json(completion));
            }

            Either::Right(tx) => {
                //let _ = tx.try_send(Event::json(&Completion::Chunk {}));
            }

            _ => {}
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to load model")]
    LoadModel,

    #[error("failed to allocate context")]
    LoadContext,

    #[error("llama error: {0:?}")]
    Llama(#[from] llama_sys::Error),
}

pub struct Api {
    queue: mpsc::Sender<Inflight>,
    inner: Arc<Mutex<Inner>>,
    tk: Tokenizer,
    n_train: usize,
    n_cells: usize,
}

impl Api {
    /// Load a gguf model from the filesystem.
    pub fn load<P: AsRef<Path>>(path: P, m: ModelParams, c: ContextParams) -> Result<Self, Error> {
        let model = Model::load_from_file(path, m).ok_or(Error::LoadModel)?;
        let mut ctx = model.context(c).ok_or(Error::LoadContext)?;
        let tk = model.tokenizer();

        let (queue, rx) = mpsc::channel(32);

        let n_train = model.n_ctx_train();
        let n_cells = ctx.kv_cache_n_cells();
        let vocab = vec![OnceCell::new(); model.n_vocab() as usize].into();
        let root = ctx.kv_cache_unique_id();

        let inner = Arc::new(Mutex::new(Inner {
            ctx,
            n_batch: c.n_batch as usize,
            queue: rx,
            sched: vec![],
            cache: RadixKv::new(root),
            vocab,
        }));

        Ok(Self {
            queue,
            inner,
            tk,
            n_train,
            n_cells,
        })
    }

    /// Feed a closed header into buf.
    fn feed(&self, buf: &mut Vec<Token>, h: [&str; 2], text: &str) {
        self.tk.tokenize(buf, h[0], true);
        self.tk.tokenize(buf, text, false);
        self.tk.tokenize(buf, h[1], true);
    }

    /// Feed an open header into buf.
    fn prep(&self, buf: &mut Vec<Token>, h: [&str; 2]) {
        self.tk.tokenize(buf, h[0], true);
    }

    /// Schedule an inflight completion req.
    async fn schedule(&self, params: Inflight) -> Result<JoinHandle<()>, Fails> {
        let mut jobs = self.queue.clone();

        jobs.send(params)
            .await
            .map_err(|_| fails(Status::InternalServerError, "failed to schedule span"))?;

        let inner = self.inner.clone();

        Ok(spawn_blocking(move || Inner::flush_all(inner).unwrap()))
    }
}

type TokenCache = [OnceCell<(String, Vec<u8>)>];

fn lookup_token(memo: &TokenCache, ctx: &Context, id: Token) -> (String, Option<Vec<u8>>) {
    let (text, bytes) = &memo[id as usize].get_or_init(|| {
        let mut bytes = vec![];
        ctx.detokenize(&mut bytes, id, false);
        bytes.shrink_to_fit();

        let text = String::from_utf8_lossy(&bytes);

        (text.into(), bytes)
    });

    (text.clone(), Some(bytes.clone()))
}

struct Inner {
    ctx: Context,
    n_batch: usize,
    queue: mpsc::Receiver<Inflight>,
    sched: Vec<Inflight>,
    cache: RadixKv,
    vocab: Arc<TokenCache>,
}

impl Inner {
    fn flush_all(mu: Arc<Mutex<Self>>) -> Result<(), Error> {
        while !{
            let mut ex = mu.lock().unwrap();
            ex.flush_bounded()?;
            ex.sched.is_empty()
        } {}

        Ok(())
    }

    // TODO: use tracing to record performance data

    fn flush_bounded(&mut self) -> Result<(), Error> {
        // move any new requests to the back of scheduled queue
        while let Ok(Some(p)) = self.queue.try_next() {
            self.sched.push(p);
        }

        // compute how much kv space may be allocated to each scheduled span
        let mut plan = Radix::new();
        let mut left = self.ctx.kv_cache_n_cells();
        let mut k = 0; // # of params that fit

        'outer: for p in self.sched.iter() {
            for span in p.enabled() {
                let a = plan.ancestor(span);
                let n = span.len() - a.n;

                if n > left {
                    break 'outer;
                }

                left -= n;
                plan.insert(a, span);
            }

            k += 1;
        }

        let sched = &mut self.sched[..k];

        // at this point, we know all spans in sched will pack into available kv
        let bounds: Vec<_> = sched
            .iter()
            .flat_map(|p| p.enabled())
            .map(|span| span.len() - self.cache.ancestor(span).n)
            .collect();

        let mut actual = Vec::with_capacity(sched.len());

        // distribute (inexact in the case of shared prefixes)
        for (m, (i, span)) in iter::zip(
            clamp_sum(self.n_batch, &bounds, |&n| n),
            sched
                .iter()
                .enumerate()
                .flat_map(|(i, p)| p.enabled().map(move |s| (i, s))),
        ) {
            if m == 0 {
                continue;
            }

            let a = self.cache.ancestor(span);
            let n = min(span.len(), a.n + m);
            actual.push((i, n - a.n));

            self.cache.insert(&mut self.ctx, a, &span[..n]);
        }

        for (i, k) in actual {
            sched[i].n_actual += k;
        }

        // prune (skipping all running) to ensure upcoming batch doesn't overflow kv size
        let it = sched.iter().flat_map(|p| p.enabled());
        self.cache.prune(&mut self.ctx, it);

        // decode w/e spans are marked as cold
        let batch = self.cache.pending_batch();
        let evals = self.ctx.decode(&batch)?;

        self.cache
            .accept_batch(evals.into_iter().map(|o| o.unwrap()));

        // dispatch logits to any running spans that are fully infilled
        let mut gc = vec![];

        for (i, p) in sched.iter_mut().enumerate() {
            for j in 0..p.span.len() {
                if p.finish[j].is_some() {
                    continue;
                }

                let span = &p.span[j];
                let a = self.cache.ancestor(span);

                if a.n < span.len() {
                    continue;
                }

                assert_eq!(a.n, span.len());
                assert_eq!(self.cache[a.i].tokens.len(), self.cache[a.i].logits.len(),);

                let logits = self.cache[a.i].logits[a.k - 1].clone();
                p.accept(&self.ctx, &self.vocab, logits, j);
            }

            if !p.has_enabled() {
                gc.push(i);
            }
        }

        for i in gc.into_iter().rev() {
            self.sched.remove(i);
        }

        Ok(())
    }
}

fn clamp_sum<'a, T, F: Fn(&'a T) -> usize>(mut left: usize, upper: &'a [T], cost: F) -> Vec<usize> {
    let mut take = vec![0; upper.len()];
    let mut more = true;

    'outer: while mem::replace(&mut more, false) {
        for (val, cur) in iter::zip(upper, take.iter_mut()) {
            let max = cost(val);

            if *cur >= max {
                continue;
            }

            if left == 0 {
                break 'outer;
            }

            *cur += 1;
            left -= 1;
            more |= *cur < max;
        }
    }

    take
}

#[test]
fn test_clamp_sum() {
    let id = |&v| -> usize { v };

    assert_eq!(clamp_sum(12, &[1, 2, 45, 0, 9], id), [1, 2, 5, 0, 4]);
    assert_eq!(clamp_sum(12, &[1, 2, 45, 1, 9], id), [1, 2, 4, 1, 4]);
    assert_eq!(clamp_sum(24, &[1, 2, 45, 1, 9], id), [1, 2, 11, 1, 9]);
    assert_eq!(clamp_sum(512, &[1, 2, 45, 1, 9], id), [1, 2, 45, 1, 9]);
    assert_eq!(clamp_sum(0, &[1, 2, 45, 1, 9], id), [0, 0, 0, 0, 0]);
    assert_eq!(clamp_sum(0, &[], id), [0usize; 0]);
    assert_eq!(clamp_sum(7, &[], id), [0usize; 0]);
    assert_eq!(clamp_sum(37, &[92, 488, 45, 1, 102], id), [9, 9, 9, 1, 9]);
}
