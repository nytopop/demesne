use std::borrow::Cow;
use std::cmp::min;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{iter, mem, str};

use either::Either;
use futures::channel::mpsc;
use futures::channel::oneshot as ones;
use futures::sink::SinkExt;
use llama_sys::sampling::{Logit, Output};
use llama_sys::{Context, ContextParams, Model, ModelParams, Token, Tokenizer};
use rocket::http::Status;
use rocket::response::stream::{Event, EventStream};
use rocket::serde::json::{json, Json, Value};
use rocket::serde::{Deserialize, Serialize};
use rocket::tokio::task::{spawn_blocking, JoinHandle};
use rocket::{Request, Route, State};
use serde_json::Map;
use validator::Validate;

use super::prompt::Vocab;
use super::radix::{Radix, RadixKv};

pub fn routes() -> Vec<Route> {
    routes![chat_completions]
}

type Fail = (Status, Json<Value>);

#[catch(default)]
pub fn catch(status: Status, _: &Request) -> Fail {
    fail(status, "unknown")
}

fn fail<R: AsRef<str>>(status: Status, reason: R) -> Fail {
    (status, Json(json! {{"reason": reason.as_ref()}}))
}

#[derive(Deserialize, Validate)]
pub struct CompletionReq<'a> {
    #[serde(borrow)]
    messages: Vec<Message<'a>>,

    max_tokens: Option<usize>, // default cx-input; # of tokens to infer

    #[validate(range(min = 1))]
    n: Option<usize>, // default 1; # of completion choices

    // 0=greedy, 2=insane
    #[validate(range(min = 0., max = 2.))]
    temperature: Option<f32>, // default 1

    // 0=greedy, 1=all
    #[validate(range(min = 0., max = 1.))]
    min_p: Option<f32>, // default NONE

    // 0=top-1, 1=top-k
    #[validate(range(min = 0., max = 1.))]
    min_s: Option<f32>, // default NONE

    // NOTE(valid): vals must be (-100, 100)
    logit_bias: Option<HashMap<Token, f32>>,

    #[serde(default)]
    logprobs: bool,

    #[validate(range(min = 0, max = 20))]
    top_logprobs: Option<usize>, // default NONE; requires logprobs

    #[serde(default)]
    response_format: Format,

    stop: Option<Stop<'a>>,

    tools: Option<Vec<Tool<'a>>>,

    tool_choice: Option<ToolChoice<'a>>, // default "auto"

    // TODO: stream_options for include usage on streams
    stream: Option<bool>, // default false
}

#[derive(Deserialize)]
#[serde(tag = "role")]
enum Message<'a> {
    #[serde(rename = "system")]
    System {
        content: Cow<'a, str>,
        name: Option<Cow<'a, str>>,
    },

    #[serde(rename = "user")]
    User {
        content: Cow<'a, str>,
        name: Option<Cow<'a, str>>,
    },

    #[serde(rename = "assistant")]
    Assistant {
        content: Cow<'a, str>,
        name: Option<Cow<'a, str>>,
        tool_calls: Option<Vec<Call<'a>>>,
    },

    #[serde(rename = "tool")]
    Tool {
        content: Cow<'a, str>,
        tool_call_id: Cow<'a, str>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum Call<'a> {
    #[serde(rename = "function")]
    Function {
        id: Cow<'a, str>,
        function: Function<'a>,
        index: Option<usize>,
    },
}

#[derive(Serialize, Deserialize)]
struct Function<'a> {
    name: Cow<'a, str>,
    arguments: Cow<'a, str>,
}

#[derive(Serialize, Deserialize, Default)]
#[serde(tag = "type")]
enum Format {
    #[default]
    #[serde(rename = "text")]
    Text,

    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Stop<'a> {
    One(&'a str),

    Many(Vec<&'a str>),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum Tool<'a> {
    Function {
        description: Option<&'a str>,
        name: &'a str,
        parameters: Option<Map<String, Value>>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum ToolChoice<'a> {
    #[serde(rename = "function")]
    Function { name: &'a str },

    #[serde(untagged)]
    Tool(&'a str), // "none" | "auto" | "required"
}

#[derive(Serialize)]
#[serde(tag = "object")]
enum Completion {
    #[serde(rename = "chat.completion")]
    Completion {
        id: String,
        choices: Vec<CompletionChoice>,
        created: u64,
        model: String,
        system_fingerprint: String,
        usage: Usage,
    },

    #[serde(rename = "chat.completion.chunk")]
    Chunk {
        id: String,
        choices: Vec<ChunkChoice>,
        created: u64,
        model: String,
        system_fingerprint: String,
        usage: Option<Usage>, // only if last && stream include usage is set
    },
}

#[derive(Serialize)]
struct CompletionChoice {
    finish_reason: &'static str, // "stop" | "length" | "content_filter" | "tool_calls"
    index: usize,
    message: RespMessage,
    logprobs: Option<LogProbs>,
}

#[derive(Serialize)]
struct ChunkChoice {
    finish_reason: Option<&'static str>, // "stop" | "length" | "content_filter" | "tool_calls"
    index: usize,
    delta: RespMessage,
    logprobs: Option<LogProbs>,
}

#[derive(Serialize)]
#[serde(tag = "role")]
enum RespMessage {
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<String>,
        tool_calls: Vec<Call<'static>>,
    },
}

// TODO: finish
#[derive(Serialize)]
struct LogProbs {}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    cache_tokens: usize,
}

type ChatCompletion = Result<Either<Json<Completion>, EventStream<mpsc::Receiver<Event>>>, Fail>;

#[post("/v1/chat/completions", format = "json", data = "<req>")]
async fn chat_completions(api: &State<Api>, req: Json<CompletionReq<'_>>) -> ChatCompletion {
    let req = req.into_inner();

    req.validate()
        .map_err(|e| Json(serde_json::to_value(e).unwrap()))
        .map_err(|e| (Status::BadRequest, e))?;

    let api = api.inner();
    let mut span = vec![api.tk.token_bos()];

    let v = Vocab::MistralV3;

    for msg in req.messages.iter() {
        match msg {
            // NOTE: must be first
            Message::System { content, name } => {
                api.feed(&mut span, v.system(), content);
            }

            // NOTE: must be followed by assistant
            Message::User { content, name } => {
                api.feed(&mut span, v.user(), content);
            }

            // NOTE: always after user
            Message::Assistant { content, name, tool_calls } => match tool_calls {
                // TODO(tools): implement
                Some(calls) => unimplemented!(),
                None => api.feed(&mut span, v.assistant(), content),
            },

            // TODO(tools): we must create the json object
            Message::Tool { content, tool_call_id } => {
                api.feed(&mut span, v.tool_out(), content);
            }
        }
    }

    // TODO(tools): we should not be adding this header, as we may want tool calls
    api.prep(&mut span, v.assistant());

    let n_max = min(api.n_train, api.n_cells);

    if span.len() > n_max {
        let msg = format!("too many infill tokens: {} > {n_max}", span.len());
        return Err(fail(Status::InsufficientStorage, msg));
    }

    let (tx, rx) = make_response_channel(req.stream.unwrap_or(false), 32);

    // token limits
    let k = api.n_cells - span.len();
    let n = req.n.unwrap_or(1);
    let q = min(k / n, api.n_train - span.len());
    let n_infer = min(req.max_tokens.unwrap_or(q), q);

    let p = Inflight {
        temp: req.temperature.unwrap_or(1.),
        min_p: req.min_p.unwrap_or(0.),
        min_s: req.min_s.unwrap_or(0.),
        n_infill: span.len(),
        n_infer,
        n_actual: 0,
        span: vec![span; n],
        text: vec![vec![]; n],
        finish: vec![(n_infer == 0).then_some("length"); n],
        tx,
    };

    api.schedule(p).await?;

    Ok(match rx {
        Either::Left(recv) => Either::Left(
            recv.await
                .map_err(|_| fail(Status::InternalServerError, "unknown"))?,
        ),

        Either::Right(recv) => Either::Right(EventStream::from(recv)),
    })
}

#[allow(clippy::type_complexity)]
fn make_response_channel(
    stream: bool,
    buffer: usize,
) -> (
    Either<Option<ones::Sender<Json<Completion>>>, mpsc::Sender<Event>>,
    Either<ones::Receiver<Json<Completion>>, mpsc::Receiver<Event>>,
) {
    if stream {
        let (tx, rx) = mpsc::channel(buffer);
        (Either::Right(tx), Either::Right(rx))
    } else {
        let (tx, rx) = ones::channel();
        (Either::Left(Some(tx)), Either::Left(rx))
    }
}

// TODO(shift): support priveleged 'sliding window' type spans

// TODO(beam-search): modifications needed aren't huge:
// - decoding logic is unchanged
// - we already store space for beams (choices)
// can then further search for CoTs ala https://arxiv.org/pdf/2402.10200

struct Inflight {
    temp: f32,  // softmax temperature
    min_p: f32, // ratio of max probability to consider min
    min_s: f32, // min ratio of probability steps

    n_infill: usize,       // how much of span is infill?
    n_infer: usize,        // max tokens to infer
    n_actual: usize,       // # tokens actually decoded (noncached)
    span: Vec<Vec<Token>>, // span tokens
    text: Vec<Vec<u8>>,    // detokenized

    finish: Vec<Option<&'static str>>, // finish reason of each choice
    tx: Either<Option<ones::Sender<Json<Completion>>>, mpsc::Sender<Event>>,
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

    fn accept_logits(&mut self, ctx: &mut Context, logits: Output<Logit>, i: usize) {
        assert!(self.finish[i].is_none(), "accept on disabled span?");

        let tok = logits
            .softmax(self.temp)
            .trunc_k(self.min_p, self.min_s, 1)
            .distribution()
            .sample();

        self.span[i].push(tok);
        ctx.detokenize(&mut self.text[i], tok, false);

        // TODO: stop strings, fsm stuff, etc etc

        // check stop conds
        if ctx.model().token_is_eog(tok) {
            self.finish[i] = Some("stop");
        } else if self.span[i].len() - self.n_infill >= self.n_infer {
            self.finish[i] = Some("length");
        }

        let done = !self.has_enabled();

        match &mut self.tx {
            Either::Left(tx) if done => {
                // TODO: tools & logprobs
                let choices = iter::zip(&self.finish, &self.text)
                    .enumerate()
                    .map(|(i, (stop, text))| CompletionChoice {
                        finish_reason: stop.unwrap(),
                        index: i,
                        message: RespMessage::Assistant {
                            content: Some(String::from_utf8_lossy(text).into_owned()),
                            tool_calls: vec![],
                        },
                        logprobs: None,
                    })
                    .collect();

                // TODO: id, timestamp, model, fingerprint, usage
                let mut usage = Usage {
                    prompt_tokens: self.n_infill,
                    completion_tokens: self.span.iter().map(|s| s.len() - self.n_infill).sum(),
                    total_tokens: 0,
                    cache_tokens: 0,
                };

                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
                usage.cache_tokens = usage.total_tokens - self.n_actual;

                let completion = Completion::Completion {
                    id: "id".to_owned(),
                    choices,
                    created: 0,
                    model: "model".to_owned(),
                    system_fingerprint: "fingerprint".to_owned(),
                    usage,
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

    #[error("decode error: {0:?}")]
    Decode(#[from] llama_sys::Error),
}

pub struct Api {
    queue: mpsc::Sender<Inflight>,
    inner: Arc<Mutex<Inner>>,
    tk: Tokenizer,
    n_train: usize,
    n_cells: usize,
}

impl Api {
    pub fn load<P: AsRef<Path>>(path: P, m: ModelParams, c: ContextParams) -> Result<Self, Error> {
        let model = Model::load_from_file(path, m).ok_or(Error::LoadModel)?;
        let mut ctx = model.context(c).ok_or(Error::LoadContext)?;
        let tk = model.tokenizer();

        let (queue, rx) = mpsc::channel(32);

        let n_train = model.n_ctx_train();
        let n_cells = ctx.kv_cache_n_cells();
        let root = ctx.kv_cache_unique_id();

        let inner = Arc::new(Mutex::new(Inner {
            ctx,
            n_batch: c.n_batch as usize,
            queue: rx,
            sched: vec![],
            cache: RadixKv::new(root),
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
    async fn schedule(&self, params: Inflight) -> Result<JoinHandle<()>, Fail> {
        let mut jobs = self.queue.clone();

        jobs.send(params)
            .await
            .map_err(|_| fail(Status::InternalServerError, "failed to schedule span"))?;

        let inner = self.inner.clone();

        Ok(spawn_blocking(move || Inner::flush_all(inner).unwrap()))
    }
}

struct Inner {
    ctx: Context,
    n_batch: usize,
    queue: mpsc::Receiver<Inflight>,
    sched: Vec<Inflight>,
    cache: RadixKv,
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
            let k = n - a.n;
            actual.push((i, k));

            self.cache.insert(&mut self.ctx, a, &span[..n]);
        }

        // TODO: figure out a way to fuse into above loop
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
                p.accept_logits(&mut self.ctx, logits, j);
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
