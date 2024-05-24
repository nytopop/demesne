#![allow(dead_code)]
use std::cmp::min;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{iter, mem};

use either::Either;
use futures::channel::mpsc::{self, Receiver, Sender};
use futures::channel::oneshot;
use futures::sink::SinkExt;
use llama_sys::sampling::{Chain, Choice, Logits, Sample};
use llama_sys::{Context, ContextParams, Model, ModelParams, Token, Tokenizer};
use rocket::http::Status;
use rocket::response::stream::{Event, EventStream};
use rocket::serde::json::{json, Json, Value};
use rocket::serde::Deserialize;
use rocket::tokio::task::{spawn_blocking, JoinHandle};
use rocket::{Request, State};
use serde_json::Map;
use validator::Validate;

use super::radix::{Kv, RadixKv};

type Fail = (Status, Json<Value>);

type FailR<T> = Result<T, Fail>;

#[catch(default)]
pub fn catch(status: Status, _: &Request) -> Fail {
    fail(status, "unknown")
}

fn fail(status: Status, reason: &str) -> Fail {
    (status, Json(json! {{"reason": reason}}))
}

#[derive(Deserialize, Validate)]
pub struct CompletionReq<'a> {
    #[serde(borrow)]
    messages: Vec<Message<'a>>,

    max_tokens: Option<u32>, // default input-cx; # of tokens to infer

    #[validate(range(min = 1))]
    n: Option<u32>, // default 1; # of completion choices

    #[validate(range(min = 0., max = 2.))]
    temperature: Option<f32>, // default 1

    #[validate(range(min = 0., max = 1.))]
    top_p: Option<f32>, // default NONE

    #[validate(range(min = -2., max = 2.))]
    frequency_penalty: Option<f32>, // default NONE

    #[validate(range(min = -2., max = 2.))]
    presence_penalty: Option<f32>, // default NONE

    // NOTE(valid): vals must be (-100, 100)
    logit_bias: Option<HashMap<Token, f32>>,

    #[serde(default)]
    logprobs: bool,

    #[validate(range(min = 0, max = 20))]
    top_logprobs: Option<u32>, // default NONE; requires logprobs

    #[serde(default)]
    response_format: Format,

    stop: Option<Stop<'a>>,

    tools: Option<Vec<Tool<'a>>>,

    tool_choice: Option<ToolChoice<'a>>, // default "auto"

    stream: Option<bool>, // default false
}

#[derive(Deserialize)]
#[serde(tag = "role")]
#[rustfmt::skip]
enum Message<'a> {
    #[serde(rename = "system")]
    System { content: &'a str, name: Option<&'a str> },

    #[serde(rename = "user")]
    User { content: &'a str, name: Option<&'a str> },

    #[serde(rename = "assistant")]
    Assistant { content: &'a str, name: Option<&'a str>, tool_calls: Option<Call<'a>> },

    #[serde(rename = "tool")]
    Tool { content: &'a str, tool_call_id: &'a str },
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Call<'a> {
    #[serde(rename = "function")]
    Function { id: &'a str, function: Function<'a> },
}

#[derive(Deserialize)]
struct Function<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Deserialize, Default)]
#[serde(tag = "type")]
enum Format {
    #[default]
    #[serde(rename = "text")]
    Text,

    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop<'a> {
    String(&'a str),

    On(Vec<&'a str>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Tool<'a> {
    Function {
        description: Option<&'a str>,
        name: &'a str,
        parameters: Option<Map<String, Value>>,
    },
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum ToolChoice<'a> {
    #[serde(rename = "function")]
    Function { name: &'a str },

    #[serde(untagged)]
    Tool(&'a str), // "none" | "auto" | "required"
}

type Completion = Result<Either<Json<Value>, EventStream<Receiver<Event>>>, Fail>;

#[post("/v1/chat/completions", format = "json", data = "<body>")]
pub async fn completions(api: &State<Api>, body: Json<CompletionReq<'_>>) -> Completion {
    let body = body.into_inner();

    body.validate()
        .map_err(|e| Json(serde_json::to_value(e).unwrap()))
        .map_err(|e| (Status::BadRequest, e))?;

    let api = api.inner();
    let mut tokens = vec![];

    for msg in body.messages {
        // TODO(templates): pull jinjas
        match msg {
            // hmmmdge... not super great on mistral variants
            Message::System { content, name } => {}

            // should be supported by p much everything?
            Message::User { content, name } => {}

            // this has tool_calls because it might trigger a tc
            Message::Assistant { content, name, tool_calls } => {}

            // this is how you give it the result of said tool call
            Message::Tool { content, tool_call_id } => {}
        }

        // let msg = format!("{}: {}\n", msg.role, msg.content);
        // api.tk.tokenize(&mut tokens, &msg, false, false);
    }

    if tokens.len() >= api.max_n {
        return Err(fail(Status::InsufficientStorage, "too many tokens :("));
    }

    let chain = Chain::sample();

    if body.stream.unwrap_or(false) {
        let (params, recv) = Params::streaming(chain);
        api.schedule(tokens, params).await?;

        Ok(Either::Right(EventStream::from(recv)))
    } else {
        let (params, recv) = Params::unary(chain);
        api.schedule(tokens, params).await?;

        let body = recv
            .await
            .map_err(|_| fail(Status::InternalServerError, "unknown"))?;

        Ok(Either::Left(body))
    }
}

// TODO: time 2 restructure
struct Params {
    tx: Either<oneshot::Sender<Json<Value>>, Sender<Event>>,
    chain: Chain<Sample>,
}

impl Params {
    fn unary(chain: Chain<Sample>) -> (Self, oneshot::Receiver<Json<Value>>) {
        let (tx, rx) = oneshot::channel();
        let tx = Either::Left(tx);
        let p = Self { tx, chain };

        (p, rx)
    }

    fn streaming(chain: Chain<Sample>) -> (Self, Receiver<Event>) {
        let (tx, rx) = mpsc::channel(32);
        let tx = Either::Right(tx);
        let p = Self { tx, chain };

        (p, rx)
    }

    fn sample(&mut self, ctx: &mut Context, logits: &mut Logits) -> Option<Token> {
        let token = self.chain.choose(ctx, logits);

        // TODO: actually send a response back or w/e

        if ctx.model().token_is_eog(token) {
            return None;
        }

        Some(token)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to load model")]
    LoadModel,

    #[error("failed to allocate context")]
    LoadContext,

    #[error("decode error: {0:?}")]
    Decode(#[from] llama_sys::DecodeError),
}

pub struct Api {
    queue: Sender<(Vec<Token>, Params)>,
    inner: Arc<Mutex<Inner>>,
    tk: Tokenizer,
    max_n: usize,
}

impl Api {
    pub fn load<P: AsRef<Path>>(path: P, m: ModelParams, c: ContextParams) -> Result<Self, Error> {
        let model = Model::load_from_file(path, m).ok_or(Error::LoadModel)?;
        let mut ctx = model.context(c).ok_or(Error::LoadContext)?;
        let tk = model.tokenizer();

        let (queue, rx) = mpsc::channel(32);

        let max_n = min(model.n_ctx_train(), ctx.kv_cache_n_cells());

        let root = ctx.kv_cache_unique_id();

        let inner = Arc::new(Mutex::new(Inner {
            ctx,
            n_batch: c.n_batch as usize,
            queue: rx,
            sched: vec![],
            cache: RadixKv::new(root),
        }));

        Ok(Self { queue, inner, tk, max_n })
    }

    async fn schedule(&self, tokens: Vec<Token>, params: Params) -> Result<JoinHandle<()>, Fail> {
        let mut jobs = self.queue.clone();

        jobs.send((tokens, params))
            .await
            .map_err(|_| fail(Status::InternalServerError, "failed to schedule span"))?;

        let inner = self.inner.clone();
        Ok(spawn_blocking(move || Inner::flush_all(inner)))
    }
}

struct Inner {
    ctx: Context,
    n_batch: usize,
    queue: Receiver<(Vec<Token>, Params)>,
    sched: Vec<(Vec<Token>, Params)>,
    cache: RadixKv,
}

impl Inner {
    fn flush_all(mu: Arc<Mutex<Self>>) {
        while !{
            let mut ex = mu.lock().unwrap();
            ex.flush().unwrap();
            ex.sched.is_empty()
        } {}
    }

    fn flush(&mut self) -> Result<(), Error> {
        while let Ok(Some(span)) = self.queue.try_next() {
            self.sched.push(span);
        }

        // decode a batch of pending tokens
        let bounds: Vec<_> = (self.sched.iter())
            .map(|(s, _)| s.len() - self.cache.ancestor(s).1)
            .collect();

        // TODO: we should distribute total cache size amongst all requests
        let budget = clamp_sum(self.n_batch, &bounds, |&n| n);

        for (i, q_alloc) in budget.into_iter().enumerate() {
            if q_alloc == 0 {
                continue;
            }

            let tokens = &self.sched[i].0;
            let (_, n_cached, _) = self.cache.ancestor(tokens);
            let len = (n_cached + q_alloc).min(tokens.len());

            println!(
                "{i}: n_cached={n_cached}, q_alloc={q_alloc}, len={}, taking={}",
                tokens.len(),
                len - n_cached,
            );

            self.cache.insert_cold(&mut self.ctx, &tokens[..len]);
        }

        let batch = self.cache.pending_batch();
        let evals = self.ctx.decode_tokens(&batch)?;

        let logits: Vec<_> = evals.into_iter().filter_map(|e| e.logits).collect();
        let act = vec![vec![]; logits.len()];

        self.cache.accept_batch(iter::zip(logits, act));
        self.cache.compact();

        // sample on any spans
        let mut gc = vec![];

        for (i, (span, params)) in self.sched.iter_mut().enumerate() {
            let (j, n_cached, _) = self.cache.ancestor(span);

            if n_cached < span.len() {
                continue;
            }

            // TODO: if no logits, do the thing
            let logits = match &mut self.cache[j].state {
                Kv::Hot(logits, _) => logits,
                s => panic!("{s:?}"),
            };

            if let Some(token) = params.sample(&mut self.ctx, logits) {
                span.push(token);
            } else {
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
