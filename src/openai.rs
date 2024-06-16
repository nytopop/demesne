use std::borrow::Cow;
use std::cmp::min;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{iter, mem, str};

use async_openai::types::*;
use constant_time_eq::constant_time_eq;
use either::Either;
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender};
use futures::channel::{mpsc, oneshot as ones};
use futures::sink::SinkExt;
use llama_sys::sampling::{Logit, Output};
use llama_sys::{Context, ContextParams, Model, ModelParams, Token, Tokenizer};
use rocket::http::Status;
use rocket::request::{FromRequest, Outcome};
use rocket::response::stream::{Event, EventStream};
use rocket::serde::json::{Json, Value};
use rocket::serde::Serialize;
use rocket::tokio::task::{spawn_blocking, JoinHandle};
use rocket::{Catcher, Request, Responder, Route, State};
use uuid::Uuid;

use super::prompt::{Template, TemplateKind};
use super::radix::{Radix, RadixKv};

pub fn catchers() -> Vec<Catcher> {
    #[catch(default)]
    fn default(status: Status, _: &Request) -> (Status, Json<ApiError>) {
        (status, Json(ApiError::new("unknown")))
    }

    catchers![default]
}

#[derive(Responder, Debug)]
enum ErrorResp {
    #[response(status = 400, content_type = "json")]
    Req(Json<ApiError>),

    #[response(status = 401, content_type = "json")]
    Auth(Json<ApiError>),

    #[response(status = 500, content_type = "json")]
    Ise(Json<ApiError>),

    #[response(status = 501, content_type = "json")]
    Unimplemented(Json<ApiError>),
}

#[derive(Serialize, Default, Debug)]
struct ApiError {
    error: ApiErrorInner,
}

#[derive(Serialize, Default, Debug)]
struct ApiErrorInner {
    message: Cow<'static, str>,
    r#type: Option<Cow<'static, str>>,
    param: Option<Cow<'static, str>>,
    code: Option<Cow<'static, str>>,
}

impl ApiError {
    fn new<S: Into<Cow<'static, str>>>(message: S) -> Self {
        let error = ApiErrorInner {
            message: message.into(),
            ..Default::default()
        };

        Self { error }
    }

    fn message<S: Into<Cow<'static, str>>>(mut self, message: S) -> Self {
        self.error.message = message.into();
        self
    }

    fn r#type<S: Into<Cow<'static, str>>>(mut self, r#type: S) -> Self {
        self.error.r#type = Some(r#type.into());
        self
    }

    fn param<S: Into<Cow<'static, str>>>(mut self, param: S) -> Self {
        self.error.param = Some(param.into());
        self
    }

    fn code<S: Into<Cow<'static, str>>>(mut self, code: S) -> Self {
        self.error.code = Some(code.into());
        self
    }

    fn into_bad_req(self) -> ErrorResp {
        ErrorResp::Req(Json(self))
    }

    fn into_auth(self) -> ErrorResp {
        ErrorResp::Auth(Json(self))
    }

    fn into_ise(self) -> ErrorResp {
        ErrorResp::Ise(Json(self))
    }

    fn into_unimplemented(self) -> ErrorResp {
        ErrorResp::Unimplemented(Json(self))
    }
}

struct Authorized<'r> {
    #[allow(dead_code)]
    key: Option<&'r str>,
}

#[rocket::async_trait]
impl<'r> FromRequest<'r> for Authorized<'r> {
    type Error = ErrorResp;

    async fn from_request(req: &'r Request<'_>) -> Outcome<Self, Self::Error> {
        let api = req.rocket().state::<Api>().unwrap();

        let Some(srv_key) = api.api_key.as_deref() else {
            return Outcome::Success(Authorized { key: None });
        };

        let key = req
            .headers()
            .get("authorization")
            .next()
            .and_then(|key| key.strip_prefix("Bearer "));

        let Some(key) = key else {
            let err = ApiError::new("missing API key")
                .r#type("invalid_request_error")
                .into_auth();

            return Outcome::Error((Status::Unauthorized, err));
        };

        if !constant_time_eq(key.as_bytes(), srv_key.as_bytes()) {
            let err = ApiError::new("invalid API key")
                .r#type("invalid_request_error")
                .code("invalid_api_key")
                .into_auth();

            return Outcome::Error((Status::Unauthorized, err));
        }

        Outcome::Success(Authorized { key: Some(key) })
    }
}

pub fn routes() -> Vec<Route> {
    routes![chat_completions]
}

#[post("/v1/chat/completions", format = "json", data = "<req>")]
async fn chat_completions(
    key: Result<Authorized<'_>, ErrorResp>,
    api: &State<Api>,
    req: Json<CreateChatCompletionRequest>,
) -> Result<
    Either<Json<CreateChatCompletionResponse>, EventStream<UnboundedReceiver<Event>>>,
    ErrorResp,
> {
    let _ = key?;
    let r = req.into_inner();

    if r.messages.is_empty() {
        return Err(ApiError::new("try again with more messages")
            .param("messages")
            .into_bad_req());
    }

    let logit_bias = r.logit_bias.into_iter().flatten().map(|(k, v)| {
        let id = k
            .parse::<u32>()
            .map_err(|_| ApiError::new("invalid token in logit bias"))
            .map_err(|e| e.param("logit_bias").into_bad_req())?;

        let Value::Number(n) = v else {
            return Err(ApiError::new("bias value is not a number")
                .param("logit_bias")
                .into_bad_req());
        };

        let bias = n
            .as_f64()
            .ok_or_else(|| ApiError::new("bias value"))
            .map_err(|e| e.param("logit_bias").into_bad_req())?;

        if !(-100. ..=100.).contains(&bias) {
            return Err(ApiError::new("bias out of range")
                .param("logit_bias")
                .into_bad_req());
        }

        Ok((id as Token, bias as f32))
    });

    let logit_bias: Vec<_> = logit_bias.collect::<Result<_, ErrorResp>>()?;

    let logprobs = r.logprobs.unwrap_or(false);

    if r.top_logprobs.is_some_and(|_| !logprobs) {
        return Err(ApiError::new("top_logprobs but logprobs=false")
            .param("top_logprobs")
            .into_bad_req());
    }

    if r.top_logprobs.is_some_and(|n| n > 20) {
        return Err(ApiError::new("too many top_logprobs")
            .param("top_logprobs")
            .into_bad_req());
    }

    let top_logprobs: usize = r.top_logprobs.unwrap_or(0).into();

    if r.temperature.is_some_and(|v| !(0. ..=2.).contains(&v)) {
        return Err(ApiError::new("temperature out of range")
            .param("temperature")
            .into_bad_req());
    }

    if r.top_p.is_some_and(|v| !(0. ..=1.).contains(&v)) {
        return Err(ApiError::new("top_p out of range")
            .param("top_p")
            .into_bad_req());
    }

    if let Some(stop) = r.stop.as_ref() {
        if match stop {
            Stop::String(s) => s.is_empty(),
            Stop::StringArray(xs) => xs.iter().any(|s| s.is_empty()),
        } {
            return Err(ApiError::new("stop strings must be non-empty")
                .param("stop")
                .into_bad_req());
        }

        if matches!(stop, Stop::StringArray(xs) if xs.len() > 4) {
            return Err(ApiError::new("too many stop strings (max 4)")
                .param("stop")
                .into_bad_req());
        }
    }

    let api = api.inner();
    let mut span = vec![api.tk.token_bos()];

    // TODO: r.tools, r.tool_choice, r.parallel_tool_calls

    for msg in r.messages.iter() {
        match msg {
            ChatCompletionRequestMessage::System(m) => {
                api.tmpl.feed_sys(&api.tk, &mut span, &m.content);
            }

            ChatCompletionRequestMessage::User(m) => match &m.content {
                ChatCompletionRequestUserMessageContent::Text(content) => {
                    api.tmpl.feed_user(&api.tk, &mut span, content);
                }

                ChatCompletionRequestUserMessageContent::Array(_) => {
                    unimplemented!();
                }
            },

            ChatCompletionRequestMessage::Assistant(m) => match (&m.content, &m.tool_calls) {
                (Some(content), None) => {
                    api.tmpl.feed_gen_main(&api.tk, &mut span, content);
                }

                (Some(content), Some(calls)) => {
                    api.tmpl.feed_gen_main(&api.tk, &mut span, content);

                    // TODO: actually serialize calls
                    let tc = r#"[{"id": "1234", "name": "example", "arguments": {}}]"#;
                    api.tmpl.feed_gen_tool(&api.tk, &mut span, tc);
                }

                (None, Some(calls)) => {
                    // TODO: actually serialize calls
                    let tc = r#"[{"id": "4321", "name": "example", "arguments": {}}]"#;
                    api.tmpl.feed_gen_tool(&api.tk, &mut span, tc);
                }

                (None, None) => {}
            },

            ChatCompletionRequestMessage::Tool(m) => {
                api.tmpl.feed_result(&api.tk, &mut span, &m.content);
            }

            ChatCompletionRequestMessage::Function(_) => {
                return Err(ApiError::new("function call api is deprecated")
                    .code("wontfix")
                    .param("messages")
                    .into_unimplemented());
            }
        }
    }

    api.tmpl.prep_gen_choice(&mut span);

    // token limits
    let n_max = min(api.n_train, api.n_cells);

    if span.len() > n_max {
        return Err(ApiError::default()
            .message(format!("too many prompt tokens: {} > {n_max}", span.len()))
            .param("messages")
            .into_bad_req());
    }

    let k = api.n_cells - span.len();
    let n: usize = r.n.unwrap_or(1).into();
    let q = min(k / n, api.n_train - span.len());
    let n_infer = min(r.max_tokens.map(|v| v as usize).unwrap_or(q), q);

    let (tx, rx, stream_usage) = if r.stream.unwrap_or(false) {
        let (tx, rx) = mpsc::unbounded();
        let usage = r.stream_options.map(|o| o.include_usage).unwrap_or(false);

        (Either::Right(tx), Either::Right(rx), usage)
    } else {
        let (tx, rx) = ones::channel();
        (Either::Left(Some(tx)), Either::Left(rx), false)
    };

    let stop_matches = r.stop.into_iter().flat_map(|s| match s {
        Stop::String(s) => vec![StopAutomaton::new(s.into_bytes())],

        Stop::StringArray(xs) => xs
            .into_iter()
            .map(|s| StopAutomaton::new(s.into_bytes()))
            .collect(),
    });

    let tools = r.tools.unwrap_or_default();

    let tool_choice_default = if tools.is_empty() {
        ChatCompletionToolChoiceOption::None
    } else {
        ChatCompletionToolChoiceOption::Auto
    };

    let span = Span {
        temperature: r.temperature.unwrap_or(1.),
        min_s: 1. - r.top_p.unwrap_or(1.),
        logit_bias,
        n_infill: span.len(),
        n_infer,
        logprobs,
        top_logprobs,
        tool_choice: r.tool_choice.unwrap_or(tool_choice_default),

        span,
        text: vec![],
        prob: vec![],
        stop_automs: stop_matches.collect(),
        stop_reason: None,
        is_first: true,
        is_dirty: false,
        is_gen_c: true,
    };

    let p = Completion {
        id: Uuid::new_v4(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap(),
        stream_usage,

        n_infill: span.n_infill,
        n_actual: 0,

        spans: vec![span; n],
        tx,
    };

    api.schedule(p).await?;

    Ok(match rx {
        Either::Left(recv) => Either::Left(
            recv.await
                .map_err(|_| ApiError::new("cancelled"))
                .map_err(|e| e.into_ise())?,
        ),

        Either::Right(recv) => Either::Right(EventStream::from(recv)),
    })
}

struct Completion {
    id: Uuid,
    created: u32,
    stream_usage: bool,
    n_infill: usize,
    n_actual: usize,
    spans: Vec<Span>,
    tx: Either<Option<ones::Sender<Json<CreateChatCompletionResponse>>>, UnboundedSender<Event>>,
}

impl Completion {
    fn has_enabled(&self) -> bool {
        self.spans.iter().any(|f| !f.is_stopped())
    }

    fn enabled(&self) -> impl Iterator<Item = &[Token]> {
        self.spans
            .iter()
            .filter(|s| !s.is_stopped())
            .map(|s| s.span.as_slice())
    }

    fn accept_and_complete(&mut self) -> bool {
        let done = !self.has_enabled();

        let choices = self.spans.iter_mut().enumerate();

        if self.tx.is_left() && done {
            // TODO: tools
            #[allow(deprecated)]
            let choices = choices.map(|(i, s)| ChatChoice {
                index: i as u32,
                message: ChatCompletionResponseMessage {
                    content: Some(String::from_utf8_lossy(&s.text).into_owned()),
                    tool_calls: None,
                    role: Role::Assistant,
                    function_call: None,
                },
                finish_reason: s.stop_reason,
                logprobs: s.take_logprobs(),
            });

            let choices = choices.collect();

            // TODO: model, system_fingerprint
            let resp = CreateChatCompletionResponse {
                id: self.id.to_string(),
                choices,
                created: self.created,
                model: "model".to_owned(),
                system_fingerprint: None,
                object: "chat.completion".to_owned(),
                usage: Some(self.compute_usage()),
            };

            let tx = self.tx.as_mut().unwrap_left();
            let _r = tx.take().unwrap().send(Json(resp));

            return done;
        }

        if self.tx.is_right() {
            let choices = choices.filter(|(_, s)| s.is_dirty);

            #[allow(deprecated)]
            let choices = choices.map(|(i, s)| {
                let content = Some(String::from_utf8_lossy(&s.text).into_owned());
                s.text.clear();

                let role = mem::replace(&mut s.is_first, false).then_some(Role::Assistant);
                s.is_dirty = false;

                ChatChoiceStream {
                    index: i as u32,
                    delta: ChatCompletionStreamResponseDelta {
                        content,
                        tool_calls: None,
                        role,
                        function_call: None,
                    },
                    finish_reason: s.stop_reason,
                    logprobs: s.take_logprobs(),
                }
            });

            let mut resp = CreateChatCompletionStreamResponse {
                id: self.id.to_string(),
                choices: choices.collect(),
                created: self.created,
                model: "".to_owned(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_owned(),
                usage: self.stream_usage.then_some(None),
            };

            let tx = self.tx.as_ref().unwrap_right();

            if tx.unbounded_send(Event::json(&resp)).is_err() && !done {
                return true;
            }

            if done && self.stream_usage {
                resp.choices.clear();
                resp.usage = Some(Some(self.compute_usage()));
                let _ = tx.unbounded_send(Event::json(&resp));
            }

            if done {
                let _ = tx.unbounded_send(Event::data("[DONE]"));
            }
        }

        done
    }

    fn compute_usage(&self) -> CompletionUsage {
        CompletionUsage {
            prompt_tokens: self.n_infill as u32,

            completion_tokens: self
                .spans
                .iter()
                .map(|s| s.span.len() - s.n_infill)
                .map(|n| n as u32)
                .sum(),

            total_tokens: self.n_actual as u32,
        }
    }
}

#[derive(Clone)]
struct Span {
    // config
    temperature: f32,
    min_s: f32,
    logit_bias: Vec<(Token, f32)>,
    n_infill: usize,
    n_infer: usize,
    logprobs: bool,
    top_logprobs: usize,
    tool_choice: ChatCompletionToolChoiceOption,

    // state
    span: Vec<Token>,
    text: Vec<u8>,
    prob: Vec<ChatCompletionTokenLogprob>,
    stop_automs: Vec<StopAutomaton>,
    stop_reason: Option<FinishReason>,
    is_first: bool,
    is_dirty: bool,
    is_gen_c: bool,
}

impl Span {
    fn is_stopped(&self) -> bool {
        self.stop_reason.is_some()
    }

    fn accept(&mut self, tk: &Tokenizer, tmpl: &Template, logits: Output<Logit>) {
        assert!(!self.is_stopped(), "accept called on stopped span");

        // convert to probability space after applying logit bias
        let prob = logits
            // TODO: apply bias for tool if tools are disabled
            .apply_logit_bias(self.logit_bias.iter().copied())
            .softmax(self.temperature);

        // if enabled, grab the top logprobs before truncating & converting the distribution
        let top_logprobs = self
            .logprobs
            .then(|| prob.logprobs().take(20).collect::<Vec<_>>());

        // j is aligned w.r.t. top_logprobs, but may be out of bounds
        let (j, tok) = prob.trunc_k(0., self.min_s, 1).distribution().sample();

        if mem::replace(&mut self.is_gen_c, false) {
            match self.tool_choice {
                ChatCompletionToolChoiceOption::None => {
                    tmpl.prep_gen_main(&mut self.span);
                }

                ChatCompletionToolChoiceOption::Auto => {
                    // TODO: branch on probability of tc
                    unimplemented!()
                }

                ChatCompletionToolChoiceOption::Required => {
                    // TODO: we need structured generation to make this work properly
                    tmpl.prep_gen_tool(&mut self.span);
                }

                ChatCompletionToolChoiceOption::Named(_) => {
                    // TODO: prefill the named tool
                    unimplemented!()
                }
            }

            // TODO: recheck max token allowance
            self.n_infill = self.span.len();

            return;
        }

        let nb = tk.detokenize(&mut self.text, tok, false);
        let n = self.text.len();

        if tk.token_is_eog(tok) {
            self.stop_reason = Some(FinishReason::Stop);
        } else if self.span.len() - self.n_infill >= self.n_infer {
            self.stop_reason = Some(FinishReason::Length);
        } else if nb > 0 && self.feed_stop_sm(n - nb) {
            self.stop_reason = Some(FinishReason::Stop);
            self.text.truncate(n - nb);
            return;
        }

        self.span.push(tok);
        self.is_dirty = true;

        // finish computing logprobs now that we know which token was sampled
        if let Some(top_logprobs) = top_logprobs {
            let detok = |id| {
                let mut bytes = Vec::with_capacity(16);
                tk.detokenize(&mut bytes, id, false);
                let token = String::from_utf8_lossy(&bytes).into_owned();

                (token, Some(bytes))
            };

            let (token, bytes) = detok(tok);

            let tlp = ChatCompletionTokenLogprob {
                token,
                logprob: top_logprobs.get(j).map(|(_, lp)| *lp).unwrap_or(-9999.),
                bytes,
                top_logprobs: top_logprobs
                    .into_iter()
                    .take(self.top_logprobs)
                    .map(|(id, lp)| (detok(id), lp))
                    .map(|((token, bytes), logprob)| TopLogprobs { token, logprob, bytes })
                    .collect(),
            };

            self.prob.push(tlp);
        }
    }

    fn feed_stop_sm(&mut self, at: usize) -> bool {
        let buf = &self.text[at..];
        self.stop_automs.iter_mut().any(|a| a.feed_done(buf))
    }

    fn take_logprobs(&mut self) -> Option<ChatChoiceLogprobs> {
        let prob = mem::take(&mut self.prob);

        (!prob.is_empty())
            .then_some(prob)
            .map(|content| ChatChoiceLogprobs { content: Some(content) })
    }
}

#[derive(Clone)]
struct StopAutomaton {
    pat: Vec<u8>,
    pre: Vec<u8>,
}

impl StopAutomaton {
    fn new(pat: Vec<u8>) -> Self {
        Self { pat, pre: vec![] }
    }

    fn feed_done(&mut self, span: &[u8]) -> bool {
        for &c in span {
            if c == self.pat[self.pre.len()] {
                self.pre.push(c);
            } else {
                self.pre.clear();
            }

            if self.pat.len() == self.pre.len() {
                return true;
            }
        }

        false
    }
}

#[derive(thiserror::Error, Debug)]
pub enum LoadError {
    #[error("failed to load model")]
    LoadModel,

    #[error("failed to allocate context")]
    LoadContext,

    #[error("llama error: {0:?}")]
    Llama(#[from] llama_sys::Error),
}

pub struct ApiBuilder {
    path: PathBuf,
    vocab: TemplateKind,
    m_param: Option<ModelParams>,
    c_param: Option<ContextParams>,
    api_key: Option<String>,
}

impl ApiBuilder {
    pub fn new<P: Into<PathBuf>>(path: P, vocab: TemplateKind) -> Self {
        Self {
            path: path.into(),
            m_param: None,
            c_param: None,
            vocab,
            api_key: None,
        }
    }

    pub fn model_params(mut self, m: ModelParams) -> Self {
        self.m_param = Some(m);
        self
    }

    pub fn context_params(mut self, c: ContextParams) -> Self {
        self.c_param = Some(c);
        self
    }

    pub fn api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key;
        self
    }

    pub fn build(self) -> Result<Api, LoadError> {
        let m = self.m_param.unwrap_or_default();
        let model = Model::load_from_file(self.path, m).ok_or(LoadError::LoadModel)?;

        let c = self.c_param.unwrap_or_default();
        let mut ctx = model.context(c).ok_or(LoadError::LoadContext)?;
        let tk = model.tokenizer();

        let tmpl = Arc::new(Template::compile(&tk, self.vocab));

        let (queue, rx) = mpsc::channel(32);

        let n_train = model.n_ctx_train();
        let n_cells = ctx.kv_cache_n_cells();
        let root = ctx.kv_cache_unique_id();

        let inner = Arc::new(Mutex::new(Inner {
            ctx,
            tk: tk.clone(),
            tmpl: tmpl.clone(),
            n_batch: c.n_batch as usize,
            queue: rx,
            sched: vec![],
            cache: RadixKv::new(root),
        }));

        Ok(Api {
            queue,
            inner,
            tk,
            tmpl,
            n_train,
            n_cells,
            api_key: self.api_key,
        })
    }
}

pub struct Api {
    queue: mpsc::Sender<Completion>,
    inner: Arc<Mutex<Inner>>,
    tk: Tokenizer,
    tmpl: Arc<Template>,
    n_train: usize,
    n_cells: usize,
    api_key: Option<String>,
}

impl Api {
    /// Schedule an inflight completion req.
    async fn schedule(&self, params: Completion) -> Result<JoinHandle<()>, ErrorResp> {
        let mut jobs = self.queue.clone();

        jobs.send(params)
            .await
            .map_err(|_| ApiError::new("failed to schedule span"))
            .map_err(|e| e.into_ise())?;

        let inner = self.inner.clone();

        Ok(spawn_blocking(move || Inner::flush_all(inner).unwrap()))
    }
}

struct Inner {
    ctx: Context,
    tk: Tokenizer,
    tmpl: Arc<Template>,
    n_batch: usize,
    queue: mpsc::Receiver<Completion>,
    sched: Vec<Completion>,
    cache: RadixKv,
}

impl Inner {
    fn flush_all(mu: Arc<Mutex<Self>>) -> Result<(), llama_sys::Error> {
        // excludes to a single task, but serves all inflight reqs so contention isn't an issue
        while !{
            let mut ex = mu.lock().unwrap();
            ex.flush_bounded()?;
            ex.sched.is_empty()
        } {}

        Ok(())
    }

    // TODO: use tracing to record performance data

    fn flush_bounded(&mut self) -> Result<(), llama_sys::Error> {
        let t0 = std::time::Instant::now();

        // move any new requests to the back of scheduled queue
        while let Ok(Some(p)) = self.queue.try_next() {
            self.sched.push(p);
        }

        if self.sched.is_empty() {
            return Ok(());
        }

        // compute how much kv space may be allocated to each scheduled span
        let max = self.ctx.kv_cache_n_cells();
        let k = n_dedup_reqs_within(max, &self.sched);

        // slice to get span of spans that pack into available kv
        let sched = &mut self.sched[..k];

        let bounds: Vec<_> = sched
            .iter()
            .flat_map(|p| p.enabled())
            .map(|span| span.len() - self.cache.ancestor(span).n)
            .collect();

        let mut actual = Vec::with_capacity(sched.len());

        let it = sched
            .iter()
            .enumerate()
            .flat_map(|(i, p)| p.enabled().map(move |s| (i, s)));

        // distribute (inexact in the case of shared prefixes)
        for (m, (i, span)) in iter::zip(clamp_sum(self.n_batch, &bounds, |&n| n), it) {
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
        let nb = batch.len();

        if !batch.is_empty() {
            let evals = self.ctx.decode(&batch)?;

            self.cache
                .accept_batch(evals.into_iter().map(|o| o.unwrap()));
        }

        let mut gc = vec![];

        // dispatch logits to any running spans that are fully infilled
        for (i, p) in sched.iter_mut().enumerate() {
            let mut touched = false;
            let mut n_ready = 0;

            // TODO: remove hard dep
            for s in p.spans.iter_mut() {
                if s.is_stopped() {
                    n_ready += 1;
                    continue;
                }

                let a = self.cache.ancestor(&s.span);

                if a.n < s.span.len() {
                    continue;
                }

                // sanity check
                assert_eq!(a.n, s.span.len());
                assert_eq!(self.cache[a.i].tokens.len(), self.cache[a.i].logits.len(),);

                let logits = self.cache[a.i].logits[a.k - 1].clone();
                s.accept(&self.tk, &self.tmpl, logits);
                touched = true;
            }

            if (touched && p.accept_and_complete()) || n_ready == p.spans.len() {
                gc.push(i);
            }
        }

        let nd = gc.len();

        for i in gc.into_iter().rev() {
            self.sched.remove(i);
        }

        println!(
            "flush_bounded: {nb} toks, {nd} completed, {} queued, in {:?}",
            self.sched.len(),
            t0.elapsed(),
        );

        Ok(())
    }
}

fn n_dedup_reqs_within(mut left: usize, sched: &[Completion]) -> usize {
    let mut plan = Radix::new();
    let mut k = 0; // # of reqs that fit

    'outer: for p in sched {
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

    k
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
