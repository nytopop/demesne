use std::cmp::min;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{iter, mem};

use either::Either;
use futures::channel::mpsc::{self, Receiver, Sender};
use futures::sink::SinkExt;
use llama_sys::sampling::Logits;
use llama_sys::{Context, ContextParams, EvalReq, Model, ModelParams, SeqId, Token, Tokenizer};
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use rocket::http::Status;
use rocket::response::stream::{Event, EventStream};
use rocket::response::Responder;
use rocket::serde::json::{json, Json, Value};
use rocket::serde::Deserialize;
use rocket::tokio::task::{spawn_blocking, JoinHandle};
use rocket::{Catcher, Request, Route, State};
use serde_json::Map;

pub fn catchers() -> Vec<Catcher> {
    #[catch(default)]
    fn default(status: Status, _: &Request) -> Fail {
        fail(status, "unknown")
    }

    catchers![default]
}

pub fn routes() -> Vec<Route> {
    routes![completions, speech, transcriptions]
}

type Fail = (Status, Json<Value>);

fn fail(status: Status, reason: &str) -> Fail {
    (status, Json(json! {{"reason": reason}}))
}

#[derive(Deserialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

type Completion = Either<Json<Value>, EventStream<Receiver<Event>>>;

#[post("/v1/chat/completions", format = "json", data = "<req>")]
async fn completions(api: &State<Api>, req: Json<Map<String, Value>>) -> Result<Completion, Fail> {
    let req = req.into_inner();

    let api = api.inner();
    let mut tokens = vec![];

    // TODO(templates): pull jinjas
    for msg in from_json_key::<Vec<Message>>(&req, "messages")? {
        let msg = format!("{}: {}\n", msg.role, msg.content);
        api.tk.tokenize(&mut tokens, &msg, false, false);
    }

    if tokens.len() >= api.max_n {
        return Err(fail(Status::InsufficientStorage, "too many tokens :("));
    }

    let params = Params { tx: Either::Left(()) };
    api.schedule(tokens, params).await?;

    // TODO: streaming vs not streaming

    Ok(Either::Left(Json(json! {{"output": "woo"}})))
}

fn from_json_key<'de, T: Deserialize<'de>>(r: &'de Map<String, Value>, k: &str) -> Result<T, Fail> {
    let val = r.get(k).ok_or_else(|| {
        let err = format!("missing {k:?} in request");
        fail(Status::BadRequest, &err)
    })?;

    T::deserialize(val).map_err(|e| {
        let err = format!("invalid {k:?} in request: {e:?}");
        fail(Status::BadRequest, &err)
    })
}

// TODO: stop conds and all the other crap
struct Params {
    tx: Either<(), ()>,
}

#[derive(Responder)]
enum Speech {} // stt

#[post("/v1/audio/speech", format = "json", data = "<req>")]
async fn speech(api: &State<Api>, req: Json<Value>) -> Result<Speech, Fail> {
    let _ = (api, req);
    unimplemented!()
}

#[derive(Responder)]
enum Transcription {} // tts

#[post("/v1/audio/transcriptions", format = "json", data = "<req>")]
async fn transcriptions(api: &State<Api>, req: Json<Value>) -> Result<Transcription, Fail> {
    let _ = (api, req);
    unimplemented!()
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

        let bounds: Vec<_> = (self.sched.iter())
            .map(|(s, _)| (s.len(), self.cache.ancestor(s).1))
            .collect();

        // TODO: may need to remove this to implement limits
        let budget = clamp_sum(self.n_batch, &bounds, |&(nt, nc)| nt - nc);
        let mut gc = vec![];

        for (j, ((n_tokens, n_cached), k_alloc)) in iter::zip(bounds, budget).enumerate() {
            let (tokens, params) = &mut self.sched[j];

            if n_tokens == n_cached {
                let logits = self.cache.logits_mut(tokens).unwrap();
                // TODO: sampler + decision logic there
                // TODO: make sure it extends by vec and gets full history
                // we're just gonna ignore that it bumps batch sizes (llama_sys handles it np)
                gc.push(j);
                continue;
            }

            if k_alloc == 0 {
                continue;
            }

            let len = n_cached + k_alloc;
            // TODO: use ret val to track stats
            self.cache.insert_cold(&mut self.ctx, &tokens[..len]);
        }

        // TODO: generate the set of prefixes that cannot be deleted (it's just self.sched)

        for j in gc.into_iter().rev() {
            self.sched.remove(j);
        }

        let batch = self.cache.pending_batch();
        // TODO: we need to handle the no kv slot case properlike
        let evals = self.ctx.decode_tokens(&batch)?;

        // TODO: activations
        let logits: Vec<_> = evals.into_iter().filter_map(|e| e.logits).collect();
        let act = vec![vec![]; logits.len()];

        self.cache.accept_batch(iter::zip(logits, act));

        println!(
            "radix_kv: {:?}",
            (self.cache.g.edge_count(), self.cache.g.node_count())
        );

        Ok(())
    }
}

#[derive(Default)]
struct RadixKv {
    g: DiGraph<Span, ()>,
    root: NodeIndex,
}

#[derive(Debug)]
struct Span {
    tag: SeqId,
    offset: usize,
    tokens: Vec<Token>,
    state: Cells,
}

#[derive(Debug)]
enum Cells {
    Hot(Logits, Vec<f32>), // decoded, all data
    Warm,                  // decoded, no logits
    Cold,                  // scheduled
}

impl RadixKv {
    fn new(tag: SeqId) -> Self {
        let mut g = DiGraph::new();

        let root = g.add_node(Span {
            tag,
            offset: 0,
            tokens: vec![],
            state: Cells::Warm,
        });

        Self { g, root }
    }

    // (i, n_shared, n_shared_i)
    pub fn ancestor(&self, mut tokens: &[Token]) -> (NodeIndex, usize, usize) {
        let (mut i, mut n, mut k) = (self.root, 0, 0);

        loop {
            n += k;

            if k != self.g[i].tokens.len() {
                break (i, n, k);
            }

            tokens = &tokens[k..];

            fn common_prefix<T: Eq>(a: &[T], b: &[T]) -> usize {
                iter::zip(a, b).take_while(|(a, b)| a == b).count()
            }

            if let Some((j, p)) = self
                .g
                .neighbors(i)
                .map(|j| (j, common_prefix(&self.g[j].tokens, tokens)))
                .max_by_key(|(_, p)| *p)
            {
                i = j;
                k = p;
            } else {
                break (i, n, k);
            }
        }
    }

    pub fn logits_mut(&mut self, tokens: &[Token]) -> Option<&mut Logits> {
        let (i, _, _) = self.ancestor(tokens);

        match &mut self.g[i].state {
            Cells::Hot(logits, _) => Some(logits),
            _ => None,
        }
    }

    // returns Some iff a node was inserted; None is not an error
    pub fn insert_cold(&mut self, ctx: &mut Context, tokens: &[Token]) -> Option<NodeIndex> {
        let (i, n, k) = self.ancestor(tokens);

        if tokens.len() == n {
            return None;
        }

        if k != self.g[i].tokens.len() {
            self.split_off(i, k);
        }

        let node = &self.g[i];

        let tag = if self.g.neighbors(i).next().is_some() {
            let tag = ctx.kv_cache_unique_id();
            ctx.kv_cache_seq_cp(node.tag, tag, ..n as i32);
            tag
        } else {
            node.tag
        };

        let next = Span {
            tag,
            offset: node.offset + node.tokens.len(),
            tokens: tokens[n..].to_vec(),
            state: Cells::Cold,
        };

        let j = self.g.add_node(next);
        self.g.add_edge(i, j, ());

        Some(j)
    }

    fn split_off(&mut self, src: NodeIndex, at: usize) -> (EdgeIndex, NodeIndex) {
        let node = &mut self.g[src];
        let next = node.tokens.split_off(at);

        let state = match &node.state {
            Cells::Hot(_, _) => Cells::Warm,
            Cells::Cold => Cells::Cold,
            Cells::Warm => Cells::Warm,
        };

        let next = Span {
            tag: node.tag,
            offset: node.offset + node.tokens.len(),
            tokens: next,
            state: mem::replace(&mut node.state, state),
        };

        let dst = self.g.add_node(next);

        let mut children = self.g.neighbors(src).detach();

        while let Some((e, c)) = children.next(&self.g) {
            self.g.remove_edge(e).unwrap();
            self.g.add_edge(dst, c, ());
        }

        (self.g.add_edge(src, dst, ()), dst)
    }

    pub fn pending_batch(&self) -> Vec<EvalReq> {
        let mut reqs = vec![];
        let mut fifo = VecDeque::new();
        fifo.push_back(self.root);

        while let Some(i) = fifo.pop_front() {
            let tags = self.implied_tags(i);
            let node = &self.g[i];

            if let Cells::Cold = node.state {
                let n = node.tokens.len() - 1;

                let it = node
                    .tokens
                    .iter()
                    .enumerate()
                    .map(|(i, t)| EvalReq::new(tags.clone(), node.offset + i, *t, i == n));

                reqs.extend(it);
            }

            fifo.extend(self.g.neighbors(i));
        }

        reqs
    }

    fn implied_tags(&self, i: NodeIndex) -> Vec<SeqId> {
        let mut tags = vec![];
        let mut fifo = VecDeque::new();
        fifo.push_back(i);

        while let Some(i) = fifo.pop_front() {
            tags.push(self.g[i].tag);
            fifo.extend(self.g.neighbors(i));
        }

        tags
    }

    //pub fn accept_batch(&mut self, logits: Vec<Logits>, act: Vec<Vec<f32>>) {
    pub fn accept_batch<I: IntoIterator<Item = (Logits, Vec<f32>)>>(&mut self, iter: I) {
        let mut fifo = VecDeque::new();
        fifo.push_back(self.root);

        // assumption: it's in the same order as when we scheduled pending_batch
        'outer: for (logits, act) in iter.into_iter() {
            while let Some(i) = fifo.pop_front() {
                fifo.extend(self.g.neighbors(i));

                if let Cells::Cold = self.g[i].state {
                    self.g[i].state = Cells::Hot(logits, act);
                    continue 'outer;
                }
            }

            panic!("o no");
        }

        fifo.clear();
        fifo.push_back(self.root);

        // compact the graph by removing any trivial 1 token extension nodes
        while let Some(i) = fifo.pop_front() {
            let mut it = self.g.neighbors(i);

            if let (Some(c), None) = (it.next(), it.next()) {
                if self.g[i].tag == self.g[c].tag {
                    let mut children = self.g.neighbors(c).detach();

                    while let Some((e, cc)) = children.next(&self.g) {
                        self.g.remove_edge(e).unwrap();
                        self.g.add_edge(i, cc, ());
                    }

                    let mut src = self.g.remove_node(c).unwrap();
                    let dst = &mut self.g[i];

                    dst.tokens.append(&mut src.tokens);
                    dst.state = src.state;
                }
            }

            fifo.extend(self.g.neighbors(i));
        }
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
