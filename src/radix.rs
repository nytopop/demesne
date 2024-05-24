use std::collections::VecDeque;
use std::ops::{Index, IndexMut};
use std::{iter, mem};

use llama_sys::sampling::Logits;
use llama_sys::{Context, EvalReq, SeqId, Token};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

#[derive(Default)]
pub struct RadixKv {
    g: StableDiGraph<Span, ()>,
    root: NodeIndex,
}

#[derive(Debug)]
pub struct Span {
    tag: SeqId,
    offset: usize,
    tokens: Vec<Token>,
    pub state: Kv,
}

#[derive(Debug)]
pub enum Kv {
    Hot(Logits, Vec<f32>), // decoded, all data
    Warm,                  // decoded, no logits
    Cold,                  // scheduled
}

impl Index<NodeIndex> for RadixKv {
    type Output = Span;

    fn index(&self, i: NodeIndex) -> &Self::Output {
        &self.g[i]
    }
}

impl IndexMut<NodeIndex> for RadixKv {
    fn index_mut(&mut self, i: NodeIndex) -> &mut Self::Output {
        &mut self.g[i]
    }
}

impl RadixKv {
    pub fn new(tag: SeqId) -> Self {
        let mut g = StableDiGraph::new();

        let root = g.add_node(Span {
            tag,
            offset: 0,
            tokens: vec![],
            state: Kv::Warm,
        });

        Self { g, root }
    }

    // (i, n_shared, n_shared_i)
    pub fn ancestor(&self, mut tokens: &[Token]) -> (NodeIndex, usize, usize) {
        fn common_prefix<T: Eq>(a: &[T], b: &[T]) -> usize {
            iter::zip(a, b).take_while(|(a, b)| a == b).count()
        }

        let mut i = self.root;
        let mut n = 0;
        let mut k = common_prefix(&self.g[i].tokens, tokens);

        loop {
            n += k;

            if k != self.g[i].tokens.len() {
                break (i, n, k);
            }

            tokens = &tokens[k..];

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

    // TODO: if logits are requested for a warm node, delete the last token and append a new node
    //
    // get_logits(warm):
    //
    // warm -> * (no logits)
    //      -> some other seq
    //      -> more
    //
    // hot -> warm -> some other seq
    //     -> *    -> more

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

        // NOTE: makes a trivial 1 token extension node
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
            state: Kv::Cold,
        };

        let j = self.g.add_node(next);
        self.g.add_edge(i, j, ());

        Some(j)
    }

    fn split_off(&mut self, src: NodeIndex, at: usize) -> (EdgeIndex, NodeIndex) {
        let node = &mut self.g[src];
        let next = node.tokens.split_off(at);

        let state = match &node.state {
            Kv::Hot(_, _) => Kv::Warm,
            Kv::Cold => Kv::Cold,
            Kv::Warm => Kv::Warm,
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

            if let Kv::Cold = node.state {
                if !node.tokens.is_empty() {
                    let n = node.tokens.len() - 1;

                    let it = node
                        .tokens
                        .iter()
                        .enumerate()
                        .map(|(i, t)| EvalReq::new(tags.clone(), node.offset + i, *t, i == n));

                    reqs.extend(it);
                }
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

    pub fn accept_batch<I: IntoIterator<Item = (Logits, Vec<f32>)>>(&mut self, iter: I) {
        let mut fifo = VecDeque::new();
        fifo.push_back(self.root);

        // assumption: it's in the same order as when we scheduled pending_batch
        'outer: for (logits, act) in iter.into_iter() {
            while let Some(i) = fifo.pop_front() {
                fifo.extend(self.g.neighbors(i));

                if let Kv::Cold = self.g[i].state {
                    if !self.g[i].tokens.is_empty() {
                        self.g[i].state = Kv::Hot(logits, act);
                        continue 'outer;
                    }

                    self.g[i].state = Kv::Warm;
                }
            }

            panic!("o no");
        }
    }

    // compact the graph by removing any trivial 1 token extension nodes
    pub fn compact(&mut self) {
        let mut fifo = VecDeque::new();
        fifo.push_back(self.root);

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
