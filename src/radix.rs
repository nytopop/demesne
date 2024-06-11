use std::cmp::{max, min};
use std::collections::{HashMap, VecDeque};
use std::iter;
use std::ops::{Index, IndexMut};

use llama_sys::sampling::{Logit, Output};
use llama_sys::{Context, EvalReq, SeqId, Token};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::Direction;

fn common_prefix<T: Eq>(a: &[T], b: &[T]) -> usize {
    iter::zip(a, b).take_while(|(a, b)| a == b).count()
}

#[derive(Default)]
pub struct RadixKv {
    g: StableDiGraph<Span, ()>,
    root: NodeIndex,
}

#[derive(Debug)]
pub struct Span {
    pub tag: SeqId,
    pub offset: usize,
    pub tokens: Vec<Token>,
    pub logits: Vec<Output<Logit>>,
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
            logits: vec![],
        });

        Self { g, root }
    }

    pub fn ancestor(&self, mut tokens: &[Token]) -> Ancestor {
        let mut a = Ancestor {
            i: self.root,
            n: 0,
            k: common_prefix(&self[self.root].tokens, tokens),
        };

        loop {
            a.n += a.k;

            if a.k != self[a.i].tokens.len() {
                break a;
            }

            tokens = &tokens[a.k..];

            if let Some((j, p)) = self
                .g
                .neighbors(a.i)
                .map(|j| (j, common_prefix(&self[j].tokens, tokens)))
                .filter(|(_, p)| *p > 0) // NOTE: for reasons
                .max_by_key(|(_, p)| *p)
            {
                a.i = j;
                a.k = p;
            } else {
                break a;
            }
        }
    }

    // returns Some iff a node was inserted; None is not an error
    pub fn insert(&mut self, ctx: &mut Context, a: Ancestor, span: &[Token]) -> Option<NodeIndex> {
        if a.n == span.len() {
            return None;
        }

        assert!(a.k <= self[a.i].tokens.len());

        if a.k < self[a.i].tokens.len() {
            self.split_off(ctx, a.i, a.k);
        }

        assert!(a.k == self[a.i].tokens.len());

        if self.g.neighbors(a.i).next().is_some() {
            let node = &self[a.i];

            let tag = ctx.kv_cache_unique_id();
            ctx.kv_cache_seq_cp(node.tag, tag, -1, a.n as i32);

            let next = Span {
                tag,
                offset: node.offset + node.tokens.len(),
                tokens: span[a.n..].to_vec(),
                logits: vec![],
            };

            let j = self.g.add_node(next);
            self.g.add_edge(a.i, j, ());

            Some(j)
        } else {
            let node = &mut self[a.i];
            node.tokens.extend_from_slice(&span[a.n..]);

            Some(a.i)
        }
    }

    fn split_off(&mut self, cx: &mut Context, src: NodeIndex, at: usize) -> (EdgeIndex, NodeIndex) {
        let node = &mut self[src];

        let tokens = node.tokens.split_off(at);

        let logits = (at <= node.logits.len())
            .then(|| node.logits.split_off(at))
            .unwrap_or_default();

        assert!(!tokens.is_empty());

        let next = Span {
            tag: cx.kv_cache_unique_id(),
            offset: node.offset + node.tokens.len(),
            tokens,
            logits,
        };

        let o = next.offset + next.tokens.len();
        cx.kv_cache_seq_cp(node.tag, next.tag, -1, o as i32);
        cx.kv_cache_seq_rm(node.tag, next.offset as i32, -1);

        let dst = self.g.add_node(next);

        let mut children = self.g.neighbors(src).detach();

        while let Some((e, c)) = children.next(&self.g) {
            self.g.remove_edge(e).unwrap();
            self.g.add_edge(dst, c, ());
        }

        (self.g.add_edge(src, dst, ()), dst)
    }

    pub fn prune<'a, I: IntoIterator<Item = &'a [Token]>>(&mut self, ctx: &mut Context, skips: I) {
        let size = self.size();
        let max_size = ctx.kv_cache_n_cells();

        if size <= max_size {
            return;
        }

        // find nodes along each skip, and how much we can prune from each node
        let mut tips: HashMap<_, usize> = HashMap::new();

        for span in skips {
            let a = self.ancestor(span);

            tips.entry(a.i)
                .and_modify(|k| *k = max(a.k, *k))
                .or_insert(a.k);
        }

        // get to deleting spans in reverse bfs order until left == 0
        let mut left = size - max_size;

        'outer: loop {
            for i in self.g.externals(Direction::Outgoing).collect::<Vec<_>>() {
                if left == 0 {
                    break 'outer;
                }

                let node = &mut self[i];

                // # of cells total
                let n = node.tokens.len();

                // starting offset of pruned range
                let k = tips.get(&i).copied().unwrap_or(0);

                if n == k {
                    continue;
                }

                // # of cells available to prune
                let p = n - k;

                // # of cells we're actually gonna prune
                let q = min(p, left);

                node.tokens.truncate(n - q);
                node.logits.truncate(n - q);

                if node.tokens.is_empty() {
                    ctx.kv_cache_seq_rm(node.tag, -1, -1);

                    if i != self.root {
                        self.g.remove_node(i).unwrap();
                    }
                } else {
                    ctx.kv_cache_seq_rm(node.tag, (node.offset + k + (p - q)) as i32, -1);
                }

                left -= q;
            }
        }

        ctx.kv_cache_defrag();
    }

    fn size(&self) -> usize {
        let mut size = 0;
        let mut fifo = VecDeque::new();
        fifo.push_back(self.root);

        while let Some(i) = fifo.pop_front() {
            size += self[i].tokens.len();
            fifo.extend(self.g.neighbors(i));
        }

        size
    }

    pub fn pending_batch(&self) -> Vec<EvalReq> {
        let mut reqs = vec![];
        let mut fifo = VecDeque::new();
        fifo.push_back(self.root);

        while let Some(i) = fifo.pop_front() {
            let mut tags = self.implied_tags(i);
            tags.sort();
            tags.dedup();

            let node = &self[i];

            if node.logits.len() < node.tokens.len() {
                let it = node
                    .tokens
                    .iter()
                    .enumerate()
                    .skip(node.logits.len())
                    .map(|(i, t)| EvalReq::new(tags.clone(), node.offset + i, *t, true));

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
            tags.push(self[i].tag);
            fifo.extend(self.g.neighbors(i));
        }

        tags
    }

    pub fn accept_batch<I: IntoIterator<Item = Output<Logit>>>(&mut self, iter: I) {
        let iter = &mut iter.into_iter();
        let mut fifo = VecDeque::new();
        fifo.push_back(self.root);

        // assumption: it's in the same order as when we scheduled pending_batch
        while let Some(i) = fifo.pop_front() {
            let node = &mut self[i];

            if node.logits.len() < node.tokens.len() {
                let n = node.tokens.len() - node.logits.len();
                node.logits.extend(iter.take(n));
            }

            fifo.extend(self.g.neighbors(i));
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ancestor {
    pub i: NodeIndex, // index of node
    pub n: usize,     // n cached
    pub k: usize,     // k cached on i
}

/// A less fat [RadixKv].
pub struct Radix {
    g: StableDiGraph<Vec<Token>, ()>,
    root: NodeIndex,
}

impl Radix {
    pub fn new() -> Self {
        let mut g = StableDiGraph::new();
        let root = g.add_node(vec![]);

        Self { g, root }
    }

    pub fn ancestor(&mut self, mut span: &[Token]) -> Ancestor {
        let mut a = Ancestor {
            i: self.root,
            n: 0,
            k: common_prefix(&self.g[self.root], span),
        };

        loop {
            a.n += a.k;

            if a.k != self.g[a.i].len() {
                break a;
            }

            span = &span[a.k..];

            if let Some((j, p)) = self
                .g
                .neighbors(a.i)
                .map(|j| (j, common_prefix(&self.g[j], span)))
                .filter(|(_, p)| *p > 0) // NOTE: for reasons
                .max_by_key(|(_, p)| *p)
            {
                a.i = j;
                a.k = p;
            } else {
                break a;
            }
        }
    }

    // returns Some iff a node was inserted; None is not an error
    pub fn insert(&mut self, a: Ancestor, span: &[Token]) -> Option<NodeIndex> {
        if a.n == span.len() {
            return None;
        }

        if a.k != self.g[a.i].len() {
            self.split_off(a.i, a.k);
        }

        let next = span[a.n..].to_vec();

        let j = self.g.add_node(next);
        self.g.add_edge(a.i, j, ());

        Some(j)
    }

    fn split_off(&mut self, src: NodeIndex, at: usize) -> (EdgeIndex, NodeIndex) {
        let node = &mut self.g[src];
        let next = node.split_off(at);
        let dst = self.g.add_node(next);

        let mut children = self.g.neighbors(src).detach();

        while let Some((e, c)) = children.next(&self.g) {
            self.g.remove_edge(e).unwrap();
            self.g.add_edge(dst, c, ());
        }

        (self.g.add_edge(src, dst, ()), dst)
    }
}
