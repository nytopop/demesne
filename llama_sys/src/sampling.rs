use std::cmp::{min, Ordering::*};
use std::mem::replace;

use rand::{thread_rng, Rng};

use super::Token;

/// A k-sparse output distribution optimized for high-throughput sampling.
#[derive(Clone, Debug)]
pub struct Output<S> {
    data: Vec<f32>,
    toks: Vec<Token>,
    mark: S,
}

#[derive(Debug, Clone)]
pub struct Logit;

#[derive(Debug, Clone)]
pub struct Probability;

#[derive(Debug, Clone)]
pub struct Distribution(f32);

impl<S> Output<S> {
    #[inline(always)]
    fn with<Q>(self, mark: Q) -> Output<Q> {
        Output {
            data: self.data,
            toks: self.toks,
            mark,
        }
    }
}

impl Output<Logit> {
    /// Produce an [Output] from a raw logit distribution, taking only the `top_k` logits.
    // TODO: add a noise parameter for injecting guassian noise
    pub fn from_logits(xs: &[f32], top_k: usize) -> Self {
        assert!(top_k <= xs.len());

        let mut o = Output {
            data: vec![0.; top_k],
            toks: vec![-1; top_k],
            mark: Logit,
        };

        for (id, x) in xs.iter().enumerate() {
            if *x <= o.data[top_k - 1] {
                continue;
            }

            let i = o
                .data
                .binary_search_by(|v| x.partial_cmp(v).unwrap())
                .unwrap_or_else(|i| i);

            if i < top_k - 1 {
                o.data.copy_within(i..top_k - 1, i + 1);
                o.toks.copy_within(i..top_k - 1, i + 1);
            }

            o.data[i] = *x;
            o.toks[i] = id as Token;
        }

        o
    }

    /// Fused numerically stable softmax with temperature.
    ///
    /// For details, see: https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html
    pub fn softmax(mut self, temp: f32) -> Output<Probability> {
        let temp = (temp + 1e-6).recip();
        let maxl = -temp * self.data[0];

        let exp_sum = self.data.iter_mut().fold(0., |ac, v| {
            *v = v.mul_add(temp, maxl).exp();
            ac + *v
        });

        let exp_sumr = exp_sum.recip();

        self.data.iter_mut().for_each(|v| *v *= exp_sumr);
        self.with(Probability)
    }
}

impl Output<Probability> {
    /// Fused min-p and p-step truncation.
    ///
    /// Implementation adapted from:
    /// - [P-Step Truncation Sampling](https://github.com/ggerganov/llama.cpp/pull/5675)
    /// - [Min P sampler implementation](https://github.com/ggerganov/llama.cpp/pull/3841)
    pub fn trunc_k(mut self, min_p: f32, min_s: f32, min_keep: usize) -> Self {
        let min_keep = min(min_keep, self.data.len());
        let min_p = min_p * self.data[0];
        let mut p = self.data[0];

        let k = (self.data.iter())
            .position(|&v| v < min_p || v < min_s * replace(&mut p, v))
            .unwrap_or(self.data.len())
            .max(min_keep);

        self.data.truncate(k);
        self.toks.truncate(k);
        self
    }

    /// Convert to a cumulatively weighted probability distribution for sampling.
    pub fn distribution(mut self) -> Output<Distribution> {
        let dist = self.data.iter_mut().fold(0., |ac, v| {
            *v += ac;
            *v
        });

        self.data.pop();
        self.with(Distribution(dist))
    }
}

impl Output<Distribution> {
    /// Sample a token from this distribution with the thread-local rng.
    pub fn sample(&self) -> Token {
        self.sample_rng(&mut thread_rng())
    }

    /// Sample a token from this distribution with the provided rng.
    pub fn sample_rng<R: Rng>(&self, rng: &mut R) -> Token {
        let w = rng.gen_range(0. ..self.mark.0);

        let i = self
            .data
            .binary_search_by(|&v| if v <= w { Less } else { Greater })
            .unwrap_err();

        self.toks[i]
    }
}
