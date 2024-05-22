use std::mem::ManuallyDrop;
use std::slice;

use smart_default::SmartDefault;

use super::sys::llama_grammar_element as GrammarElement;
use super::sys::llama_gretype::*;
use super::{sys, Context, Token};

// TODO: regex

#[derive(Debug)]
pub struct Logits {
    candidates: sys::llama_token_data_array,
    cap: usize,
}

unsafe impl Send for Logits {}

unsafe impl Sync for Logits {}

impl Drop for Logits {
    fn drop(&mut self) {
        unsafe {
            Vec::from_raw_parts(self.candidates.data, self.candidates.size, self.cap);
        }
    }
}

impl Clone for Logits {
    fn clone(&self) -> Self {
        let mut l = Self::from_vec(self.as_slice().to_vec());
        l.candidates.sorted = self.candidates.sorted;
        l
    }
}

impl AsRef<sys::llama_token_data_array> for Logits {
    fn as_ref(&self) -> &sys::llama_token_data_array {
        &self.candidates
    }
}

impl AsMut<sys::llama_token_data_array> for Logits {
    fn as_mut(&mut self) -> &mut sys::llama_token_data_array {
        &mut self.candidates
    }
}

impl Logits {
    pub fn from_vec(token_data: Vec<sys::llama_token_data>) -> Self {
        let mut m = ManuallyDrop::new(token_data);

        unsafe {
            // SAFETY: Logits is a Vec alloc in disguise
            Self::from_raw_parts(m.as_mut_ptr(), m.len(), m.capacity())
        }
    }

    unsafe fn from_raw_parts(data: *mut sys::llama_token_data, len: usize, cap: usize) -> Self {
        Self {
            candidates: sys::llama_token_data_array { data, size: len, sorted: false },
            cap,
        }
    }

    pub fn into_vec(self) -> Vec<sys::llama_token_data> {
        let l = ManuallyDrop::new(self);

        unsafe {
            // SAFETY: Logits is a Vec alloc in disguise
            Vec::from_raw_parts(l.candidates.data, l.candidates.size, l.cap)
        }
    }

    pub fn as_slice(&self) -> &[sys::llama_token_data] {
        unsafe { slice::from_raw_parts(self.candidates.data, self.candidates.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [sys::llama_token_data] {
        unsafe { slice::from_raw_parts_mut(self.candidates.data, self.candidates.size) }
    }
}

/// A terminal choice over logit distributions.
pub trait Choice: Send + 'static {
    fn choose(&mut self, ctx: &mut Context, logits: &mut Logits) -> Token;
}

impl Choice for Box<dyn Choice> {
    fn choose(&mut self, ctx: &mut Context, logits: &mut Logits) -> Token {
        (**self).choose(ctx, logits)
    }
}

pub struct Chain<C> {
    passes: Vec<Box<dyn CondPass>>,
    choice: C,
}

impl<C: Choice> Chain<C> {
    pub fn new(choice: C) -> Self {
        Self { passes: vec![], choice }
    }

    pub fn pass<P: CondPass>(mut self, pass: P) -> Self {
        self.passes.push(Box::new(pass));
        self
    }
}

impl Chain<Sample> {
    /// Sample from the logit distribution.
    pub fn sample() -> Self {
        Self::new(Sample)
    }
}

impl Chain<Greedy> {
    /// Always choose the token with the highest logit.
    pub fn greedy() -> Self {
        Self::new(Greedy)
    }
}

impl<C: Choice> Choice for Chain<C> {
    fn choose(&mut self, ctx: &mut Context, logits: &mut Logits) -> Token {
        for pass in self.passes.iter_mut() {
            pass.condition(ctx, logits);
        }

        let token = self.choice.choose(ctx, logits);

        for pass in self.passes.iter_mut() {
            pass.accept(ctx, token);
        }

        token
    }
}

#[derive(Default)]
pub struct Sample;

impl Choice for Sample {
    fn choose(&mut self, ctx: &mut Context, logits: &mut Logits) -> Token {
        ctx.sample_token(logits)
    }
}

#[derive(Default)]
pub struct Greedy;

impl Choice for Greedy {
    fn choose(&mut self, ctx: &mut Context, logits: &mut Logits) -> Token {
        ctx.sample_token_greedy(logits)
    }
}

#[derive(Clone, SmartDefault)]
pub struct MirostatV2 {
    #[default = 3.]
    pub tau: f32,

    #[default = 1.]
    pub eta: f32,

    #[default = 6.]
    pub mu: f32,
}

impl Choice for MirostatV2 {
    fn choose(&mut self, ctx: &mut Context, logits: &mut Logits) -> Token {
        ctx.sample_token_mirostat_v2(logits, self.tau, self.eta, &mut self.mu)
    }
}

/// A conditioning pass over logit distributions.
pub trait CondPass: Send + 'static {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits);

    fn accept(&mut self, ctx: &mut Context, token: Token) {
        let _ = (ctx, token);
    }
}

// TODO: pass that scales EOS based on length target to accurately steer for desired length
// #[derive(Clone, Copy)]
// pub struct LengthHint {}

#[derive(Clone, Copy, SmartDefault)]
pub struct TopK {
    #[default = 1]
    pub k: usize,

    #[default = 10]
    pub n: usize,
}

impl CondPass for TopK {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits) {
        ctx.sample_top_k(logits, self.k, self.n);
    }
}

#[derive(Clone, Copy, SmartDefault)]
pub struct TfsZ {
    #[default = 1.]
    pub z: f32,

    #[default = 10]
    pub n: usize,
}

impl CondPass for TfsZ {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits) {
        ctx.sample_tail_free(logits, self.z, self.n);
    }
}

#[derive(Clone, Copy, SmartDefault)]
pub struct TypP {
    #[default = 1.]
    pub p: f32,

    #[default = 10]
    pub n: usize,
}

impl CondPass for TypP {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits) {
        ctx.sample_typical(logits, self.p, self.n);
    }
}

#[derive(Clone, Copy, SmartDefault)]
pub struct TopP {
    #[default = 1.]
    pub p: f32,

    #[default = 10]
    pub n: usize,
}

impl CondPass for TopP {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits) {
        ctx.sample_top_p(logits, self.p, self.n);
    }
}

#[derive(Clone, Copy, SmartDefault)]
pub struct MinP {
    #[default = 0.004]
    pub p: f32,

    #[default = 10]
    pub n: usize,
}

impl CondPass for MinP {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits) {
        ctx.sample_min_p(logits, self.p, self.n);
    }
}

#[derive(Clone, Copy, SmartDefault)]
pub struct Temp(#[default = 1.] pub f32);

impl CondPass for Temp {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits) {
        ctx.sample_temp(logits, self.0);
    }
}

/// Dynamic temperature implementation described in the paper <https://arxiv.org/abs/2309.02772>.
#[derive(Clone, Copy, SmartDefault)]
pub struct Entropy {
    #[default = 0.6]
    pub min: f32,

    #[default = 1.9]
    pub max: f32,

    #[default = 1.]
    pub exp: f32,
}

impl CondPass for Entropy {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits) {
        ctx.sample_entropy(logits, self.min, self.max, self.exp);
    }
}

/* TODO: dry
pub struct Dry {
    multiplier: f32,
    base: f32,
    allowed_len: usize,
}

impl CondPass for Dry {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits) {
        todo!()
    }

    fn accept(&mut self, ctx: &mut Context, token: Token) {
        let _ = (ctx, token);
    }
}
*/

// TODO: implement grammar parsing of gbnf (ggml bnf)
// see: https://docs.rs/llama-cpp-2/latest/src/llama_cpp_2/grammar.rs.html#459-482
#[derive(Debug)]
pub struct Grammar(pub(super) *mut sys::llama_grammar);

unsafe impl Send for Grammar {}

unsafe impl Sync for Grammar {}

impl Drop for Grammar {
    fn drop(&mut self) {
        unsafe { sys::llama_grammar_free(self.0) }
    }
}

impl Clone for Grammar {
    fn clone(&self) -> Self {
        Self(unsafe { sys::llama_grammar_copy(self.0) })
    }
}

impl CondPass for Grammar {
    fn condition(&mut self, ctx: &mut Context, logits: &mut Logits) {
        ctx.sample_grammar(logits, self);
    }

    fn accept(&mut self, ctx: &mut Context, token: Token) {
        ctx.accept_grammar(self, token);
    }
}

impl Grammar {
    pub fn new(rules: &[GrammarElement], start_rule_index: usize) -> Option<Self> {
        let mut rb: Vec<_> = rules.iter().map(|r| r as *const _).collect();

        unsafe {
            super::checked(sys::llama_grammar_init(
                rb.as_mut_ptr(),
                rb.len(),
                start_rule_index,
            ))
            .map(Self)
        }
    }
}

impl GrammarElement {
    fn new(type_: sys::llama_gretype, value: u32) -> Self {
        Self { type_, value }
    }

    // end of rule definition
    pub fn end(value: u32) -> Self {
        Self::new(LLAMA_GRETYPE_END, value)
    }

    // start of alternate definition for rule
    pub fn alt(value: u32) -> Self {
        Self::new(LLAMA_GRETYPE_ALT, value)
    }

    // non-terminal element: reference to rule
    pub fn rule_ref(value: u32) -> Self {
        Self::new(LLAMA_GRETYPE_RULE_REF, value)
    }

    // terminal element: character (code point)
    pub fn char(value: u32) -> Self {
        Self::new(LLAMA_GRETYPE_CHAR, value)
    }

    // inverse char(s) ([^a], [^a-b] [^abc])
    pub fn char_not(value: u32) -> Self {
        Self::new(LLAMA_GRETYPE_CHAR_NOT, value)
    }

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to be an inclusive range ([a-z])
    pub fn char_rng_upper(value: u32) -> Self {
        Self::new(LLAMA_GRETYPE_CHAR_RNG_UPPER, value)
    }

    // modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char
    // to match ([ab], [a-zA])
    pub fn char_alt(value: u32) -> Self {
        Self::new(LLAMA_GRETYPE_CHAR_ALT, value)
    }
}
