use std::ffi::{c_void, CString};
use std::path::Path;
use std::ptr::{self, copy_nonoverlapping};
use std::slice;
use std::sync::{Arc, Once};

use bitflags::bitflags;

#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
#[allow(rustdoc::bare_urls)]
pub mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    impl std::str::FromStr for llama_split_mode {
        type Err = &'static str;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "none" => Ok(llama_split_mode::LLAMA_SPLIT_MODE_NONE),
                "layer" => Ok(llama_split_mode::LLAMA_SPLIT_MODE_LAYER),
                "row" => Ok(llama_split_mode::LLAMA_SPLIT_MODE_ROW),

                _ => Err("expected 'none' | 'layer' | 'row'"),
            }
        }
    }
}

pub use sys::{
    llama_context_params as ContextParams, llama_grammar_element as GrammarElement,
    llama_model_params as ModelParams, llama_pos as Pos, llama_seq_id as SeqId,
    llama_token as Token,
};

pub mod control;
pub mod sampling;

use control::{ControlVector, TensorProbe};
use sampling::{Logit, Output};

/// As the name implies, disable logging.
///
/// # Safety
/// Probably not threadsafe.
pub unsafe fn disable_logging() {
    extern "C" fn noop(_level: sys::ggml_log_level, _text: *const i8, _user_data: *mut c_void) {}

    sys::llama_log_set(Some(noop), ptr::null_mut());
}

pub fn max_devices() -> usize {
    unsafe { sys::llama_max_devices() }
}

unsafe impl Sync for ContextParams {}

unsafe impl Send for ContextParams {}

impl Default for ContextParams {
    fn default() -> Self {
        unsafe { sys::llama_context_default_params() }
    }
}

impl ContextParams {
    /// RNG seed, -1 for random
    pub fn seed(mut self, seed: usize) -> Self {
        self.seed = seed as u32;
        self
    }

    /// Text context, 0 = from model
    pub fn n_ctx(mut self, n_ctx: usize) -> Self {
        self.n_ctx = n_ctx as u32;
        self
    }

    /// Logical maximum batch size.
    pub fn n_batch(mut self, n_batch: usize) -> Self {
        self.n_batch = n_batch as u32;
        self
    }

    /// Physical maximum batch size.
    pub fn n_ubatch(mut self, n_ubatch: usize) -> Self {
        self.n_ubatch = n_ubatch as u32;
        self
    }

    /// Number of parallel sequences (i.e. distinct states for recurrent models).
    pub fn n_seq_max(mut self, n_seq_max: usize) -> Self {
        self.n_seq_max = n_seq_max as u32;
        self
    }

    /// Number of threads to use for generation
    pub fn n_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = n_threads as u32;
        self
    }

    /// Number of threads to use for batch processing
    pub fn n_threads_batch(mut self, n_threads_batch: usize) -> Self {
        self.n_threads_batch = n_threads_batch as u32;
        self
    }

    /// RoPE scaling type, from `enum llama_rope_scaling_type`
    pub fn rope_scaling_type(mut self, rope_scaling_type: sys::llama_rope_scaling_type) -> Self {
        self.rope_scaling_type = rope_scaling_type;
        self
    }

    /// Whether to pool (sum) embedding results by sequence id (ignored if no pooling layer)
    pub fn pooling_type(mut self, pooling_type: sys::llama_pooling_type) -> Self {
        self.pooling_type = pooling_type;
        self
    }

    /// RoPE base frequency, 0 = from model
    pub fn rope_freq_base(mut self, rope_freq_base: f32) -> Self {
        self.rope_freq_base = rope_freq_base;
        self
    }

    /// RoPE frequency scaling factor, 0 = from model
    pub fn rope_freq_scale(mut self, rope_freq_scale: f32) -> Self {
        self.rope_freq_scale = rope_freq_scale;
        self
    }

    /// YaRN extrapolation mix factor, negative = from model
    pub fn yarn_ext_factor(mut self, yarn_ext_factor: f32) -> Self {
        self.yarn_ext_factor = yarn_ext_factor;
        self
    }

    /// YaRN magnitude scaling factor
    pub fn yarn_attn_factor(mut self, yarn_attn_factor: f32) -> Self {
        self.yarn_attn_factor = yarn_attn_factor;
        self
    }

    /// YaRN low correction dim
    pub fn yarn_beta_fast(mut self, yarn_beta_fast: f32) -> Self {
        self.yarn_beta_fast = yarn_beta_fast;
        self
    }

    /// YaRN high correction dim
    pub fn yarn_beta_slow(mut self, yarn_beta_slow: f32) -> Self {
        self.yarn_beta_slow = yarn_beta_slow;
        self
    }

    /// YaRN original context size
    pub fn yarn_orig_ctx(mut self, yarn_orig_ctx: usize) -> Self {
        self.yarn_orig_ctx = yarn_orig_ctx as u32;
        self
    }

    /// Defragment the KV cache if holes/size > thold, < 0 disabled (default)
    pub fn defrag_threshold(mut self, threshold: f32) -> Self {
        self.defrag_thold = threshold;
        self
    }

    unsafe fn cb_eval(mut self, cb_eval: sys::ggml_backend_sched_eval_callback) -> Self {
        self.cb_eval = cb_eval;
        self
    }

    unsafe fn cb_eval_user_data(mut self, user_data: *mut c_void) -> Self {
        self.cb_eval_user_data = user_data;
        self
    }

    /// Data type for K cache
    pub fn type_k(mut self, type_k: sys::ggml_type) -> Self {
        self.type_k = type_k;
        self
    }

    /// Data type for V cache
    pub fn type_v(mut self, type_v: sys::ggml_type) -> Self {
        self.type_v = type_v;
        self
    }

    #[deprecated(note = "set llama_batch.logits instead")]
    pub fn logits_all(mut self, logits_all: bool) -> Self {
        self.logits_all = logits_all;
        self
    }

    /// If true, extract embeddings (together with logits).
    pub fn embeddings(mut self, embeddings: bool) -> Self {
        self.embeddings = embeddings;
        self
    }

    /// Whether to offload the KQV ops (including the KV cache) to GPU
    pub fn offload_kqv(mut self, offload_kqv: bool) -> Self {
        self.offload_kqv = offload_kqv;
        self
    }

    /// Whether to use flash attention.
    pub fn flash_attn(mut self, flash_attn: bool) -> Self {
        self.flash_attn = flash_attn;
        self
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn abort_callback(mut self, cb: sys::ggml_abort_callback) -> Self {
        self.abort_callback = cb;
        self
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn abort_callback_data(mut self, data: *mut c_void) -> Self {
        self.abort_callback_data = data;
        self
    }
}

unsafe impl Sync for ModelParams {}

unsafe impl Send for ModelParams {}

impl Default for ModelParams {
    fn default() -> Self {
        unsafe { sys::llama_model_default_params() }
    }
}

impl ModelParams {
    pub fn offload(layers: usize) -> Self {
        Self {
            n_gpu_layers: layers as i32,
            use_mmap: false,
            ..Self::default()
        }
    }

    pub fn n_gpu_layers(mut self, n_gpu_layers: usize) -> Self {
        self.n_gpu_layers = n_gpu_layers as i32;
        self
    }

    pub fn split_mode(mut self, split_mode: sys::llama_split_mode) -> Self {
        self.split_mode = split_mode;
        self
    }

    pub fn main_gpu(mut self, main_gpu: usize) -> Self {
        self.main_gpu = main_gpu as i32;
        self
    }

    pub fn tensor_split(mut self, mut tensor_split: Vec<f32>) -> Self {
        let n = max_devices();
        assert!(tensor_split.len() <= n);

        if tensor_split.len() < n {
            tensor_split.resize(n, 0.);
        }

        let b = tensor_split.into_boxed_slice();
        let b = Box::leak::<'static>(b);

        self.tensor_split = b.as_ptr();
        self
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn progress_callback(
        mut self,
        progress_callback: sys::llama_progress_callback,
    ) -> Self {
        self.progress_callback = progress_callback;
        self
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn progress_callback_user_data(mut self, user_data: *mut c_void) -> Self {
        self.progress_callback_user_data = user_data;
        self
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn kv_overrides(mut self, overrides: *const sys::llama_model_kv_override) -> Self {
        self.kv_overrides = overrides;
        self
    }

    pub fn vocab_only(mut self, vocab_only: bool) -> Self {
        self.vocab_only = vocab_only;
        self
    }

    pub fn use_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    pub fn use_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = use_mlock;
        self
    }

    pub fn check_tensors(mut self, check_tensors: bool) -> Self {
        self.check_tensors = check_tensors;
        self
    }
}

#[derive(Clone)]
pub struct Tokenizer(Arc<Model>);

unsafe impl Sync for Tokenizer {}

unsafe impl Send for Tokenizer {}

impl Tokenizer {
    /// Detokenize `token` into the provided buffer, allocating if there isn't enough space to store
    /// the entire text fragment.
    ///
    /// Returns the number of bytes decoded.
    pub fn detokenize(&self, buf: &mut Vec<u8>, token: Token, special: bool) -> usize {
        self.0.detokenize(buf, token, special)
    }

    /// Tokenize `text` into the provided buffer, allocating if there isn't enough space to store
    /// all tokens.
    ///
    /// Returns the number of encoded tokens.
    pub fn tokenize(&self, buf: &mut Vec<Token>, text: &str, special: bool) -> usize {
        self.0.tokenize(buf, text, special)
    }

    /// Returns the vocabulary type.
    pub fn vocab_type(&self) -> sys::llama_vocab_type {
        self.0.vocab_type()
    }

    /// Returns the RoPE type.
    pub fn rope_type(&self) -> sys::llama_rope_type {
        self.0.rope_type()
    }

    /// Returns the model's vocabulary size.
    pub fn n_vocab(&self) -> i32 {
        self.0.n_vocab()
    }

    pub fn token_attr(&self, token: Token) -> Attr {
        self.0.token_attr(token)
    }

    /// Returns true if `token` is meant to halt generation (EOS, EOT, etc)
    pub fn token_is_eog(&self, token: Token) -> bool {
        self.0.token_is_eog(token)
    }

    /// Returns true if `token` is a control token.
    pub fn token_is_control(&self, token: Token) -> bool {
        self.0.token_is_control(token)
    }

    /// Returns the beginning of string token.
    pub fn token_bos(&self) -> Token {
        self.0.token_bos()
    }

    /// Returns the end of string token.
    pub fn token_eos(&self) -> Token {
        self.0.token_eos()
    }

    /// Returns the newline token.
    pub fn token_nl(&self) -> Token {
        self.0.token_nl()
    }

    pub fn add_bos_token(&self) -> Option<bool> {
        self.0.add_bos_token()
    }

    pub fn add_eos_token(&self) -> Option<bool> {
        self.0.add_eos_token()
    }

    /// Returns the beginning of infill prefix token (codellama).
    pub fn token_prefix(&self) -> Token {
        self.0.token_prefix()
    }

    /// Returns the beginning of infill middle token (codellama).
    pub fn token_middle(&self) -> Token {
        self.0.token_middle()
    }

    /// Returns the beginning of infill suffix token (codellama).
    pub fn token_suffix(&self) -> Token {
        self.0.token_suffix()
    }

    /// Returns the end of infill middle token (codellama).
    pub fn token_eot(&self) -> Token {
        self.0.token_eot()
    }
}

bitflags! {
    #[derive(Debug)]
    pub struct Attr: sys::llama_token_attr {
        const UNDEFINED    = sys::llama_token_attr_LLAMA_TOKEN_ATTR_UNDEFINED;
        const UNKNOWN      = sys::llama_token_attr_LLAMA_TOKEN_ATTR_UNKNOWN;
        const UNUSED       = sys::llama_token_attr_LLAMA_TOKEN_ATTR_UNUSED;
        const NORMAL       = sys::llama_token_attr_LLAMA_TOKEN_ATTR_NORMAL;
        const CONTROL      = sys::llama_token_attr_LLAMA_TOKEN_ATTR_CONTROL;
        const USER_DEFINED = sys::llama_token_attr_LLAMA_TOKEN_ATTR_USER_DEFINED;
        const BYTE         = sys::llama_token_attr_LLAMA_TOKEN_ATTR_BYTE;
        const NORMALIZED   = sys::llama_token_attr_LLAMA_TOKEN_ATTR_NORMALIZED;
        const LSTRIP       = sys::llama_token_attr_LLAMA_TOKEN_ATTR_LSTRIP;
        const RSTRIP       = sys::llama_token_attr_LLAMA_TOKEN_ATTR_RSTRIP;
        const SINGLE_WORD  = sys::llama_token_attr_LLAMA_TOKEN_ATTR_SINGLE_WORD;

        const _ = !0;
    }
}

pub struct Model(*mut sys::llama_model);

unsafe impl Send for Model {}

unsafe impl Sync for Model {}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { sys::llama_free_model(self.0) }
    }
}

impl Model {
    /// Load a model from the filesystem.
    pub fn load_from_file<P: AsRef<Path>>(path: P, p: ModelParams) -> Option<Arc<Self>> {
        ensure_init_backend();

        let path = CString::new(path.as_ref().to_str()?.as_bytes()).ok()?;

        let raw = checked(unsafe { sys::llama_load_model_from_file(path.as_ptr(), p) })?;

        Some(Arc::new(Self(raw)))
    }

    /// Get a tokenizer that is safe to use in multiple threads.
    pub fn tokenizer(self: &Arc<Self>) -> Tokenizer {
        Tokenizer(self.clone())
    }

    /// Allocate a new inference context.
    pub fn context(self: &Arc<Self>, p: ContextParams) -> Option<Context> {
        unsafe {
            let probe = Box::into_raw(Box::new(TensorProbe::new("l_out-")));

            let p = p
                .cb_eval(Some(control::tensor_probe_cb_eval))
                .cb_eval_user_data(probe as *mut c_void);

            let ctx = checked(sys::llama_new_context_with_model(self.0, p))?;

            let batch_tok = sys::llama_batch_init(p.n_batch as i32, 0, 32);
            let batch_emb = sys::llama_batch_init(p.n_batch as i32, self.n_embed(), 32);

            Some(Context {
                model: Arc::clone(self),
                seq_id: 0,
                batch_tok,
                batch_emb,
                ctx,
                probe,
            })
        }
    }

    /// Detokenize `token` into the provided buffer, allocating if there isn't enough space to store
    /// the entire text fragment.
    ///
    /// Returns the number of bytes decoded.
    pub fn detokenize(&self, buf: &mut Vec<u8>, token: Token, special: bool) -> usize {
        unsafe {
            loop {
                let len = buf.len();
                // SAFETY: ptr is at most 1 past end of the allocation (if cap == 0)
                let ptr = buf.as_mut_ptr().add(len) as *mut i8;
                let cap = (buf.capacity() - len) as i32;

                // SAFETY: ptr is valid for writes of up to cap bytes
                let n = sys::llama_token_to_piece(self.0, token, ptr, cap, special);

                // SAFETY: hope llama.cpp didn't lie
                if n >= 0 {
                    buf.set_len(len + n as usize);
                    break n as usize;
                }

                buf.reserve(-n as usize);
            }
        }
    }

    /// Tokenize `text` into the provided buffer, allocating if there isn't enough space to store
    /// all tokens.
    ///
    /// Returns the number of encoded tokens.
    pub fn tokenize(&self, buf: &mut Vec<Token>, text: &str, special: bool) -> usize {
        unsafe {
            loop {
                let len = buf.len();
                // SAFETY: ptr is at most 1 past end of the allocation (if cap == 0)
                let ptr = buf.as_mut_ptr().add(len);
                let cap = (buf.capacity() - len) as i32;

                // SAFETY: ptr is valid for writes of up to cap elements
                let n = sys::llama_tokenize(
                    self.0,
                    text.as_ptr() as *const i8,
                    text.len() as i32,
                    ptr,
                    cap,
                    false,
                    special,
                );

                // SAFETY: hope llama.cpp didn't lie
                if n >= 0 {
                    buf.set_len(len + n as usize);
                    break n as usize;
                }

                buf.reserve(-n as usize);
            }
        }
    }

    // TODO: meta_[val_str, count, key_by_index, val_str_by_index]

    /// Returns a ~short text description of the loaded model.
    pub fn desc(&self) -> CString {
        let mut buf: Vec<u8> = Vec::with_capacity(64);

        unsafe {
            loop {
                // SAFETY: buf is valid for writes of up to cap elements
                let ptr = buf.as_mut_ptr() as *mut i8;
                let n = 1 + sys::llama_model_desc(self.0, ptr, buf.capacity()) as usize;

                // SAFETY: hope llama.cpp didn't lie
                if n <= buf.capacity() {
                    buf.set_len(n);
                    break;
                }

                buf.reserve(n);
            }
        }

        CString::from_vec_with_nul(buf).unwrap()
    }

    /// Returns the vocabulary type.
    pub fn vocab_type(&self) -> sys::llama_vocab_type {
        unsafe { sys::llama_vocab_type(self.0) }
    }

    /// Returns the RoPE type.
    pub fn rope_type(&self) -> sys::llama_rope_type {
        unsafe { sys::llama_rope_type(self.0) }
    }

    /// Returns the model's vocabulary size.
    pub fn n_vocab(&self) -> i32 {
        unsafe { sys::llama_n_vocab(self.0) }
    }

    /// Returns the model's training context window size in tokens.
    pub fn n_ctx_train(&self) -> usize {
        unsafe { sys::llama_n_ctx_train(self.0) }
            .try_into()
            .unwrap()
    }

    /// Returns the model's embedding size.
    pub fn n_embed(&self) -> i32 {
        unsafe { sys::llama_n_embd(self.0) }
    }

    pub fn n_layer(&self) -> i32 {
        unsafe { sys::llama_n_layer(self.0) }
    }

    /// Returns the RoPE frequency scale used in training.
    pub fn rope_freq_scale_train(&self) -> f32 {
        unsafe { sys::llama_rope_freq_scale_train(self.0) }
    }

    /// Returns the model's size in parameters.
    pub fn n_params(&self) -> u64 {
        unsafe { sys::llama_model_n_params(self.0) }
    }

    /// Returns the model's size in bytes.
    pub fn size(&self) -> u64 {
        unsafe { sys::llama_model_size(self.0) }
    }

    pub fn token_attr(&self, token: Token) -> Attr {
        let bits = unsafe { sys::llama_token_get_attr(self.0, token) };

        Attr::from_bits(bits).unwrap()
    }

    /// Returns true if `token` is meant to halt generation (EOS, EOT, etc)
    pub fn token_is_eog(&self, token: Token) -> bool {
        unsafe { sys::llama_token_is_eog(self.0, token) }
    }

    /// Returns true if `token` is a control token.
    pub fn token_is_control(&self, token: Token) -> bool {
        unsafe { sys::llama_token_is_control(self.0, token) }
    }

    /// Returns the beginning of string token.
    pub fn token_bos(&self) -> Token {
        unsafe { sys::llama_token_bos(self.0) }
    }

    /// Returns the end of string token.
    pub fn token_eos(&self) -> Token {
        unsafe { sys::llama_token_eos(self.0) }
    }

    /// Returns the newline token.
    pub fn token_nl(&self) -> Token {
        unsafe { sys::llama_token_nl(self.0) }
    }

    pub fn add_bos_token(&self) -> Option<bool> {
        match unsafe { sys::llama_add_bos_token(self.0) } {
            0 => Some(false),
            1 => Some(true),
            -1 => None,
            _ => panic!(),
        }
    }

    pub fn add_eos_token(&self) -> Option<bool> {
        match unsafe { sys::llama_add_eos_token(self.0) } {
            0 => Some(false),
            1 => Some(true),
            -1 => None,
            _ => panic!(),
        }
    }

    /// Returns the beginning of infill prefix token (codellama).
    pub fn token_prefix(&self) -> Token {
        unsafe { sys::llama_token_prefix(self.0) }
    }

    /// Returns the beginning of infill middle token (codellama).
    pub fn token_middle(&self) -> Token {
        unsafe { sys::llama_token_middle(self.0) }
    }

    /// Returns the beginning of infill suffix token (codellama).
    pub fn token_suffix(&self) -> Token {
        unsafe { sys::llama_token_suffix(self.0) }
    }

    /// Returns the end of infill middle token (codellama).
    pub fn token_eot(&self) -> Token {
        unsafe { sys::llama_token_eot(self.0) }
    }
}

fn ensure_init_backend() {
    static INIT: Once = Once::new();

    INIT.call_once(|| unsafe { sys::llama_backend_init() });
}

fn checked<T>(ptr: *mut T) -> Option<*mut T> {
    if ptr.is_null() {
        return None;
    }
    Some(ptr)
}

pub struct Context {
    model: Arc<Model>,
    seq_id: SeqId,
    batch_tok: sys::llama_batch,
    batch_emb: sys::llama_batch,
    ctx: *mut sys::llama_context,
    probe: *mut TensorProbe,
}

unsafe impl Sync for Context {}

unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            sys::llama_batch_free(self.batch_tok);
            sys::llama_batch_free(self.batch_emb);
            sys::llama_free(self.ctx);
            drop(Box::from_raw(self.probe));
        }
    }
}

// kv cache
impl Context {
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Returns the pooling type.
    pub fn pooling_type(&self) -> sys::llama_pooling_type {
        unsafe { sys::llama_pooling_type(self.ctx) }
    }

    /// Returns the logical maximum batch size in tokens.
    pub fn n_batch(&self) -> u32 {
        unsafe { sys::llama_n_batch(self.ctx) }
    }

    /// Returns the physical maximum batch size in tokens.
    pub fn n_ubatch(&self) -> u32 {
        unsafe { sys::llama_n_ubatch(self.ctx) }
    }

    pub fn n_seq_max(&self) -> u32 {
        unsafe { sys::llama_n_seq_max(self.ctx) }
    }

    /// Returns the total number of available kv cells.
    pub fn kv_cache_n_cells(&self) -> usize {
        unsafe { sys::llama_n_ctx(self.ctx) }.try_into().unwrap()
    }

    /// Returns the number of free kv cells.
    pub fn kv_cache_free_cells(&self) -> usize {
        self.kv_cache_n_cells() - self.kv_cache_used_cells()
    }

    /// Returns the number of used kv cells.
    pub fn kv_cache_used_cells(&self) -> usize {
        unsafe { sys::llama_get_kv_cache_used_cells(self.ctx) }
            .try_into()
            .unwrap()
    }

    /// Returns a ~unique monotonically increasing sequence id, barring rollovers at i32 bounds.
    pub fn kv_cache_unique_id(&mut self) -> SeqId {
        let id = self.seq_id;
        self.seq_id += 1;
        id
    }

    /// Clear the kv cache.
    pub fn kv_cache_clear(&mut self) {
        unsafe { sys::llama_kv_cache_clear(self.ctx) }
    }

    /// Remove a range of tokens from a sequence.
    ///
    /// If `seq_id` is < 0, matches all sequences.
    pub fn kv_cache_seq_rm(&mut self, seq_id: SeqId, p0: Pos, p1: Pos) -> bool {
        unsafe { sys::llama_kv_cache_seq_rm(self.ctx, seq_id, p0, p1) }
    }

    /// Copy tokens in the provided range to another sequence. This doesn't allocate; kv cache entries
    /// can be referenced by multiple sequences.
    pub fn kv_cache_seq_cp(&mut self, src: SeqId, dst: SeqId, p0: Pos, p1: Pos) {
        unsafe { sys::llama_kv_cache_seq_cp(self.ctx, src, dst, p0, p1) }
    }

    /// Remove any tokens that don't belong to the provided sequence.
    pub fn kv_cache_seq_keep(&mut self, seq_id: SeqId) {
        unsafe { sys::llama_kv_cache_seq_keep(self.ctx, seq_id) }
    }

    /// Add relative position `delta` to a range of tokens in a sequence.
    pub fn kv_cache_seq_add(&mut self, seq_id: SeqId, p0: Pos, p1: Pos, delta: Pos) {
        unsafe { sys::llama_kv_cache_seq_add(self.ctx, seq_id, p0, p1, delta) }
    }

    /// Divide relative positions of a range of tokens by a constant factor.
    pub fn kv_cache_seq_div(&mut self, seq_id: SeqId, p0: Pos, p1: Pos, by: Pos) {
        unsafe { sys::llama_kv_cache_seq_div(self.ctx, seq_id, p0, p1, by) }
    }

    /// Returns the largest position present for the provided sequence.
    pub fn kv_cache_seq_pos_max(&self, seq_id: SeqId) -> Pos {
        unsafe { sys::llama_kv_cache_seq_pos_max(self.ctx, seq_id) }
    }

    /// Defragment the kv cache.
    pub fn kv_cache_defrag(&mut self) {
        unsafe { sys::llama_kv_cache_defrag(self.ctx) }
    }

    /// Apply any pending kv cache operations (K-shifts, defrag, etc).
    pub fn kv_cache_update(&mut self) {
        unsafe { sys::llama_kv_cache_update(self.ctx) }
    }
}

// state
impl Context {
    pub fn state_data_get(&mut self, dst: &mut [u8]) -> usize {
        unsafe {
            assert!(dst.len() >= sys::llama_get_state_size(self.ctx));
            sys::llama_state_get_data(self.ctx, dst.as_mut_ptr())
        }
    }

    pub fn state_data_put(&mut self, src: &[u8]) -> usize {
        unsafe {
            assert!(src.len() >= sys::llama_get_state_size(self.ctx));
            sys::llama_state_set_data(self.ctx, src.as_ptr())
        }
    }

    pub fn state_data_size(&self) -> usize {
        unsafe { sys::llama_get_state_size(self.ctx) }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Clone, Copy, Debug)]
pub enum Error {
    #[error("invalid input embedding length")]
    InputEmbedding,
    #[error("batch is empty")]
    Empty,
    #[error("overflowed n_seq")]
    TooManySeqs,
    #[error("llama.cpp: couldn't find kv slot")]
    Slot,
    #[error("llama.cpp: unknown error code {0}")]
    Code(i32),
}

// decoding
impl Context {
    pub fn decode(&mut self, batch: &[EvalReq<Token>]) -> Result<Vec<Option<Output<Logit>>>> {
        let mut evals = Vec::with_capacity(batch.len());

        for chunk in batch.chunks(self.n_batch() as usize) {
            unsafe {
                let batch = &mut self.batch_tok;
                batch.n_tokens = chunk.len() as i32;

                for (i, req) in chunk.iter().enumerate() {
                    let seq_ids = &req.seq_ids;

                    if req.seq_ids.len() > 32 {
                        return Err(Error::TooManySeqs);
                    }

                    // SAFETY: bounds checked above
                    copy_nonoverlapping(seq_ids.as_ptr(), *batch.seq_id.add(i), seq_ids.len());

                    // SAFETY: allocd in context() for n_batch, worst case these are uninit
                    *batch.n_seq_id.add(i) = req.seq_ids.len() as i32;
                    *batch.pos.add(i) = req.pos;
                    *batch.token.add(i) = req.input;
                    *batch.logits.add(i) = req.logits as i8;
                }

                match sys::llama_decode(self.ctx, *batch) {
                    e if e < 0 => return Err(Error::Code(e)),
                    e if e > 0 => return Err(Error::Slot),
                    _ => {}
                }

                evals.extend(self.evals(chunk));
            }
        }

        Ok(evals)
    }

    pub fn decode_embed<Q>(&mut self, batch: &[EvalReq<Q>]) -> Result<Vec<Option<Output<Logit>>>>
    where
        Q: AsRef<[f32]>,
    {
        let mut evals = Vec::with_capacity(batch.len());

        for chunk in batch.chunks(self.n_batch() as usize) {
            unsafe {
                let batch = &mut self.batch_emb;
                let n_embed = self.model.n_embed() as usize;

                batch.n_tokens = chunk.len() as i32;

                for (i, req) in chunk.iter().enumerate() {
                    let seq_ids = &req.seq_ids;

                    if seq_ids.len() > 32 {
                        return Err(Error::TooManySeqs);
                    }

                    // SAFETY: bounds checked above
                    copy_nonoverlapping(seq_ids.as_ptr(), *batch.seq_id.add(i), seq_ids.len());

                    let input = req.input.as_ref();
                    if input.len() != n_embed {
                        return Err(Error::InputEmbedding);
                    }

                    // SAFETY: bounds checked above
                    copy_nonoverlapping(input.as_ptr(), batch.embd.add(i * n_embed), n_embed);

                    // SAFETY: allocd in context() for n_batch, worst case these are uninit
                    *batch.n_seq_id.add(i) = req.seq_ids.len() as i32;
                    *batch.pos.add(i) = req.pos;
                    *batch.logits.add(i) = req.logits as i8;
                }

                match sys::llama_decode(self.ctx, *batch) {
                    e if e < 0 => return Err(Error::Code(e)),
                    e if e > 0 => return Err(Error::Slot),
                    _ => {}
                }

                evals.extend(self.evals(chunk));
            }
        }

        Ok(evals)
    }

    unsafe fn evals<'a, Q>(
        &'a self,
        reqs: &'a [EvalReq<Q>],
    ) -> impl Iterator<Item = Option<Output<Logit>>> + 'a {
        reqs.iter()
            .enumerate()
            .map(move |(i, req)| req.logits.then(|| self.get_logits_ith(i)))
    }

    unsafe fn get_logits_ith(&self, i: usize) -> Output<Logit> {
        let ptr = sys::llama_get_logits_ith(self.ctx, i as i32);
        let voc = self.model.n_vocab() as usize;
        let raw = slice::from_raw_parts(ptr, voc);

        Output::from_logits(raw, 128)
    }

    /// Set whether to use causal attention or not (default: ?).
    ///
    /// If set to true, the model will only attend to past tokens.
    pub fn set_causal_attn(&mut self, causal_attn: bool) {
        unsafe { sys::llama_set_causal_attn(self.ctx, causal_attn) }
    }

    /// Wait until all computations are finished.
    // NOTE: not really necessary; implicitly called by get_logits_ith
    pub fn synchronize(&mut self) {
        unsafe { sys::llama_synchronize(self.ctx) }
    }
}

#[derive(Clone, Debug)]
pub struct EvalReq<Input = Token> {
    pub seq_ids: Vec<SeqId>,
    pub pos: Pos,
    pub input: Input,
    pub logits: bool,
}

impl<Input> EvalReq<Input> {
    // TODO: rename to be the new new
    pub fn new(seq_ids: Vec<SeqId>, i: usize, input: Input, logits: bool) -> Self {
        Self {
            seq_ids,
            pos: i as Pos,
            input,
            logits,
        }
    }

    pub fn new_(seq_id: SeqId, i: usize, input: Input, logits: bool) -> Self {
        Self::new(vec![seq_id], i, input, logits)
    }
}

// tokenization
impl Context {
    /// Detokenize `token` into the provided buffer, allocating if there isn't enough space to store
    /// the entire text fragment.
    ///
    /// Returns the number of bytes decoded.
    pub fn detokenize(&self, buf: &mut Vec<u8>, token: Token, special: bool) -> usize {
        self.model.detokenize(buf, token, special)
    }

    /// Tokenize `text` into the provided buffer, allocating if there isn't enough space to store
    /// all tokens.
    ///
    /// Returns the number of encoded tokens.
    pub fn tokenize(&self, buf: &mut Vec<Token>, text: &str, special: bool) -> usize {
        self.model.tokenize(buf, text, special)
    }
}

// control vectors
impl Context {
    pub fn control_vector_apply(&mut self, v: Option<ControlVector<'_>>) -> Result<()> {
        let to_res = |code| match code {
            0 => Ok(()),
            e => Err(Error::Code(e)),
        };

        unsafe {
            let Some(v) = v else {
                let n_embed = self.model.n_embed();
                let r =
                    sys::llama_control_vector_apply(self.ctx, ptr::null(), 0, n_embed, -1, -1, -1);
                return to_res(r);
            };

            let r = sys::llama_control_vector_apply(
                self.ctx,
                v.data.as_ptr(),
                v.data.len(),
                v.n_embed,
                v.n_dirs,
                v.il_start,
                v.il_end,
            );

            to_res(r)
        }
    }

    pub fn probe_enable(&mut self) {
        unsafe { (*self.probe).enable() }
    }

    pub fn probe_disable(&mut self) {
        unsafe { (*self.probe).disable() }
    }

    // FIXME(perf): this is horrid, just alloc once and provide slices into the buffer
    pub fn drain_tensors(&mut self) -> impl Iterator<Item = Vec<f32>> + '_ {
        unsafe { (*self.probe).drain() }
    }
}
