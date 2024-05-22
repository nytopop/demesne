use std::ffi::{c_void, CStr};
use std::mem::size_of;

use super::sys;

// TODO(doc): pr, repr engineering, lw paper, google paper
pub struct ControlVector<'a> {
    pub(crate) data: &'a [f32], // [ 0 1 2 e ... e*d ][ ... ]
    pub(crate) n_embed: i32,
    pub(crate) n_dirs: i32,
    pub(crate) il_start: i32,
    pub(crate) il_end: i32,
}

impl<'a> AsRef<[f32]> for ControlVector<'a> {
    fn as_ref(&self) -> &[f32] {
        self.data
    }
}

impl<'a> ControlVector<'a> {
    pub fn new(data: &'a [f32], n_embed: i32, n_dirs: i32, il_start: i32, il_end: i32) -> Self {
        assert!(data.len() as i32 % n_embed == 0);
        assert!(data.len() as i32 % n_embed * n_dirs == 0);
        // actually has to be the whole net sized, weirdge af
        // assert!(data.len() as i32 / n_embed == il_end - il_start);

        Self {
            data,
            n_embed,
            n_dirs,
            il_start,
            il_end,
        }
    }
}

#[derive(Clone, Default)]
pub struct TensorProbe {
    prefix: &'static str,
    tensors: Vec<Vec<f32>>,
    enable: bool,
}

impl TensorProbe {
    pub fn new(prefix: &'static str) -> Self {
        Self {
            prefix,
            tensors: vec![],
            enable: false,
        }
    }

    pub fn enable(&mut self) {
        self.enable = true;
    }

    pub fn disable(&mut self) {
        self.enable = false;
    }

    pub fn drain(&mut self) -> impl Iterator<Item = Vec<f32>> + '_ {
        self.tensors.drain(..)
    }
}

pub(crate) unsafe extern "C" fn tensor_probe_cb_eval(
    tensor: *mut sys::ggml_tensor,
    ask: bool,
    ctx: *mut c_void,
) -> bool {
    let probe = ctx as *mut TensorProbe;

    let name = sys::ggml_get_name(tensor);
    assert!(!name.is_null());
    let name = CStr::from_ptr(name).to_string_lossy();

    if ask {
        // WARN: FIXME: use the actual prefix
        return (*probe).enable
            && (name.starts_with((*probe).prefix) || name.starts_with("ffn_inp-"));
    }

    if sys::ggml_nelements(tensor) == 0 {
        return true;
    }

    // WARN: FIXME: ^
    let il: usize = name
        .strip_prefix((*probe).prefix)
        .or_else(|| name.strip_prefix("ffn_inp-"))
        .unwrap()
        .parse()
        .unwrap();

    if il >= (*probe).tensors.len() {
        (*probe).tensors.resize(il + 1, vec![]);
    }

    let dst = &mut (*probe).tensors[il];

    let rows = sys::ggml_nrows(tensor) as usize;
    let embd = sys::ggml_nelements(tensor) as usize / rows;

    dst.reserve(rows * embd);

    let len = dst.len();
    let dst_p = dst.as_mut_ptr().add(len);

    let n = rows * embd * size_of::<f32>();
    sys::ggml_backend_tensor_get(tensor, dst_p as *mut c_void, 0, n);

    dst.set_len(len + (rows * embd));

    // NOTE: false cancels graph compute
    true
}
