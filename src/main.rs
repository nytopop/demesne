use std::net::IpAddr;
use std::path::PathBuf;

use llama_sys::{sys, ContextParams, ModelParams};
use structopt::StructOpt;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt::format::Format;

#[macro_use]
extern crate rocket;

mod openai;
mod prompt;
mod radix;

#[derive(Debug, StructOpt)]
#[structopt(name = "demesne")]
struct Opts {
    /// listen address
    #[structopt(short = "a", long = "addr", name = "addr", default_value = "0.0.0.0")]
    addr: IpAddr,

    /// listen port
    #[structopt(short = "p", long = "port", name = "port", default_value = "9090")]
    port: u16,

    /// path to some gguf model
    #[structopt(short = "m", long = "model", name = "path")]
    model_path: PathBuf,

    /// cache size in tokens
    #[structopt(short = "c", long = "cache", default_value = "32768")]
    n_cache: usize,

    /// batch size in tokens
    #[structopt(short = "b", long = "batch", default_value = "256")]
    n_batch: usize,

    /// number of layers to offload to gpu (-1 = offload all)
    #[structopt(short = "l", long = "layers", name = "layers", default_value = "-1")]
    n_gpu_layers: isize,

    /// how to split compute amongst gpus
    #[structopt(short = "s", long = "split-mode", name = "mode", default_value = "row")]
    split_mode: sys::llama_split_mode,

    #[structopt(short = "g", long = "main-gpu", name = "gpu", default_value = "0")]
    main_gpu: usize,

    /// proportion of the model (layers or rows) to offload to each gpu (pass N times)
    #[structopt(short = "t", long = "tensor-split", name = "split")]
    tensor_split: Option<Vec<f32>>,

    /// use flash attention
    #[structopt(short = "f", long = "flash-attn")]
    flash_attn: bool,

    /// memory map the loaded model (not very useful if offloaded)
    #[structopt(long = "mmap")]
    mmap: bool,

    /// api key to access the api
    #[structopt(short = "k", long = "api-key", name = "api-key")]
    api_key: Option<String>,

    /// override vocab kind (mis3 | l3 | phi3 | cml)
    #[structopt(short = "v", long = "vocab", name = "vocab", default_value = "cml")]
    vocab: prompt::Vocab,
}

#[rocket::main]
async fn main() -> Result<(), rocket::Error> {
    let fmt = Format::default()
        .compact()
        .with_line_number(false)
        .with_source_location(false);

    tracing_subscriber::fmt()
        .event_format(fmt)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let o = Opts::from_clap(
        &Opts::clap()
            .help_message("print help information")
            .version_message("print version information")
            .get_matches(),
    );

    let mut m = ModelParams::default()
        .n_gpu_layers(if o.n_gpu_layers < 0 {
            32768
        } else {
            o.n_gpu_layers as usize
        })
        .use_mmap(o.mmap)
        .check_tensors(true)
        .split_mode(o.split_mode)
        .main_gpu(o.main_gpu);

    if let Some(tensor_split) = o.tensor_split {
        m = m.tensor_split(tensor_split);
    }

    let n_cpu = num_cpus::get_physical();

    let c = ContextParams::default()
        .n_ctx(o.n_cache)
        .n_batch(o.n_batch)
        .n_ubatch(o.n_batch)
        .n_threads(n_cpu)
        .n_threads_batch(n_cpu)
        .flash_attn(o.flash_attn);

    let config = rocket::Config {
        address: o.addr,
        port: o.port,
        ..Default::default()
    };

    let api = openai::ApiBuilder::new(o.model_path, o.vocab)
        .model_params(m)
        .context_params(c)
        .api_key(o.api_key)
        .build()
        .unwrap();

    rocket::custom(config)
        .manage(api)
        .mount("/", openai::routes())
        .register("/", openai::catchers())
        .launch()
        .await?;

    Ok(())
}
