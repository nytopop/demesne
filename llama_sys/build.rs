use std::env;
use std::path::PathBuf;

fn main() {
    let mut conf = cmake::Config::new("llama.cpp");

    conf.build_target("llama")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_STATIC", "OFF")
        .define("LLAMA_NATIVE", "ON") // FIXME
        .define("LLAMA_LTO", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .very_verbose(true);

    // cpu features
    let on = |x| if x { "ON" } else { "OFF" };
    conf.define("LLAMA_AVX", on(cfg!(target_feature = "avx")));
    conf.define("LLAMA_AVX2", on(cfg!(target_feature = "avx2")));
    // TODO: avx512_*** and the cpu HBM stuff
    conf.define("LLAMA_FMA", on(cfg!(target_feature = "fma")));

    println!("cargo:rustc-link-lib=dylib=gomp");

    // rocm (amdgpu)
    #[cfg(feature = "rocm")]
    {
        conf.define("LLAMA_HIPBLAS", "ON");
        println!("cargo:rustc-link-search=/opt/rocm/lib");
        println!("cargo:rustc-link-lib=dylib=hipblas");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-lib=dylib=rocblas");
    }

    // cuda (nvidia)
    #[cfg(feature = "cuda")]
    {
        conf.define("LLAMA_CUDA", "ON");
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
    }

    // TODO: metal (apple silicon)

    let dst = conf.build();

    println!("cargo:rerun-if-changed=llama.cpp/llama.h");
    println!("cargo:rerun-if-changed=llama.cpp/llama.cpp");
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=stdc++");

    let bind = bindgen::builder()
        .header("llama.cpp/llama.h")
        .allowlist_function("llama_.*|ggml_.*|gguf_.*")
        .allowlist_type("llama_.*|ggml_.*|gguf_.*")
        .merge_extern_blocks(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .rustified_enum("llama_vocab_type")
        .rustified_enum("llama_rope_type")
        .rustified_enum("llama_token_type")
        // TODO: ftype
        .rustified_enum("llama_rope_scaling_type")
        .rustified_enum("llama_pooling_type")
        .rustified_enum("llama_split_mode")
        // TODO: kvoverridetype
        .rustified_enum("llama_gretype")
        .generate()
        .expect("failed to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bind.write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
