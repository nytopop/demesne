# demesne
An OpenAI API compatible inference server built on llama.cpp that (hopefully) doesn't suck.

# another? why?
The aim of this project is reliably serving high-throughput inference on typical edge hardware. Lacking a current gen datacenter GPU shouldn't lock you out of critical performance features.

No tensor cores? No problem.

In demesne, we assume by default that system performance is limited somehow. Maybe you're on pascal, or gfx906, or some CPU/APU with DDR memory. In any case, we apply various techniques to speed things up.

First, continuous batching is practically a given if you want high-throughput from an LLM. We implement it as a matter of course.

Next, transparent prefix elision. Much like [sglang][0], we maintain a radix trie over in-flight completions and decode shared prefixes at most once. This variant however, is outwardly stateless and doesn't force clients to explicitly utilize it: simply send the entire context with each API call as per usual OpenAI API flow and get the performance gains with 0 effort.

[0]: https://lmsys.org/blog/2024-01-17-sglang/

Godspeed.

# status & feature set
Not ready for prod use. Needs polish & lots of stuff is as yet unimplemented

- [ ] broad accelerator support inherited from llama.cpp (nvidia, amd, metal, partial offload, etc)

- [ ] openai chat api compatibility

- [ ] native function calling

- [x] continuous batching

- [x] minalloc simd-aware sampling

- [x] transparent prefix elision caching

# install
```sh
git clone --recurse-submodules git@github.com:nytopop/demesne.git
cd demesne
```

And install with:

```sh
cargo install --path .
```

## cuda

```sh
cargo install --path . --features cuda
```

## rocm

```sh
cargo install --path . --features rocm
```

# usage
```sh
demesne --model <model.gguf> --api-key <key>
```

then point your llm app at the uri and go to town:

```sh
curl http://127.0.0.1:9090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <key>' \
  -d '{"model": "", "messages": [{"role": "user", "content": "What are the highest mountain peaks?"}]}'
```

# contribution
pls
