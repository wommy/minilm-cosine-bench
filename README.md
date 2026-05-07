# minilm-cosine-bench

Drop-in benchmarking harness for **MiniLM-L6-v2 i8 cosine-similarity kernels** — implement your own variant for your hardware, run `zig build run`, get correctness + perf.

The abstraction lets impls compete on different uarchs (Zen+ AVX2, Zen4+ AVX-VNNI, Ice Lake+ AVX-512, ARM SVE, etc) without anyone needing to clone the original project that motivated this.

## Quick start

```
git clone https://github.com/wommy/minilm-cosine-bench
cd minilm-cosine-bench
zig build run
```

Output: correctness check (max-diff vs scalar reference) + perf (ns/op + ops/s) for each registered impl. Always built `ReleaseFast` — Debug buries SIMD codegen.

Requires Zig 0.16+.

## Sample output (Ryzen 3400G, Zen+ AVX2)

```
MiniLM-L6-v2 i8 cosine kernel bench
  dim=384  corpus_rows=1000  iters=100  seed=42

[scalar]  target: any (reference)
  correctness max-diff vs scalar: 0.00e0
  perf: 282 ns/op  (3546099 ops/s)

[zen2]  target: Zen+ AVX2 (Ryzen 3400G)
  correctness max-diff vs scalar: 0.00e0
  perf: 108 ns/op  (9259259 ops/s)
```

zen2 is ~2.6× scalar via `@Vector(16, i32)` widening on AVX2.

## Adding your impl

1. Create `src/cosine_<yours>.zig` (e.g. `src/cosine_avx512.zig`)
2. Export a function matching `cosine.KernelFn` signature:
   ```zig
   pub fn cosineBatchI8_<yours>(
       query: [*]const i8, corpus: [*]const i8,
       n_rows: usize, out: [*]f32
   ) void
   ```
3. Register it in `src/main.zig` `IMPLS` table — uncomment the example or add similar
4. `zig build run` → see your impl benched alongside scalar + zen2

The contract: 384-dim i8 query and corpus rows, output is `1 - cosine_similarity` per row (cosine distance). Numerical guard: clamp norm denominators to `1e-12`.

## What's worth trying

If you're on Zen4+ / Ice Lake+ / Sapphire Rapids+:
- **AVX-VNNI** (`vpdpbusd`) — direct i8×u8 → i32 dot-product, 4× theoretical over `vpmaddwd` chain
- **AVX-512** (`vpdpbusd_512`) — same, wider
- **AMX** (Sapphire Rapids+) — tile-level matmul, overkill for single-query but compelling for batch>>1

If you're on ARM:
- **SVE / SVE2** — `sdot` instruction, scalable vector len, maps cleanly to dot-product
- **NEON `sdot`** — Cortex-A55+ baseline

## Why

Saga-21 of [glom_MR](https://github.com/wommy/gifts) standing on Zen+ Ryzen 3400G was getting 8M ops/s on `zen2` impl, but suspected modern uarchs could 2-4× via VPDPBUSD. Rather than guess, share a sandbox where anyone can drop in their best try.

## Sample integration (codemogger-shape demo)

A runnable demo of the kernel-in-context, no codemogger fork required:

```
zig build example
```

Outputs query→top-K ranking against a 100-chunk corpus. Source: `src/example.zig` — read it for the integration pattern (kernel call → distance array → sort by similarity → top-K). In real codemogger, `chunk_idx` maps to `(file_path, span_start, span_end)` from sqlite alongside the embeddings.

Swap kernels in `example.zig` (one line: `const kernel = cosine_<yours>.cosineBatchI8_<yours>;`) to see your impl in the consumption shape.


## License

CC0 — public domain. Fork it, adapt it, ship it.

## Origin

Sister to [`wommy/gifts`](https://github.com/wommy/gifts) — the doctrine corpus. This is the working-tool corpus.

## Context — the actual use case

This bench exists because of [**codemogger**](https://github.com/wommy/codemogger) — a semantic code-search tool that wants this kernel to be fast on whatever hardware it runs on. **You don't need to clone or read codemogger to help here**; this harness is the integration boundary.

Pipeline that consumes the kernel:

```
source code files
  ↓ chunk into spans (size depends on tokenizer; smallish)
  ↓ embed with MiniLM-L6-v2 q8                  ← inference layer (ONNX, see below)
  ↓ store as 384-dim i8 vectors in sqlite
  ↓ query: embed natural-language query → 384-dim i8
  ↓ cosine-similarity vs corpus                 ← THIS KERNEL
  ↓ top-K results
```

For an N-chunk codebase index, every query computes N cosine distances against the corpus. The kernel is the rank-time hot path. Currently ~9M ops/s on Zen+; modern uarchs (AVX-VNNI / AVX-512 / ARM SVE) likely 2-4× faster. Drop in your impl, the bench tells the truth.

## Two open questions (kernel layer + runtime layer)

**Q1 (this repo) — kernel optimization**: faster cosine on better silicon. Drop `cosine_<yours>.zig`, see speedup. Sandbox is here.

**Q2 (separate, no public repo yet) — ONNX runtime port-away**: codemogger currently uses vendored ORT 1.21.0 C API for the MiniLM q8 inference layer. We want off it entirely — pure-Zig path. Two candidates surveyed:

- **[abyesilyurt/minilm.c](https://github.com/abyesilyurt/minilm.c) port** to Zig fp32 (port size + kernel breakdown approximate from prior research; 3 hot kernels we identified: `QGemmU8S8`, `LayerNorm`, `Softmax`)
- **[ZantFoundation/Z-Ant](https://github.com/ZantFoundation/Z-Ant)** — pure-Zig neural-net deployment toolkit (microcontroller-shaped, but adoptable for desktop ONNX runtime use); vendor-instead-of-write path

Concrete pain we're hitting on Q2: ORT debug-from-source fills /tmp 8×; sccache wired to local S3 but rebuild not retried; q8 vs fp32 parity unvalidated; combined onnxruntime-node + @huggingface/transformers is ~280MB of platform binaries we want gone.

**Q1 vs Q2 are separate surfaces** — Q2 is the bigger / harder ask. If runtime-internals is more your jam than SIMD kernels, that's the conversation.

## License

CC0 — public domain. Fork it, adapt it, ship it.

## Origin

Sister to [`wommy/gifts`](https://github.com/wommy/gifts) — the doctrine corpus.
