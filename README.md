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

## License

CC0 — public domain. Fork it, adapt it, ship it.

## Origin

Sister to [`wommy/gifts`](https://github.com/wommy/gifts) — the doctrine corpus. This is the working-tool corpus.
