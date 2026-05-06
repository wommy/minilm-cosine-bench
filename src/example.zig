// Codemogger-shaped usage example: query → cosine kernel → top-K ranking.
//
// Illustrative — in real codemogger, embeddings come from MiniLM-L6-v2 q8 inference
// over source-code chunks (loaded from sqlite); the kernel runs at search time to
// rank query→corpus similarity. This file shows the integration shape without
// requiring you to clone codemogger.
//
// Run: `zig build example`

const std = @import("std");
const cosine = @import("cosine.zig");
const cosine_zen2 = @import("cosine_zen2.zig");
const data = @import("data.zig");

const SEED: u64 = 1337;
const N_CHUNKS: usize = 100; // tiny demo corpus
const TOP_K: usize = 5;

const SortCtx = struct {
    d: []f32,
    pub fn lt(self: @This(), a: usize, b: usize) bool {
        return self.d[a] < self.d[b];
    }
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Mock data — real codemogger wires:
    //   query: result of `MiniLM-L6-v2.embed("find UDP socket setup")` then quantized i8
    //   corpus: pre-embedded source-code chunks from sqlite (one row per (file_path, span_start, span_end))
    const query = try data.generateQuery(allocator, SEED);
    defer allocator.free(query);
    const corpus = try data.generateCorpus(allocator, N_CHUNKS, SEED + 1);
    defer allocator.free(corpus);

    // Swap kernel → coworker drops their impl here:
    const kernel: cosine.KernelFn = cosine_zen2.cosineBatchI8_zen2;

    // 1. Compute cosine distances (kernel is the hot path)
    const distances = try allocator.alloc(f32, N_CHUNKS);
    defer allocator.free(distances);
    kernel(query.ptr, corpus.ptr, N_CHUNKS, distances.ptr);

    // 2. Sort chunk indices by distance ascending (smaller = more similar)
    const ranked = try allocator.alloc(usize, N_CHUNKS);
    defer allocator.free(ranked);
    for (ranked, 0..) |*idx, i| idx.* = i;
    std.sort.heap(usize, ranked, SortCtx{ .d = distances }, SortCtx.lt);

    // 3. Emit top-K
    std.debug.print("Codemogger-shape demo: query vs {d}-chunk corpus, top-{d}\n\n", .{ N_CHUNKS, TOP_K });
    for (ranked[0..TOP_K], 0..) |idx, rank| {
        std.debug.print("  #{d}: chunk_idx={d:4}  cosine_dist={d:.6}\n", .{ rank + 1, idx, distances[idx] });
    }
    std.debug.print("\nIn real codemogger: chunk_idx maps to (file_path, span_start, span_end) loaded from sqlite alongside embeddings; results render as 'src/foo.zig:42-87 (similarity 0.93)'.\n", .{});
}
