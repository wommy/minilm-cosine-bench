const std = @import("std");

/// Embedding dimension. MiniLM-L6-v2 is 384-dim.
pub const DIM: usize = 384;

/// Cosine-distance batch kernel contract.
/// Inputs: query [DIM]i8, corpus [n_rows*DIM]i8 (row-major)
/// Output: out [n_rows]f32 — `1 - cosine_similarity` per row
pub const KernelFn = *const fn (query: [*]const i8, corpus: [*]const i8, n_rows: usize, out: [*]f32) void;

/// Scalar reference impl. Slow but provably correct. Use as ground-truth for parity checks.
pub fn cosineBatchI8_scalar(query: [*]const i8, corpus: [*]const i8, n_rows: usize, out: [*]f32) void {
    const eps: f32 = 1e-12;
    var q_sq: i64 = 0;
    for (0..DIM) |k| {
        const q_i: i64 = query[k];
        q_sq += q_i * q_i;
    }
    const q_norm: f32 = @sqrt(@max(@as(f32, @floatFromInt(q_sq)), eps));
    for (0..n_rows) |i| {
        const row_off = i * DIM;
        var dot: i64 = 0;
        var b_sq: i64 = 0;
        for (0..DIM) |k| {
            const q_i: i64 = query[k];
            const b_i: i64 = corpus[row_off + k];
            dot += q_i * b_i;
            b_sq += b_i * b_i;
        }
        const b_norm: f32 = @sqrt(@max(@as(f32, @floatFromInt(b_sq)), eps));
        out[i] = 1.0 - @as(f32, @floatFromInt(dot)) / (q_norm * b_norm);
    }
}
