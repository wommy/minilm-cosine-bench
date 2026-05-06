const std = @import("std");
const cosine = @import("cosine.zig");

const DIM = cosine.DIM;
const I8_LANES: usize = 16;
const I8_CHUNKS: usize = DIM / I8_LANES; // 24
const VI32 = @Vector(I8_LANES, i32);

/// Zen+ AVX2 impl using `@Vector(16, i32)` widening + dot/q²/b² single-pass.
/// LLVM emits VPMADDWD-equivalent on AVX2; no AVX-VNNI / vpdpbusd usage (not present pre-Zen4).
pub fn cosineBatchI8_zen2(query: [*]const i8, corpus: [*]const i8, n_rows: usize, out: [*]f32) void {
    const eps: f32 = 1e-12;
    var q_vecs: [I8_CHUNKS]VI32 = undefined;
    var q_sq: i64 = 0;
    inline for (0..I8_CHUNKS) |k| {
        const q_i8: @Vector(I8_LANES, i8) = query[k * I8_LANES ..][0..I8_LANES].*;
        const q_i32: VI32 = q_i8;
        q_vecs[k] = q_i32;
        q_sq += @reduce(.Add, q_i32 * q_i32);
    }
    const q_norm: f32 = @sqrt(@max(@as(f32, @floatFromInt(q_sq)), eps));
    for (0..n_rows) |i| {
        const row_off = i * DIM;
        var dot: i64 = 0;
        var b_sq: i64 = 0;
        inline for (0..I8_CHUNKS) |k| {
            const b_i8: @Vector(I8_LANES, i8) = corpus[row_off + k * I8_LANES ..][0..I8_LANES].*;
            const b_i32: VI32 = b_i8;
            dot += @reduce(.Add, q_vecs[k] * b_i32);
            b_sq += @reduce(.Add, b_i32 * b_i32);
        }
        const b_norm: f32 = @sqrt(@max(@as(f32, @floatFromInt(b_sq)), eps));
        out[i] = 1.0 - @as(f32, @floatFromInt(dot)) / (q_norm * b_norm);
    }
}
