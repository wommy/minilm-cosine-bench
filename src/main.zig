const std = @import("std");
const cosine = @import("cosine.zig");
const cosine_zen2 = @import("cosine_zen2.zig");
const data = @import("data.zig");

const Impl = struct {
    name: []const u8,
    target: []const u8,
    fn_ptr: cosine.KernelFn,
};

/// Add your impl here. Format: `.{ .name = "yours", .target = "uarch", .fn_ptr = yourimpl.fn }`
const IMPLS = [_]Impl{
    .{ .name = "scalar", .target = "any (reference)", .fn_ptr = cosine.cosineBatchI8_scalar },
    .{ .name = "zen2", .target = "Zen+ AVX2 (Ryzen 3400G)", .fn_ptr = cosine_zen2.cosineBatchI8_zen2 },
    // .{ .name = "avx512", .target = "AVX-512 (Zen4 / Ice Lake+)", .fn_ptr = cosine_avx512.cosineBatchI8_avx512 },
};

const N_ROWS: usize = 1000;
const N_ITERS: usize = 100;
const SEED: u64 = 42;

fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(std.os.linux.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const query = try data.generateQuery(allocator, SEED);
    defer allocator.free(query);
    const corpus = try data.generateCorpus(allocator, N_ROWS, SEED + 1);
    defer allocator.free(corpus);

    const ref_out = try allocator.alloc(f32, N_ROWS);
    defer allocator.free(ref_out);
    cosine.cosineBatchI8_scalar(query.ptr, corpus.ptr, N_ROWS, ref_out.ptr);

    std.debug.print("MiniLM-L6-v2 i8 cosine kernel bench\n", .{});
    std.debug.print("  dim={d}  corpus_rows={d}  iters={d}  seed={d}\n\n", .{ cosine.DIM, N_ROWS, N_ITERS, SEED });

    for (IMPLS) |impl| {
        const out = try allocator.alloc(f32, N_ROWS);
        defer allocator.free(out);
        impl.fn_ptr(query.ptr, corpus.ptr, N_ROWS, out.ptr);

        var max_diff: f32 = 0;
        for (out, ref_out) |a, b| {
            const d = @abs(a - b);
            if (d > max_diff) max_diff = d;
        }

        const t_start = nowNs();
        for (0..N_ITERS) |_| impl.fn_ptr(query.ptr, corpus.ptr, N_ROWS, out.ptr);
        const elapsed_ns = nowNs() - t_start;
        const ns_per_op = elapsed_ns / (N_ITERS * N_ROWS);
        const ops_per_s = if (ns_per_op > 0) 1_000_000_000 / ns_per_op else 0;

        std.debug.print("[{s}]  target: {s}\n", .{ impl.name, impl.target });
        std.debug.print("  correctness max-diff vs scalar: {e:.2}\n", .{max_diff});
        std.debug.print("  perf: {d} ns/op  ({d} ops/s)\n\n", .{ ns_per_op, ops_per_s });
    }
}
