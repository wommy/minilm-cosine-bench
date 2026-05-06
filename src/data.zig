const std = @import("std");
const cosine = @import("cosine.zig");
const DIM = cosine.DIM;

/// Deterministic seeded i8 test data — reproducible across machines + runs.
pub fn generateQuery(allocator: std.mem.Allocator, seed: u64) ![]i8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const buf = try allocator.alloc(i8, DIM);
    for (buf) |*v| v.* = random.intRangeAtMost(i8, -127, 127);
    return buf;
}

pub fn generateCorpus(allocator: std.mem.Allocator, n_rows: usize, seed: u64) ![]i8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const buf = try allocator.alloc(i8, n_rows * DIM);
    for (buf) |*v| v.* = random.intRangeAtMost(i8, -127, 127);
    return buf;
}
