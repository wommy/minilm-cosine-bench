const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize: std.builtin.OptimizeMode = .ReleaseFast;

    // Bench harness
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const bench_exe = b.addExecutable(.{
        .name = "minilm-cosine-bench",
        .root_module = bench_mod,
    });
    b.installArtifact(bench_exe);
    const bench_run = b.addRunArtifact(bench_exe);
    bench_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| bench_run.addArgs(args);
    const bench_step = b.step("run", "Run the bench (correctness + ns/op per impl)");
    bench_step.dependOn(&bench_run.step);

    // Codemogger-shaped usage example
    const example_mod = b.createModule(.{
        .root_source_file = b.path("src/example.zig"),
        .target = target,
        .optimize = optimize,
    });
    const example_exe = b.addExecutable(.{
        .name = "minilm-cosine-example",
        .root_module = example_mod,
    });
    b.installArtifact(example_exe);
    const example_run = b.addRunArtifact(example_exe);
    example_run.step.dependOn(b.getInstallStep());
    const example_step = b.step("example", "Run codemogger-shape usage demo (query → top-K)");
    example_step.dependOn(&example_run.step);
}
