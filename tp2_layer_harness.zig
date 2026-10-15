const std = @import("std");

const RoceCtx = anyopaque;
extern "c" fn roce_abi_version() c_int;
extern "c" fn roce_blob_bytes() u64;
extern "c" fn roce_layout(world: c_int, slot_bytes: u64, out: [*]u64) c_int;
extern "c" fn roce_create(
    world: c_int,
    rank: c_int,
    hca_names: [*]const [*:0]const u8,
    n_hca: c_int,
    gid_index: [*]const c_int,
    traffic_class: c_int,
    region: *anyopaque,
    region_bytes: u64,
    slot_bytes: u64,
    err: [*]u8,
    err_len: u64,
) ?*RoceCtx;
extern "c" fn roce_local_blob(c_ctx: *RoceCtx, out: [*]u8, out_len: u64) c_int;
extern "c" fn roce_connect(c_ctx: *RoceCtx, blobs: [*]const u8, blobs_len: u64) c_int;
extern "c" fn roce_start(c_ctx: *RoceCtx) c_int;
extern "c" fn roce_stop(c_ctx: *RoceCtx) void;
extern "c" fn roce_failed(c_ctx: *RoceCtx) c_int;
extern "c" fn roce_error(c_ctx: *RoceCtx) [*:0]const u8;
extern "c" fn roce_destroy(c_ctx: ?*RoceCtx) void;

// TCP synchronization helpers
extern "c" fn tcp_connect_sync(rank: c_int, peer_ip: [*:0]const u8, port: c_int) c_int;
extern "c" fn tcp_sync_barrier(fd: c_int) c_int;
extern "c" fn tcp_sync_exchange(fd: c_int, send_buf: [*]const u8, recv_buf: [*]u8, len: c_int) c_int;
extern "c" fn current_time_ns() u64;
extern "c" fn close(fd: c_int) c_int;

// CUDA Runtime API
extern "c" fn cudaHostAlloc(pHost: *?*anyopaque, size: usize, flags: c_uint) c_int;
extern "c" fn cudaFreeHost(pHost: ?*anyopaque) c_int;
extern "c" fn cudaMalloc(devPtr: *?*anyopaque, size: usize) c_int;
extern "c" fn cudaMemset(devPtr: ?*anyopaque, value: c_int, count: usize) c_int;
extern "c" fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_int) c_int;
extern "c" fn cudaFree(devPtr: ?*anyopaque) c_int;
extern "c" fn cudaStreamCreate(pStream: *?*anyopaque) c_int;
extern "c" fn cudaStreamSynchronize(stream: ?*anyopaque) c_int;
extern "c" fn cudaStreamDestroy(stream: ?*anyopaque) c_int;

const cudaMemcpyHostToDevice: c_int = 1;
const cudaMemcpyDeviceToHost: c_int = 2;
const cudaHostAllocPortable: c_uint = 0x01;
const cudaHostAllocMapped: c_uint = 0x02;

// cuBLAS & Layer Kernels
extern "c" fn create_cublas_handle() ?*anyopaque;
extern "c" fn destroy_cublas_handle(handle: ?*anyopaque) void;
extern "c" fn run_hgemm(handle: ?*anyopaque, M: c_int, N: c_int, K: c_int, W: ?*const anyopaque, X: ?*const anyopaque, Y: ?*anyopaque, stream: ?*anyopaque) c_int;
extern "c" fn run_silu_mul(gate: ?*const anyopaque, up: ?*const anyopaque, out: ?*anyopaque, n: c_int, stream: ?*anyopaque) c_int;

extern "c" fn launch_roce_gather(
    in: ?*const anyopaque,
    out: ?*anyopaque,
    shard_packs: c_int,
    nbytes: u32,
    row_packs: c_int,
    recv_base: ?*anyopaque,
    flag_base: ?*anyopaque,
    send_base: ?*anyopaque,
    ctrl: ?*anyopaque,
    slot_bytes: c_longlong,
    epoch: ?*anyopaque,
    stage_ctr: ?*anyopaque,
    tail_ctr: ?*anyopaque,
    poison: ?*anyopaque,
    spin_limit: u32,
    world: c_int,
    rank: c_int,
    slots: c_int,
    flag_stride: c_int,
    n_hca: c_int,
    grid: c_int,
    threads: c_int,
    stream: ?*anyopaque,
) c_int;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);

    var rank: c_int = 0;
    var peer_ip: [*:0]const u8 = "10.10.7.4";
    var tcp_port: c_int = 29779;

    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        if (std.mem.eql(u8, argv[i], "--rank") and i + 1 < argv.len) {
            i += 1;
            rank = try std.fmt.parseInt(c_int, argv[i], 10);
        } else if (std.mem.eql(u8, argv[i], "--peer-ip") and i + 1 < argv.len) {
            i += 1;
            const buf = try arena.alloc(u8, argv[i].len + 1);
            @memcpy(buf[0..argv[i].len], argv[i]);
            buf[argv[i].len] = 0;
            peer_ip = @ptrCast(buf.ptr);
        } else if (std.mem.eql(u8, argv[i], "--port") and i + 1 < argv.len) {
            i += 1;
            tcp_port = try std.fmt.parseInt(c_int, argv[i], 10);
        }
    }

    std.debug.print("[Rank {d}] Phase 2: Native Zig TP=2 Transformer Layer Prototype\n", .{rank});

    // 1. TCP rendezvous
    const sync_fd = tcp_connect_sync(rank, peer_ip, tcp_port);
    if (sync_fd < 0) {
        std.debug.print("[Rank {d}] TCP rendezvous failed with rc {d}\n", .{ rank, sync_fd });
        return error.TcpSyncFailed;
    }
    defer _ = close(sync_fd);
    std.debug.print("[Rank {d}] TCP control rendezvous established (fd {d}).\n", .{ rank, sync_fd });

    // 2. RoCE layout
    const world: c_int = 2;
    const slot_bytes: u64 = 256 * 1024; // 256 KiB slots
    var layout: [7]u64 = undefined;
    if (roce_layout(world, slot_bytes, &layout) != 0) return error.RoceLayoutFailed;
    const recv_off = layout[0];
    const flag_off = layout[1];
    const send_off = layout[2];
    const ctrl_off = layout[3];
    const total_bytes = layout[4];
    const flag_stride: c_int = @intCast(layout[5]);
    const slots: c_int = @intCast(layout[6]);

    // 3. Allocate Pinned Host Memory for RoCE
    var host_ptr: ?*anyopaque = null;
    if (cudaHostAlloc(&host_ptr, total_bytes, cudaHostAllocPortable | cudaHostAllocMapped) != 0) {
        return error.CudaHostAllocFailed;
    }
    defer _ = cudaFreeHost(host_ptr);
    const host_mem: [*]u8 = @ptrCast(host_ptr.?);
    @memset(host_mem[0..total_bytes], 0);

    // 4. Initialize RoCE Context with twin CX7 rails
    const hcas = [_][*:0]const u8{ "rocep1s0f0", "roceP2p1s0f0" };
    const gids = [_]c_int{ 3, 3 };
    var err_buf: [512]u8 = undefined;
    @memset(&err_buf, 0);

    const ctx = roce_create(
        world,
        rank,
        &hcas,
        2,
        &gids,
        0,
        host_mem,
        total_bytes,
        slot_bytes,
        &err_buf,
        err_buf.len,
    ) orelse return error.RoceCreateFailed;
    defer roce_destroy(ctx);

    const blob_bytes = roce_blob_bytes();
    const my_blob = try arena.alloc(u8, blob_bytes);
    _ = roce_local_blob(ctx, my_blob.ptr, blob_bytes);
    const peer_blob = try arena.alloc(u8, blob_bytes);
    if (tcp_sync_exchange(sync_fd, my_blob.ptr, peer_blob.ptr, @intCast(blob_bytes)) != 0) return error.BlobExchangeFailed;

    const all_blobs = try arena.alloc(u8, blob_bytes * 2);
    if (rank == 0) {
        @memcpy(all_blobs[0..blob_bytes], my_blob);
        @memcpy(all_blobs[blob_bytes..], peer_blob);
    } else {
        @memcpy(all_blobs[0..blob_bytes], peer_blob);
        @memcpy(all_blobs[blob_bytes..], my_blob);
    }

    if (roce_connect(ctx, all_blobs.ptr, all_blobs.len) != 0) return error.RoceConnectFailed;
    if (roce_start(ctx) != 0) return error.RoceStartFailed;
    std.debug.print("[Rank {d}] Dual ConnectX-7 rails connected & ready.\n", .{rank});

    // 5. Initialize cuBLAS and CUDA Stream
    const cublas = create_cublas_handle() orelse return error.CublasCreateFailed;
    defer destroy_cublas_handle(cublas);

    var stream: ?*anyopaque = null;
    if (cudaStreamCreate(&stream) != 0) return error.CudaStreamFailed;
    defer _ = cudaStreamDestroy(stream);

    // 6. Layer Geometry (GLM-5.3-Flash style MLP dimensions)
    const D: c_int = 4096;          // hidden dimension
    const N: c_int = 7168;          // intermediate dimension per rank (TP=2: total H = 14336)
    const H: c_int = N * 2;         // full intermediate dimension (14336)
    const max_batch: c_int = 8;

    // Allocate Device Tensors (FP16: 2 bytes per element)
    const w_gate_bytes: usize = @intCast(D * N * 2);
    const w_up_bytes: usize = @intCast(D * N * 2);
    const w_down_bytes: usize = @intCast(N * D * 2);
    const max_x_bytes: usize = @intCast(max_batch * D * 2);
    const max_g_bytes: usize = @intCast(max_batch * N * 2);
    const max_act_gather_bytes: usize = @intCast(max_batch * H * 2);
    const max_z_bytes: usize = @intCast(max_batch * D * 2);

    var d_w_gate: ?*anyopaque = null;
    var d_w_up: ?*anyopaque = null;
    var d_w_down: ?*anyopaque = null;
    var d_x: ?*anyopaque = null;
    var d_g: ?*anyopaque = null;
    var d_u: ?*anyopaque = null;
    var d_act_local: ?*anyopaque = null;
    var d_act_gathered: ?*anyopaque = null;
    var d_z_local: ?*anyopaque = null;
    var d_counters: ?*anyopaque = null;

    if (cudaMalloc(&d_w_gate, w_gate_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_w_gate);
    if (cudaMalloc(&d_w_up, w_up_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_w_up);
    if (cudaMalloc(&d_w_down, w_down_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_w_down);
    if (cudaMalloc(&d_x, max_x_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_x);
    if (cudaMalloc(&d_g, max_g_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_g);
    if (cudaMalloc(&d_u, max_g_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_u);
    if (cudaMalloc(&d_act_local, max_g_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_act_local);
    if (cudaMalloc(&d_act_gathered, max_act_gather_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_act_gathered);
    if (cudaMalloc(&d_z_local, max_z_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_z_local);
    if (cudaMalloc(&d_counters, 10 * @sizeOf(u32)) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_counters);

    _ = cudaMemset(d_counters, 0, 10 * @sizeOf(u32));

    // Initialize weights and input with deterministic FP16 data
    // FP16 representation of 0.01 is 0x211F
    _ = cudaMemset(d_w_gate, 0x21, w_gate_bytes);
    _ = cudaMemset(d_w_up, 0x21, w_up_bytes);
    _ = cudaMemset(d_w_down, 0x21, w_down_bytes);
    _ = cudaMemset(d_x, 0x3C, max_x_bytes); // 0x3C00 is 1.0 in FP16

    std.debug.print("[Rank {d}] Model weights (D={d}, N={d}, H={d}) allocated on GPU.\n", .{ rank, D, N, H });

    const batches = [_]c_int{ 1, 4, 8 };
    const iters: usize = 1000;
    const warmup_iters: usize = 15;
    const threads: c_int = 512;
    const classes: usize = 4;

    if (rank == 0) {
        std.debug.print("\n========================================================================================================\n", .{});
        std.debug.print("  NATIVE ZIG TP=2 TRANSFORMER LAYER (cuBLAS GEMM + SwiGLU + Dual CX7 RoCE All-Gather)                  \n", .{});
        std.debug.print("  Cluster: 2x GX10 GB10 (sm_121) | Dual ConnectX-7 100 GbE RoCE Rails | Hidden D={d}, MLP H={d}        \n", .{ D, H });
        std.debug.print("========================================================================================================\n", .{});
        std.debug.print("{s:<8} | {s:<14} | {s:<14} | {s:<16} | {s:<18} | {s:<14}\n",
            .{ "Batch", "Activation", "GEMM 1+SwiGLU", "RoCE All-Gather", "Total Layer Step", "Throughput" });
        std.debug.print("{s:-<8}-|-{s:-<14}-|-{s:-<14}-|-{s:-<16}-|-{s:-<18}-|-{s:-<14}\n",
            .{ "", "", "", "", "", "" });
    }

    for (batches) |M| {
        const act_elements: c_int = M * N;
        const act_bytes: u32 = @intCast(act_elements * 2);
        const shard_packs: c_int = @intCast(act_bytes / 16);
        const row_packs: c_int = shard_packs;

        const need = @max(1, @divTrunc(shard_packs + (2 * threads) - 1, 2 * threads));
        var grid: c_int = 1;
        while (grid < need and grid < 8) : (grid *= 2) {}
        const klass: usize = std.math.log2_int(usize, @intCast(grid));

        const c_u32 = @as([*]u32, @ptrCast(@alignCast(d_counters.?)));
        const p_epoch = @as(?*anyopaque, @ptrCast(c_u32));
        const p_stage = @as(?*anyopaque, @ptrCast(c_u32 + (1 + klass)));
        const p_tail = @as(?*anyopaque, @ptrCast(c_u32 + (1 + classes + klass)));
        const p_poison = @as(?*anyopaque, @ptrCast(c_u32 + (1 + 2 * classes)));

        // Synchronize before warmup
        _ = tcp_sync_barrier(sync_fd);

        for (0..warmup_iters) |_| {
            // 1. Column-Parallel GEMM 1: Gate
            _ = run_hgemm(cublas, M, N, D, d_w_gate, d_x, d_g, stream);
            // 2. Column-Parallel GEMM 2: Up
            _ = run_hgemm(cublas, M, N, D, d_w_up, d_x, d_u, stream);
            // 3. Fused SwiGLU Activation
            _ = run_silu_mul(d_g, d_u, d_act_local, act_elements, stream);
            // 4. RoCE All-Gather over Dual CX7 Rails
            _ = launch_roce_gather(
                d_act_local, d_act_gathered, shard_packs, act_bytes, row_packs,
                host_mem + recv_off,
                host_mem + flag_off,
                host_mem + send_off,
                host_mem + ctrl_off,
                @intCast(slot_bytes),
                p_epoch, p_stage, p_tail, p_poison,
                20000000, world, rank, slots, flag_stride, 2, grid, threads, stream,
            );
            // 5. Row-Parallel GEMM 3: Down projection
            // Slice corresponding to this rank's shard
            const rank_act_offset: usize = if (rank == 0) 0 else @intCast(act_bytes);
            const d_act_slice: ?*anyopaque = @ptrCast(@as([*]u8, @ptrCast(d_act_gathered.?)) + rank_act_offset);
            _ = run_hgemm(cublas, M, D, N, d_w_down, d_act_slice, d_z_local, stream);

            _ = cudaStreamSynchronize(stream);
        }

        // Measure Isolated RoCE All-Gather component
        _ = tcp_sync_barrier(sync_fd);
        const t_roce_0 = current_time_ns();
        for (0..iters) |_| {
            _ = launch_roce_gather(
                d_act_local, d_act_gathered, shard_packs, act_bytes, row_packs,
                host_mem + recv_off,
                host_mem + flag_off,
                host_mem + send_off,
                host_mem + ctrl_off,
                @intCast(slot_bytes),
                p_epoch, p_stage, p_tail, p_poison,
                20000000, world, rank, slots, flag_stride, 2, grid, threads, stream,
            );
            _ = cudaStreamSynchronize(stream);
        }
        const t_roce_1 = current_time_ns();
        const avg_roce_us = @as(f64, @floatFromInt(t_roce_1 - t_roce_0)) / @as(f64, @floatFromInt(iters * 1000));

        // Measure Full End-to-End TP=2 Layer (Compute + RoCE All-Gather + Compute)
        _ = tcp_sync_barrier(sync_fd);
        const t0 = current_time_ns();

        for (0..iters) |_| {
            // Step 1: Gate GEMM
            _ = run_hgemm(cublas, M, N, D, d_w_gate, d_x, d_g, stream);
            // Step 2: Up GEMM
            _ = run_hgemm(cublas, M, N, D, d_w_up, d_x, d_u, stream);
            // Step 3: SwiGLU activation
            _ = run_silu_mul(d_g, d_u, d_act_local, act_elements, stream);
            // Step 4: RoCE All-Gather over Dual CX7 Rails
            _ = launch_roce_gather(
                d_act_local, d_act_gathered, shard_packs, act_bytes, row_packs,
                host_mem + recv_off,
                host_mem + flag_off,
                host_mem + send_off,
                host_mem + ctrl_off,
                @intCast(slot_bytes),
                p_epoch, p_stage, p_tail, p_poison,
                20000000, world, rank, slots, flag_stride, 2, grid, threads, stream,
            );
            // Step 5: Down GEMM
            const rank_act_offset: usize = if (rank == 0) 0 else @intCast(act_bytes);
            const d_act_slice: ?*anyopaque = @ptrCast(@as([*]u8, @ptrCast(d_act_gathered.?)) + rank_act_offset);
            _ = run_hgemm(cublas, M, D, N, d_w_down, d_act_slice, d_z_local, stream);

            _ = cudaStreamSynchronize(stream);
        }

        const t1 = current_time_ns();
        const total_layer_ns = t1 - t0;
        const avg_total_us = @as(f64, @floatFromInt(total_layer_ns)) / @as(f64, @floatFromInt(iters * 1000));
        const avg_compute_us = avg_total_us - avg_roce_us;
        const tokens_per_sec = (@as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(iters))) / (@as(f64, @floatFromInt(total_layer_ns)) / 1e9);

        if (rank == 0) {
            var b_str: [16]u8 = undefined;
            const b_label = try std.fmt.bufPrint(&b_str, "B = {d}", .{M});

            var act_str: [32]u8 = undefined;
            const act_label = try std.fmt.bufPrint(&act_str, "{d:.1} KiB", .{@as(f64, @floatFromInt(act_bytes)) / 1024.0});

            std.debug.print("{s:<8} | {s:<14} | {d:>10.2} us | {d:>12.2} us | {d:>14.2} us | {d:>10.1} tok/s\n",
                .{ b_label, act_label, avg_compute_us, avg_roce_us, avg_total_us, tokens_per_sec });
        }
    }

    _ = tcp_sync_barrier(sync_fd);
    std.debug.print("[Rank {d}] Native Zig TP=2 Layer benchmark complete!\n", .{rank});
    roce_stop(ctx);
}
