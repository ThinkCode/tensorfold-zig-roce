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
extern "c" fn cudaGetErrorString(err: c_int) [*:0]const u8;

const cudaMemcpyHostToDevice: c_int = 1;
const cudaMemcpyDeviceToHost: c_int = 2;
const cudaHostAllocPortable: c_uint = 0x01;
const cudaHostAllocMapped: c_uint = 0x02;

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
    var tcp_port: c_int = 29778;

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

    std.debug.print("[Rank {d}] Phase 2: Native Zig + CUDA Driver RoCE TP=2 Harness\n", .{rank});

    // 1. Establish persistent TCP synchronization socket
    const sync_fd = tcp_connect_sync(rank, peer_ip, tcp_port);
    if (sync_fd < 0) {
        std.debug.print("[Rank {d}] TCP rendezvous failed with rc {d}\n", .{ rank, sync_fd });
        return error.TcpSyncFailed;
    }
    defer _ = close(sync_fd);
    std.debug.print("[Rank {d}] TCP control rendezvous established (fd {d}).\n", .{ rank, sync_fd });

    // 2. Query layout
    const world: c_int = 2;
    const slot_bytes: u64 = 256 * 1024; // 256 KiB slots
    var layout: [7]u64 = undefined;
    if (roce_layout(world, slot_bytes, &layout) != 0) {
        std.debug.print("roce_layout failed\n", .{});
        return error.RoceLayoutFailed;
    }
    const recv_off = layout[0];
    const flag_off = layout[1];
    const send_off = layout[2];
    const ctrl_off = layout[3];
    const total_bytes = layout[4];
    const flag_stride: c_int = @intCast(layout[5]);
    const slots: c_int = @intCast(layout[6]);

    // 3. Allocate Pinned Host Memory with CUDA UVA mapping (cudaHostAlloc)
    var host_ptr: ?*anyopaque = null;
    if (cudaHostAlloc(&host_ptr, total_bytes, cudaHostAllocPortable | cudaHostAllocMapped) != 0) {
        std.debug.print("[Rank {d}] cudaHostAlloc failed for {d} bytes\n", .{ rank, total_bytes });
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
    ) orelse {
        std.debug.print("roce_create failed: {s}\n", .{std.mem.sliceTo(&err_buf, 0)});
        return error.RoceCreateFailed;
    };
    defer roce_destroy(ctx);

    // 5. Exchange connection blobs via TCP
    const blob_bytes = roce_blob_bytes();
    const my_blob = try arena.alloc(u8, blob_bytes);
    _ = roce_local_blob(ctx, my_blob.ptr, blob_bytes);

    const peer_blob = try arena.alloc(u8, blob_bytes);
    if (tcp_sync_exchange(sync_fd, my_blob.ptr, peer_blob.ptr, @intCast(blob_bytes)) != 0) {
        std.debug.print("tcp_sync_exchange failed\n", .{});
        return error.BlobExchangeFailed;
    }

    const all_blobs = try arena.alloc(u8, blob_bytes * 2);
    if (rank == 0) {
        @memcpy(all_blobs[0..blob_bytes], my_blob);
        @memcpy(all_blobs[blob_bytes..], peer_blob);
    } else {
        @memcpy(all_blobs[0..blob_bytes], peer_blob);
        @memcpy(all_blobs[blob_bytes..], my_blob);
    }

    if (roce_connect(ctx, all_blobs.ptr, all_blobs.len) != 0) {
        std.debug.print("roce_connect failed: {s}\n", .{roce_error(ctx)});
        return error.RoceConnectFailed;
    }
    if (roce_start(ctx) != 0) {
        std.debug.print("roce_start failed: {s}\n", .{roce_error(ctx)});
        return error.RoceStartFailed;
    }
    std.debug.print("[Rank {d}] Twin ConnectX-7 rails connected & ready.\n", .{rank});

    // 6. Allocate GPU Device Memory (CUDA Driver / Runtime)
    const max_shard_bytes: usize = 256 * 1024; // 256 KiB
    var d_in: ?*anyopaque = null;
    var d_out: ?*anyopaque = null;
    var d_counters: ?*anyopaque = null; // 10 uint32 words for epoch, stage, tail, poison

    if (cudaMalloc(&d_in, max_shard_bytes) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_in);
    if (cudaMalloc(&d_out, max_shard_bytes * 2) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_out);
    if (cudaMalloc(&d_counters, 10 * @sizeOf(u32)) != 0) return error.CudaMallocFailed;
    defer _ = cudaFree(d_counters);

    _ = cudaMemset(d_counters, 0, 10 * @sizeOf(u32));

    var stream: ?*anyopaque = null;
    if (cudaStreamCreate(&stream) != 0) return error.CudaStreamFailed;
    defer _ = cudaStreamDestroy(stream);

    // Host staging buffers for CPU verification
    const h_in = try arena.alloc(u8, max_shard_bytes);
    const h_out = try arena.alloc(u8, max_shard_bytes * 2);

    // Distinct rank signatures: Rank 0 = 0xAA, Rank 1 = 0xBB
    const fill_val: u8 = if (rank == 0) 0xAA else 0xBB;
    @memset(h_in, fill_val);
    _ = cudaMemcpy(d_in, h_in.ptr, max_shard_bytes, cudaMemcpyHostToDevice);

    std.debug.print("[Rank {d}] GPU device memory allocated & initialized.\n", .{rank});

    // 7. Benchmark Shard Sizes
    const test_sizes = [_]usize{ 1024, 4096, 16384, 65536, 131072, 262144 };
    const iters: usize = 1000;
    const warmup_iters: usize = 10;
    const threads: c_int = 512;
    const classes: usize = 4; // BLOCKS = 8 -> 4 power-of-two classes (1, 2, 4, 8)

    if (rank == 0) {
        std.debug.print("\n========================================================================================\n", .{});
        std.debug.print("  NATIVE ZIG + CUDA DRIVER: ZERO-PYTHON GPU-TO-GPU ALL-GATHER OVER RoCE                \n", .{});
        std.debug.print("  Cluster: 2x GX10 GB10 (sm_121) | Dual ConnectX-7 100 GbE RoCE Rails                  \n", .{});
        std.debug.print("========================================================================================\n", .{});
        std.debug.print("{s:<12} | {s:<10} | {s:<14} | {s:<18} | {s:<14}\n", .{ "Shard Size", "Grid Dim", "Latency (GPU)", "All-Gather BW", "Verification" });
        std.debug.print("{s:-<12}-|-{s:-<10}-|-{s:-<14}-|-{s:-<18}-|-{s:-<14}\n", .{ "", "", "", "", "" });
    }

    for (test_sizes) |shard_size| {
        const shard_packs: c_int = @intCast(shard_size / 16);
        const row_packs: c_int = shard_packs;
        const nbytes: u32 = @intCast(shard_size);

        // Calculate power-of-two gridDim and klass
        const need = @max(1, @divTrunc(shard_packs + (2 * threads) - 1, 2 * threads));
        var grid: c_int = 1;
        while (grid < need and grid < 8) : (grid *= 2) {}
        const klass: usize = std.math.log2_int(usize, @intCast(grid));

        const c_u32 = @as([*]u32, @ptrCast(@alignCast(d_counters.?)));
        const p_epoch = @as(?*anyopaque, @ptrCast(c_u32));
        const p_stage = @as(?*anyopaque, @ptrCast(c_u32 + (1 + klass)));
        const p_tail = @as(?*anyopaque, @ptrCast(c_u32 + (1 + classes + klass)));
        const p_poison = @as(?*anyopaque, @ptrCast(c_u32 + (1 + 2 * classes)));

        // Pre-warmup TCP barrier
        _ = tcp_sync_barrier(sync_fd);

        // Warmup runs
        for (0..warmup_iters) |_| {
            _ = launch_roce_gather(
                d_in, d_out, shard_packs, nbytes, row_packs,
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

        // Pre-benchmark TCP barrier
        _ = tcp_sync_barrier(sync_fd);

        const t0 = current_time_ns();

        for (0..iters) |_| {
            const rc = launch_roce_gather(
                d_in, d_out, shard_packs, nbytes, row_packs,
                host_mem + recv_off,
                host_mem + flag_off,
                host_mem + send_off,
                host_mem + ctrl_off,
                @intCast(slot_bytes),
                p_epoch, p_stage, p_tail, p_poison,
                20000000, world, rank, slots, flag_stride, 2, grid, threads, stream,
            );
            if (rc != 0) {
                std.debug.print("[Rank {d}] Kernel launch error: {d}\n", .{ rank, rc });
                return error.KernelLaunchFailed;
            }
            _ = cudaStreamSynchronize(stream);
        }

        const t1 = current_time_ns();
        const total_ns = t1 - t0;

        // Check proxy health
        if (roce_failed(ctx) != 0) {
            std.debug.print("[Rank {d}] RoCE proxy failure: {s}\n", .{ rank, roce_error(ctx) });
            return error.RoceProxyFailed;
        }

        // Host verification of GPU buffers:
        // Rank 0 slice: [0 .. shard_size] must be all 0xAA
        // Rank 1 slice: [shard_size .. shard_size*2] must be all 0xBB
        @memset(h_out, 0);
        _ = cudaMemcpy(h_out.ptr, d_out, shard_size * 2, cudaMemcpyDeviceToHost);

        var verified = true;
        for (h_out[0..shard_size]) |b| {
            if (b != 0xAA) {
                verified = false;
                break;
            }
        }
        for (h_out[shard_size .. shard_size * 2]) |b| {
            if (b != 0xBB) {
                verified = false;
                break;
            }
        }

        if (rank == 0) {
            const avg_lat_us = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iters * 1000));
            // Total bidirectional gathered payload across both nodes = shard_size * 2 * iters
            const total_bytes_f = @as(f64, @floatFromInt(shard_size * 2 * iters));
            const total_sec = @as(f64, @floatFromInt(total_ns)) / 1e9;
            const bw_gb_s = (total_bytes_f / total_sec) / 1e9;

            var size_str: [32]u8 = undefined;
            const s = if (shard_size >= 1024)
                try std.fmt.bufPrint(&size_str, "{d} KiB", .{shard_size / 1024})
            else
                try std.fmt.bufPrint(&size_str, "{d} B", .{shard_size});

            var grid_str: [16]u8 = undefined;
            const g = try std.fmt.bufPrint(&grid_str, "{d} blocks", .{grid});

            const v_str: []const u8 = if (verified) "EXACT MATCH" else "MISMATCH";
            std.debug.print("{s:<12} | {s:<10} | {d:>9.2} us   | {d:>12.2} GB/s  | {s:<14}\n",
                .{ s, g, avg_lat_us, bw_gb_s, v_str });
        }
    }

    _ = tcp_sync_barrier(sync_fd);
    std.debug.print("[Rank {d}] Phase 2 GPU-to-GPU Harness completed successfully!\n", .{rank});
    roce_stop(ctx);
}
