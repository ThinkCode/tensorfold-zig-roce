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
extern "c" fn tcp_exchange_blobs(rank: c_int, peer_ip: [*:0]const u8, port: c_int, send_buf: [*]const u8, recv_buf: [*]u8, len: c_int) c_int;
extern "c" fn current_time_ns() u64;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);

    var rank: c_int = 0;
    var peer_ip: [*:0]const u8 = "10.10.7.4";
    var tcp_port: c_int = 29777;

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

    std.debug.print("[Rank {d}] Starting Native Zig RoCE Benchmark (Dual CX7 Rails)\n", .{rank});
    const world: c_int = 2;
    const slot_bytes: u64 = 256 * 1024; // 256 KiB
    var layout: [7]u64 = undefined;
    if (roce_layout(world, slot_bytes, &layout) != 0) {
        std.debug.print("roce_layout failed\n", .{});
        return;
    }
    const flag_off = layout[1];
    const send_off = layout[2];
    const ctrl_off = layout[3];
    const total_bytes = layout[4];
    const flag_stride = layout[5];
    const slots = layout[6];

    const mem = try std.heap.page_allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), total_bytes);
    defer std.heap.page_allocator.free(mem);
    @memset(mem, 0);

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
        mem.ptr,
        total_bytes,
        slot_bytes,
        &err_buf,
        err_buf.len,
    ) orelse {
        std.debug.print("roce_create failed: {s}\n", .{ std.mem.sliceTo(&err_buf, 0) });
        return;
    };
    defer roce_destroy(ctx);

    const blob_bytes = roce_blob_bytes();
    const my_blob = try arena.alloc(u8, blob_bytes);
    _ = roce_local_blob(ctx, my_blob.ptr, blob_bytes);

    const peer_blob = try arena.alloc(u8, blob_bytes);

    if (tcp_exchange_blobs(rank, peer_ip, tcp_port, my_blob.ptr, peer_blob.ptr, @intCast(blob_bytes)) != 0) {
        std.debug.print("TCP blob exchange failed\n", .{});
        return;
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
        return;
    }
    if (roce_start(ctx) != 0) {
        std.debug.print("roce_start failed: {s}\n", .{roce_error(ctx)});
        return;
    }
    std.debug.print("[Rank {d}] ConnectX-7 RoCE v2 Handshake complete! Both rails active.\n", .{rank});

    const ctrl: [*]volatile u32 = @ptrCast(@alignCast(mem.ptr + ctrl_off));
    const send_buf = mem.ptr + send_off;
    const peer_rank: u64 = if (rank == 0) 1 else 0;

    const test_sizes = [_]usize{ 1024, 4096, 16384, 65536, 131072, 262144 };
    const iters: usize = 2000;
    var seq: u32 = 1;

    if (rank == 0) {
        std.debug.print("\n========================================================================\n", .{});
        std.debug.print("  NATIVE ZIG RoCE TP=2 MICROBENCHMARK (Twin ConnectX-7 100 GbE Rails)   \n", .{});
        std.debug.print("========================================================================\n", .{});
        std.debug.print("{s:<12} | {s:<14} | {s:<14} | {s:<16}\n", .{ "Payload", "RTT Latency", "1-Way Latency", "Bandwidth" });
        std.debug.print("{s:-<12}-|-{s:-<14}-|-{s:-<14}-|-{s:-<16}\n", .{ "", "", "", "" });
    }

    for (test_sizes) |payload_size| {
        @memset(send_buf[0..payload_size], @as(u8, @intCast(seq & 0xFF)));

        var total_ns: u64 = 0;

        for (0..iters) |_| {
            const slot: u64 = seq & 1;
            const peer_flag_ptr: *const volatile u32 = @ptrCast(@alignCast(mem.ptr + flag_off +
                (((peer_rank * slots + slot) * 2 + 0) * flag_stride)));

            if (rank == 0) {
                const t0 = current_time_ns();

                // Publish nbytes at ctrl[4 + slot], then seq at ctrl[0]
                ctrl[4 + slot] = @as(u32, @intCast(payload_size));
                @atomicStore(u32, &ctrl[0], seq, .release);

                while (@atomicLoad(u32, peer_flag_ptr, .acquire) != seq) {
                    if (roce_failed(ctx) != 0) {
                        std.debug.print("RoCE proxy failed on rank 0: {s}\n", .{roce_error(ctx)});
                        return;
                    }
                    std.atomic.spinLoopHint();
                }

                const t1 = current_time_ns();
                total_ns += (t1 - t0);
            } else {
                while (@atomicLoad(u32, peer_flag_ptr, .acquire) != seq) {
                    if (roce_failed(ctx) != 0) {
                        std.debug.print("RoCE proxy failed on rank 1: {s}\n", .{roce_error(ctx)});
                        return;
                    }
                    std.atomic.spinLoopHint();
                }

                // Echo back
                ctrl[4 + slot] = @as(u32, @intCast(payload_size));
                @atomicStore(u32, &ctrl[0], seq, .release);
            }
            seq += 1;
        }

        if (rank == 0) {
            const avg_rtt_us = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iters * 1000));
            const one_way_us = avg_rtt_us / 2.0;
            const transferred_bytes = @as(f64, @floatFromInt(payload_size * iters * 2));
            const total_sec = @as(f64, @floatFromInt(total_ns)) / 1e9;
            const bw_gb_s = (transferred_bytes / total_sec) / 1e9;

            var size_str: [32]u8 = undefined;
            const s = try std.fmt.bufPrint(&size_str, "{d} KiB", .{ payload_size / 1024 });
            std.debug.print("{s:<12} | {d:>9.2} us   | {d:>9.2} us   | {d:>11.2} GB/s\n",
                .{ s, avg_rtt_us, one_way_us, bw_gb_s });
        }
    }

    std.debug.print("[Rank {d}] RoCE Microbenchmark completed cleanly.\n", .{rank});
    roce_stop(ctx);
}
