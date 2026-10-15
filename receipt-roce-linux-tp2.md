# Receipt: Linux ConnectX-7 Dual-Rail RoCE TP=2 Fabric & Kernel

Receipt for native Linux ConnectX-7 RoCE TP=2 implementation on NVIDIA Blackwell (GB10).

## Environment

- **Hardware:** 2x ASUS GX10 (NVIDIA GB10 Blackwell, `sm_121`, 121.7 GiB unified memory per node)
- **Interconnect:** Dual 100 GbE Mellanox ConnectX-7 rails (`rocep1s0f0`, `roceP2p1s0f0`), GID 3, traffic class 0
- **OS / Driver:** Ubuntu 24.04 LTS (kernel `7.0.0-1019-nvidia`), NVIDIA driver `580.178.04`, CUDA 13.0
- **Runtime:** Zig 0.17.0 (`-O ReleaseFast`), `libibverbs.so.1` (rdma-core 47.0), `libcudart.so`
- **Addressing:** RFC 5737 documentation addresses (`192.0.2.1` head <-> `192.0.2.2` worker)

## Phase 1: Native RDMA Microbenchmark (Dual CX7 Rails)

Measured across 12,000 continuous RDMA exchanges (1,000 warmup, 1,000 timed passes per size) with zero packet drops.

| Shard Size | One-Way Latency | RTT | Bandwidth | Verification |
|:---|:---:|:---:|:---:|:---:|
| 1 KiB | 4.35 us | 8.69 us | 0.23 GB/s | Exact Match |
| 4 KiB | 4.65 us | 9.30 us | 0.88 GB/s | Exact Match |
| 16 KiB | 5.63 us | 11.27 us | 2.91 GB/s | Exact Match |
| 64 KiB | 8.58 us | 17.15 us | 7.64 GB/s | Exact Match |
| 128 KiB | 11.70 us | 23.40 us | 11.20 GB/s | Exact Match |
| 256 KiB | 17.19 us | 34.38 us | 15.25 GB/s | Exact Match |

## Phase 2: Native Zig + CUDA Driver All-Gather (GB10 sm_121)

6,000 continuous GPU-to-GPU all-gather operations using `cudaHostAllocMapped` UVA and relaxed PTX synchronization.

| Shard Size | Grid Dim | Latency (GPU-to-GPU) | All-Gather Throughput | Verification |
|:---|:---:|:---:|:---:|:---:|
| 1 KiB | 1 block (512 thd) | 14.87 us | 0.14 GB/s | 100% Exact Match |
| 4 KiB | 1 block (512 thd) | 15.32 us | 0.53 GB/s | 100% Exact Match |
| 16 KiB | 1 block (512 thd) | 17.77 us | 1.84 GB/s | 100% Exact Match |
| 64 KiB | 4 blocks (2048 thd) | 22.17 us | 5.80 GB/s | 100% Exact Match |
| 128 KiB | 8 blocks (4096 thd) | 26.63 us | 9.80 GB/s | 100% Exact Match |
| 256 KiB | 8 blocks (4096 thd) | 34.33 us | 15.24 GB/s | 100% Exact Match |

## Phase 3: TP=2 Transformer Layer Forward Pass

Measured across 1,000 iterations for GLM-5.3-Flash MLP geometry (Hidden D=4096, Intermediate H=14336).

| Batch Size | Activation Shard | GEMM 1 + SwiGLU | RoCE All-Gather | Total Layer Step | Layer Throughput |
|:---|:---:|:---:|:---:|:---:|:---:|
| B = 1 | 14.0 KiB | 1271.2 us | 17.80 us | 1289.0 us | 775.8 tok/s |
| B = 4 | 56.0 KiB | 1265.6 us | 21.94 us | 1287.5 us | 3106.7 tok/s |
| B = 8 | 112.0 KiB | 1273.1 us | 25.40 us | 1298.5 us | 6160.8 tok/s |

RoCE communication overhead accounts for under 2.0% of total layer forward execution time.
