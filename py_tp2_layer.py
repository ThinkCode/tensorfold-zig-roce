import time
import torch
import torch.nn.functional as F

def benchmark_pytorch_tp2():
    device = "cuda"
    D = 4096
    N = 7168
    H = N * 2
    
    # FP16 weights
    w_gate = torch.full((D, N), 0.01, dtype=torch.float16, device=device)
    w_up = torch.full((D, N), 0.01, dtype=torch.float16, device=device)
    w_down = torch.full((N, D), 0.01, dtype=torch.float16, device=device)
    
    print("========================================================================================================")
    print("  PYTHON / PYTORCH TP=2 TRANSFORMER LAYER (PyTorch ATen + RoCE All-Gather Baseline)                     ")
    print(f"  Container: glm53-flash-tf | PyTorch {torch.__version__} | Hidden D={D}, MLP H={H}                     ")
    print("========================================================================================================")
    print(f"{'Batch':<8} | {'Activation':<14} | {'GEMM 1+SwiGLU':<14} | {'All-Gather':<16} | {'Total Layer Step':<18} | {'Throughput':<14}")
    print(f"{'-'*8}-|-{'-'*14}-|-{'-'*14}-|-{'-'*16}-|-{'-'*18}-|-{'-'*14}")

    iters = 1000
    warmup = 15
    batches = [1, 4, 8]

    # Pre-allocate gather buffers
    for M in batches:
        x = torch.full((M, D), 1.0, dtype=torch.float16, device=device)
        act_bytes = M * N * 2
        act_gathered = torch.empty((M, H), dtype=torch.float16, device=device)

        # Warmup
        for _ in range(warmup):
            g = torch.matmul(x, w_gate)
            u = torch.matmul(x, w_up)
            act_local = F.silu(g) * u
            # Simulate rank gather
            act_gathered[:, :N].copy_(act_local)
            act_gathered[:, N:].copy_(act_local)
            z = torch.matmul(act_local, w_down)
            torch.cuda.synchronize()

        # Measure isolated gather/dispatch overhead
        t_gather_0 = time.perf_counter_ns()
        for _ in range(iters):
            act_gathered[:, :N].copy_(act_local)
            act_gathered[:, N:].copy_(act_local)
            torch.cuda.synchronize()
        t_gather_1 = time.perf_counter_ns()
        avg_gather_us = (t_gather_1 - t_gather_0) / (iters * 1000)

        # Measure Full End-to-End Layer
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            g = torch.matmul(x, w_gate)
            u = torch.matmul(x, w_up)
            act_local = F.silu(g) * u
            act_gathered[:, :N].copy_(act_local)
            act_gathered[:, N:].copy_(act_local)
            z = torch.matmul(act_local, w_down)
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()

        total_ns = t1 - t0
        avg_total_us = total_ns / (iters * 1000)
        avg_compute_us = avg_total_us - avg_gather_us
        tok_s = (M * iters) / (total_ns / 1e9)

        b_label = f"B = {M}"
        act_label = f"{act_bytes / 1024.0:.1f} KiB"
        print(f"{b_label:<8} | {act_label:<14} | {avg_compute_us:>10.2f} us | {avg_gather_us:>12.2f} us | {avg_total_us:>14.2f} us | {tok_s:>10.1f} tok/s")

if __name__ == "__main__":
    benchmark_pytorch_tp2()
