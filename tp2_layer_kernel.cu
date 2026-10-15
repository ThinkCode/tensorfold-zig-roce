#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t atom_add_relaxed_gpu(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

constexpr int C_SEQ = 0, C_NBYTES = 1, C_ERROR = 2, C_PEER = 3, C_SLOT_NBYTES = 4, C_HCA = 6;

__global__ void gather_kernel(const uint4* __restrict__ in, uint4* __restrict__ out, int shard_packs, uint32_t nbytes,
                              int row_packs, char* recv_base, char* flag_base, char* send_base, uint32_t* ctrl,
                              long long slot_bytes, uint32_t* epoch, uint32_t* stage_ctr, uint32_t* tail_ctr,
                              uint32_t* poison, uint32_t spin_limit, int world, int rank, int slots, int flag_stride,
                              int n_hca) {
    const int tid = threadIdx.x;
    const uint32_t seq = ld_relaxed_gpu(epoch) + 1u;
    const uint32_t slot = seq & 1u;
    uint4* send_slot = reinterpret_cast<uint4*>(send_base + (long long)slot * slot_bytes);
    const int index = blockIdx.x * blockDim.x + tid, stride = gridDim.x * blockDim.x;
    if (ld_relaxed_gpu(poison) != 0u) return;

    for (int i = index; i < shard_packs; i += stride) send_slot[i] = in[i];                         // 1. stage
    __syncthreads();
    if (tid == 0) {                                                                                 // 2. doorbell
        fence_sc_sys();
        const uint32_t prior = atom_add_relaxed_gpu(stage_ctr, 1u);
        if ((prior + 1u) % gridDim.x == 0u) {
            st_relaxed_sys(ctrl + C_NBYTES, nbytes);
            st_relaxed_sys(ctrl + C_SLOT_NBYTES + slot, nbytes);
            fence_sc_sys();
            st_relaxed_sys(ctrl + C_SEQ, seq);
        }
    }
    if (tid < world * n_hca) {                                                                      // 3. wait
        const int peer = tid / n_hca, hca = tid - peer * n_hca;
        if (peer != rank) {
            const uint32_t* flag = reinterpret_cast<const uint32_t*>(
                flag_base + (((long long)peer * slots + slot) * n_hca + hca) * flag_stride);
            uint32_t polls = 0;
            while (ld_acquire_sys(flag) != seq) {
                if (++polls >= spin_limit) {
                    st_relaxed_sys(ctrl + C_PEER, (uint32_t)peer);
                    st_relaxed_sys(ctrl + C_HCA, (uint32_t)hca);
                    st_relaxed_sys(ctrl + C_ERROR, seq);
                    st_release_gpu(poison, seq);
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (ld_relaxed_gpu(poison) == 0u) {                                                             // 4. copy out
        const int out_row_packs = world * row_packs;
        for (int source = 0; source < world; ++source) {
            const uint4* peer_slot = reinterpret_cast<const uint4*>(
                recv_base + ((long long)source * slots + slot) * slot_bytes);
            for (int i = index; i < shard_packs; i += stride) {
                const int row = i / row_packs, col = i - row * row_packs;
                const uint4 w = source == rank ? in[i] : ld_relaxed_sys_v4(peer_slot + i);
                out[(long long)row * out_row_packs + (long long)source * row_packs + col] = w;
            }
        }
    }
    fence_sc_gpu();                                                                                 // 5. epoch
    __syncthreads();
    if (tid == 0) {
        const uint32_t prior = atom_add_relaxed_gpu(tail_ctr, 1u);
        if ((prior + 1u) % gridDim.x == 0u) {
            fence_sc_gpu();
            if (ld_relaxed_sys(ctrl + C_ERROR) == 0u) st_release_gpu(epoch, seq);
        }
    }
}

__global__ void silu_and_mul_kernel(const half* __restrict__ gate, const half* __restrict__ up, half* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu_g = g / (1.0f + __expf(-g));
        out[idx] = __float2half(silu_g * u);
    }
}

} // namespace

extern "C" {

int launch_roce_gather(
    const void* in, void* out, int shard_packs, uint32_t nbytes, int row_packs,
    void* recv_base, void* flag_base, void* send_base, void* ctrl,
    long long slot_bytes, void* epoch, void* stage_ctr, void* tail_ctr,
    void* poison, uint32_t spin_limit, int world, int rank, int slots, int flag_stride,
    int n_hca, int grid, int threads, void* stream)
{
    gather_kernel<<<(unsigned)grid, (unsigned)threads, 0, (cudaStream_t)stream>>>(
        (const uint4*)in, (uint4*)out, shard_packs, nbytes, row_packs,
        (char*)recv_base, (char*)flag_base, (char*)send_base, (uint32_t*)ctrl,
        slot_bytes, (uint32_t*)epoch, (uint32_t*)stage_ctr, (uint32_t*)tail_ctr,
        (uint32_t*)poison, spin_limit, world, rank, slots, flag_stride, n_hca);
    return (int)cudaGetLastError();
}

void* create_cublas_handle(void) {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return nullptr;
    return (void*)handle;
}

void destroy_cublas_handle(void* handle) {
    if (handle) cublasDestroy((cublasHandle_t)handle);
}

// Compute Y = X * W where X is (M, K) row-major, W is (K, N) row-major -> Y is (M, N) row-major
int run_hgemm(void* handle, int M, int N, int K, const void* W, const void* X, void* Y, void* stream) {
    cublasHandle_t h = (cublasHandle_t)handle;
    cublasSetStream(h, (cudaStream_t)stream);
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);
    cublasStatus_t status = cublasHgemm(
        h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (const half*)W, N,
        (const half*)X, K,
        &beta,
        (half*)Y, N
    );
    return (int)status;
}

int run_silu_mul(const void* gate, const void* up, void* out, int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_and_mul_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const half*)gate, (const half*)up, (half*)out, n);
    return (int)cudaGetLastError();
}

}
