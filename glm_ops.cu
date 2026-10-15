#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <math.h>

namespace {

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
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

    for (int i = index; i < shard_packs; i += stride) send_slot[i] = in[i];
    __syncthreads();
    if (tid == 0) {
        fence_sc_sys();
        const uint32_t prior = atom_add_relaxed_gpu(stage_ctr, 1u);
        if ((prior + 1u) % gridDim.x == 0u) {
            st_relaxed_sys(ctrl + C_NBYTES, nbytes);
            st_relaxed_sys(ctrl + C_SLOT_NBYTES + slot, nbytes);
            fence_sc_sys();
            st_relaxed_sys(ctrl + C_SEQ, seq);
        }
    }
    if (tid < world * n_hca) {
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
    if (ld_relaxed_gpu(poison) == 0u) {
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
    fence_sc_gpu();
    __syncthreads();
    if (tid == 0) {
        const uint32_t prior = atom_add_relaxed_gpu(tail_ctr, 1u);
        if ((prior + 1u) % gridDim.x == 0u) {
            fence_sc_gpu();
            if (ld_acquire_sys(ctrl + C_ERROR) == 0u) st_release_gpu(epoch, seq);
        }
    }
}

__global__ void rmsnorm_kernel(const half* __restrict__ x, const half* __restrict__ weight, half* __restrict__ out, int dim, float eps) {
    int row = blockIdx.x;
    const half* rx = x + (long long)row * dim;
    half* rout = out + (long long)row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(rx[i]);
        sum_sq += v * v;
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum_sq);
    __syncthreads();

    float rsqrt_val = rsqrtf(s_sum / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(rx[i]) * rsqrt_val * __half2float(weight[i]);
        rout[i] = __float2half(v);
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

__global__ void argmax_kernel(const half* __restrict__ logits, uint32_t* __restrict__ out_token, int vocab_size) {
    __shared__ float s_max[256];
    __shared__ int s_idx[256];
    int tid = threadIdx.x;

    float max_val = -1e30f;
    int max_idx = 0;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    s_max[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) {
                s_max[tid] = s_max[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_token = (uint32_t)s_idx[0];
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

int run_rmsnorm(const void* x, const void* weight, void* out, int rows, int dim, float eps, void* stream) {
    rmsnorm_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(
        (const half*)x, (const half*)weight, (half*)out, dim, eps);
    return (int)cudaGetLastError();
}

int run_silu_mul(const void* gate, const void* up, void* out, int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_and_mul_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (const half*)gate, (const half*)up, (half*)out, n);
    return (int)cudaGetLastError();
}

int run_argmax(const void* logits, uint32_t* out_token, int vocab_size, void* stream) {
    argmax_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(
        (const half*)logits, out_token, vocab_size);
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

int run_hgemm(void* handle, int M, int N, int K, const void* W, const void* X, void* Y, void* stream) {
    cublasHandle_t h = (cublasHandle_t)handle;
    cublasSetStream(h, (cudaStream_t)stream);
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);
    cublasStatus_t status = cublasHgemm(
        h, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (const half*)W, N,
        (const half*)X, K,
        &beta,
        (half*)Y, N
    );
    return (int)status;
}

}
