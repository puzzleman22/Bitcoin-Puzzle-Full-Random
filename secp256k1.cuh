//Author telegram: https://t.me/nmn5436

#ifndef SECP256K1_CUH
#define SECP256K1_CUH
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8

// Precomputation settings
#define PRECOMP_BITS 8
#define PRECOMP_SIZE (1 << PRECOMP_BITS)  // 256 points
#define WINDOW_SIZE 4  // For general scalar multiplication

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

struct ECPoint {
    BigInt x, y;
    bool infinity;
};

struct ECPointJac {
    BigInt X, Y, Z;
    bool infinity;
};

// Constants
__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ BigInt const_n;

// Precomputed table for base point G
__device__ ECPointJac *d_precomp_G = nullptr;

// Host function to initialize precomputed table
void init_precomputed_table_host();

__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__host__ __device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool is_zero(const BigInt *a) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->data[i]) return false;
    }
    return true;
}

__host__ __device__ __forceinline__ int get_bit(const BigInt *a, int i) {
    int word_idx = i >> 5; // i / 32
    int bit_idx = i & 31;  // i % 32
    if (word_idx >= BIGINT_WORDS) return 0;
    return (a->data[word_idx] >> bit_idx) & 1;
}

__device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile(
        "add.cc.u32 %0, %8, %16;\n\t"
        "addc.cc.u32 %1, %9, %17;\n\t"
        "addc.cc.u32 %2, %10, %18;\n\t"
        "addc.cc.u32 %3, %11, %19;\n\t"
        "addc.cc.u32 %4, %12, %20;\n\t"
        "addc.cc.u32 %5, %13, %21;\n\t"
        "addc.cc.u32 %6, %14, %22;\n\t"
        "addc.u32 %7, %15, %23;\n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

__device__ __forceinline__ void ptx_u256Sub(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile(
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.u32 %7, %15, %23;\n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

// Optimized multiply_bigint_by_const with unrolling
__device__ __forceinline__ void multiply_bigint_by_const(const BigInt *a, uint32_t c, uint32_t result[9]) {
    uint64_t carry = 0;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t prod = (uint64_t)a->data[i] * c + carry;
        result[i] = (uint32_t)prod;
        carry = prod >> 32;
    }
    result[8] = (uint32_t)carry;
}

// Optimized shift_left_word
__device__ __forceinline__ void shift_left_word(const BigInt *a, uint32_t result[9]) {
    result[0] = 0;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i+1] = a->data[i];
    }
}

// Optimized add_9word with unrolling
__device__ __forceinline__ void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    uint64_t carry = 0;
    
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        uint64_t sum = (uint64_t)r[i] + addend[i] + carry;
        r[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__device__ __forceinline__ void convert_9word_to_bigint(const uint32_t r[9], BigInt *res) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = r[i];
    }
}
__device__ __forceinline__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t prod[16] = {0};
    
    // Multiplication phase - optimized with better memory access patterns
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        uint32_t ai = a->data[i];
        
        // Use PTX for the inner loop multiplication
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t tmp = (uint64_t)prod[i + j] + (uint64_t)ai * b->data[j] + carry;
            prod[i + j] = (uint32_t)tmp;
            carry = tmp >> 32;
        }
        prod[i + 8] += (uint32_t)carry;
    }
    
    // Split into L and H
    BigInt L, H;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        L.data[i] = prod[i];
        H.data[i] = prod[i + 8];
    }
    
    // Initialize Rext with L
    uint32_t Rext[9];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        Rext[i] = L.data[i];
    }
    Rext[8] = 0;
    
    // Optimized: Add H * 977 and H * 2^32 in a single pass
    uint64_t carry = 0;
    
    // For i=0: just add H[0] * 977
    {
        uint64_t prod = (uint64_t)H.data[0] * 977;
        uint64_t sum = (uint64_t)Rext[0] + (uint32_t)prod;
        Rext[0] = (uint32_t)sum;
        carry = (prod >> 32) + (sum >> 32);
    }
    
    // For i=1 to 7: add H[i] * 977 + H[i-1] (shifted)
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        uint64_t prod = (uint64_t)H.data[i] * 977;
        uint64_t sum = (uint64_t)Rext[i] + (uint32_t)prod + H.data[i-1] + carry;
        Rext[i] = (uint32_t)sum;
        carry = (prod >> 32) + (sum >> 32);
    }
    
    // For i=8: add H[7] (shifted) + carry
    {
        uint64_t sum = (uint64_t)Rext[8] + H.data[7] + carry;
        Rext[8] = (uint32_t)sum;
        // Any overflow beyond this is handled next
    }
    
    // Handle overflow exactly as in original
    if (Rext[8]) {
        BigInt extraBI;
        init_bigint(&extraBI, Rext[8]);
        Rext[8] = 0;
        
        // Compute extra977 = extraBI * 977 (optimized for single word)
        uint64_t prod = (uint64_t)extraBI.data[0] * 977;
        uint32_t extra977_low = (uint32_t)prod;
        uint32_t extra977_high = (uint32_t)(prod >> 32);
        
        // Add extra977 to Rext[0] and Rext[1]
        uint64_t sum = (uint64_t)Rext[0] + extra977_low;
        Rext[0] = (uint32_t)sum;
        carry = (sum >> 32);
        
        sum = (uint64_t)Rext[1] + extra977_high + extraBI.data[0] + carry;
        Rext[1] = (uint32_t)sum;
        carry = (sum >> 32);
        
        // Propagate carry if needed
        for (int i = 2; i < 9 && carry; i++) {
            sum = (uint64_t)Rext[i] + carry;
            Rext[i] = (uint32_t)sum;
            carry = (sum >> 32);
        }
    }
    
    // Convert back to BigInt
    BigInt R_temp;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        R_temp.data[i] = Rext[i];
    }
    
    // Final reductions - exactly as original
    if (Rext[8] || compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    if (compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    
    copy_bigint(res, &R_temp);
}

__device__ __forceinline__ void sub_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt temp;
    if (compare_bigint(a, b) < 0) {
         BigInt sum;
         ptx_u256Add(&sum, a, &const_p);
         ptx_u256Sub(&temp, &sum, b);
    } else {
         ptx_u256Sub(&temp, a, b);
    }
    copy_bigint(res, &temp);
}

__device__ void mod_inverse_fast(BigInt *res, const BigInt *a) {
    BigInt R0, R1;
    init_bigint(&R0, 1);
    copy_bigint(&R1, a);
    
    // Process from MSB (bit 255) to LSB (bit 0)
    // p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    
    // Handle bits 255-33 (all are 1s in p-2)
    #pragma unroll
    for (int i = 255; i >= 33; i--) {
        // bit = 1, so always:
        mul_mod_device(&R0, &R0, &R1);    // R0 = R0 * R1
        mul_mod_device(&R1, &R1, &R1);    // R1 = R1^2
    }
    
    // Handle bit 32 (which is 0 in p-2)
    mul_mod_device(&R1, &R0, &R1);       // R1 = R0 * R1
    mul_mod_device(&R0, &R0, &R0);       // R0 = R0^2
    
    // Handle bits 31-0 with the correct pattern
    // The low 32 bits of p-2 are 0xFFFFFC2D
    // We need to process from bit 31 down to bit 0
    
    #pragma unroll 32
    for (int i = 31; i >= 0; i--) {
        uint32_t bit = (0xFFFFFC2D >> i) & 1;
        
        if (bit == 1) {
            mul_mod_device(&R0, &R0, &R1);    // R0 = R0 * R1
            mul_mod_device(&R1, &R1, &R1);    // R1 = R1^2
        } else {
            mul_mod_device(&R1, &R0, &R1);    // R1 = R0 * R1
            mul_mod_device(&R0, &R0, &R0);    // R0 = R0^2
        }
    }
    
    copy_bigint(res, &R0);
}

__device__ __forceinline__ void point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
}

__device__ __forceinline__ void point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    copy_bigint(&dest->X, &src->X);
    copy_bigint(&dest->Y, &src->Y);
    copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P); // 声明
__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q); // 声明

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P) {
    if (P->infinity || is_zero(&P->Y)) {
        point_set_infinity_jac(R);
        return;
    }
    BigInt A, B, C, D, X3, Y3, Z3, temp, temp2;
    mul_mod_device(&A, &P->Y, &P->Y);
    mul_mod_device(&temp, &P->X, &A);
    init_bigint(&temp2, 4);
    mul_mod_device(&B, &temp, &temp2);
    mul_mod_device(&temp, &A, &A);
    init_bigint(&temp2, 8);
    mul_mod_device(&C, &temp, &temp2);
    mul_mod_device(&temp, &P->X, &P->X);
    init_bigint(&temp2, 3);
    mul_mod_device(&D, &temp, &temp2);
    BigInt D2, two, twoB;
    mul_mod_device(&D2, &D, &D);
    init_bigint(&two, 2);
    mul_mod_device(&twoB, &B, &two);
    sub_mod_device(&X3, &D2, &twoB);
    sub_mod_device(&temp, &B, &X3);
    mul_mod_device(&temp, &D, &temp);
    sub_mod_device(&Y3, &temp, &C);
    init_bigint(&temp, 2);
    mul_mod_device(&temp, &temp, &P->Y);
    mul_mod_device(&Z3, &temp, &P->Z);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    if (P->infinity) { point_copy_jac(R, Q); return; }
    if (Q->infinity) { point_copy_jac(R, P); return; }

    BigInt Z1Z1, Z2Z2, U1, U2, S1, S2, H, R_big, H2, H3, U1H2, X3, Y3, Z3, temp;
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&Z2Z2, &Q->Z, &Q->Z);
    mul_mod_device(&U1, &P->X, &Z2Z2);
    mul_mod_device(&U2, &Q->X, &Z1Z1);
    BigInt Z2_cubed, Z1_cubed;
    mul_mod_device(&temp, &Z2Z2, &Q->Z); copy_bigint(&Z2_cubed, &temp);
    mul_mod_device(&temp, &Z1Z1, &P->Z); copy_bigint(&Z1_cubed, &temp);
    mul_mod_device(&S1, &P->Y, &Z2_cubed);
    mul_mod_device(&S2, &Q->Y, &Z1_cubed);

    if (compare_bigint(&U1, &U2) == 0) {
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
            return;
        } else {
            double_point_jac(R, P);
            return;
        }
    }
    sub_mod_device(&H, &U2, &U1);
    sub_mod_device(&R_big, &S2, &S1);
    mul_mod_device(&H2, &H, &H);
    mul_mod_device(&H3, &H2, &H);
    mul_mod_device(&U1H2, &U1, &H2);
    BigInt R2, two, twoU1H2;
    mul_mod_device(&R2, &R_big, &R_big);
    init_bigint(&two, 2);
    mul_mod_device(&twoU1H2, &U1H2, &two);
    sub_mod_device(&temp, &R2, &H3);
    sub_mod_device(&X3, &temp, &twoU1H2);
    sub_mod_device(&temp, &U1H2, &X3);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&Y3, &S1, &H3);
    sub_mod_device(&Y3, &temp, &Y3);
    mul_mod_device(&temp, &P->Z, &Q->Z);
    mul_mod_device(&Z3, &temp, &H);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}


__device__ void jacobian_to_affine_fast(ECPoint *R, const ECPointJac *P) {
    if (P->infinity) {
        R->infinity = true;
        init_bigint(&R->x, 0);
        init_bigint(&R->y, 0);
        return;
    }
    BigInt Zinv, Zinv2, Zinv3;
    mod_inverse_fast(&Zinv, &P->Z);  // <-- Use fast inverse here
    mul_mod_device(&Zinv2, &Zinv, &Zinv);
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);
    mul_mod_device(&R->x, &P->X, &Zinv2);
    mul_mod_device(&R->y, &P->Y, &Zinv3);
    R->infinity = false;
}


// General scalar multiplication for arbitrary points (using shared memory)
__device__ void scalar_multiply_jac_device(ECPointJac *result, const ECPointJac *point, const BigInt *scalar) {
    const int SHARED_WINDOW_SIZE = 4;
    const int SHARED_PRECOMP_SIZE = 1 << SHARED_WINDOW_SIZE;
    
    // Use shared memory for precomputed points
    __shared__ ECPointJac shared_precomp[1 << SHARED_WINDOW_SIZE];
    
    // Collaborative precomputation using threads in the block
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Each thread computes some precomputed points
    for (int i = tid; i < SHARED_PRECOMP_SIZE; i += block_size) {
        if (i == 0) {
            point_set_infinity_jac(&shared_precomp[0]);
        } else if (i == 1) {
            point_copy_jac(&shared_precomp[1], point);
        } else {
            add_point_jac(&shared_precomp[i], &shared_precomp[i-1], point);
        }
    }
    
    // Ensure all threads have finished precomputation
    __syncthreads();
    
    // Find the highest non-zero bit
    int highest_bit = BIGINT_WORDS * 32 - 1;
    for (; highest_bit >= 0; highest_bit--) {
        if (get_bit(scalar, highest_bit)) break;
    }
    
    if (highest_bit < 0) {
        point_set_infinity_jac(result);
        return;
    }
    
    // Initialize result
    ECPointJac res;
    point_set_infinity_jac(&res);
    
    // Process scalar in windows of SHARED_WINDOW_SIZE bits
    int i = highest_bit;
    while (i >= 0) {
        // Determine window size for this iteration
        int window_bits = (i >= SHARED_WINDOW_SIZE - 1) ? SHARED_WINDOW_SIZE : (i + 1);
        
        // Double 'window_bits' times
        for (int j = 0; j < window_bits; j++) {
            double_point_jac(&res, &res);
        }
        
        // Extract window value
        int window_value = 0;
        for (int j = 0; j < window_bits; j++) {
            if (i - j >= 0 && get_bit(scalar, i - j)) {
                window_value |= (1 << (window_bits - 1 - j));
            }
        }
        
        // Add precomputed point if window value is non-zero
        if (window_value > 0) {
            add_point_jac(&res, &res, &shared_precomp[window_value]);
        }
        
        i -= window_bits;
    }
    
    point_copy_jac(result, &res);
}

// Helper function to check if a point equals the base point G
__device__ bool is_base_point_G(const ECPointJac *point) {
    return (compare_bigint(&point->X, &const_G_jacobian.X) == 0 &&
            compare_bigint(&point->Y, &const_G_jacobian.Y) == 0 &&
            compare_bigint(&point->Z, &const_G_jacobian.Z) == 0 &&
            point->infinity == const_G_jacobian.infinity);
}

// Modified scalar multiplication to use global memory
__device__ void scalar_multiply_G_precomputed_large(ECPointJac *result, const BigInt *scalar) {
    ECPointJac res;
    point_set_infinity_jac(&res);
    
    // Process from highest bit to lowest in PRECOMP_BITS chunks
    for (int i = 256 - PRECOMP_BITS; i >= 0; i -= PRECOMP_BITS) {
        // Double for the window size
        for (int j = 0; j < PRECOMP_BITS; j++) {
            double_point_jac(&res, &res);
        }
        
        // Extract window value correctly (big-endian style)
        uint32_t window = 0;
        for (int j = PRECOMP_BITS - 1; j >= 0; j--) {
            window <<= 1;
            if (i + j < 256 && get_bit(scalar, i + j)) {
                window |= 1;
            }
        }
        
        // Add precomputed point
        if (window > 0) {
            add_point_jac(&res, &res, &d_precomp_G[window]);
        }
    }
    
    point_copy_jac(result, &res);
}

// Cleanup function (call at program end)
void cleanup_precomputed_table() {
    ECPointJac *h_precomp_G_ptr;
    CHECK_CUDA(cudaMemcpyFromSymbol(&h_precomp_G_ptr, d_precomp_G, sizeof(ECPointJac*)));
    CHECK_CUDA(cudaFree(h_precomp_G_ptr));
}
// Main scalar multiplication function that automatically chooses optimal method
__device__ void scalar_multiply_optimized(ECPointJac *result, const ECPointJac *point, const BigInt *scalar) {
    if (is_base_point_G(point)) {
        // Use precomputed table for base point G
        scalar_multiply_G_precomputed_large(result, scalar);
    } else {
        // Use general method with shared memory
		printf("manual scalar");
        scalar_multiply_jac_device(result, point, scalar);
    }
}

#endif

// Host function implementations (should be in a separate .cu file)

// Host-side BigInt and ECPoint operations (simplified versions)
void host_init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

void host_copy_bigint(BigInt *dest, const BigInt *src) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

void host_point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
}

void host_point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    host_copy_bigint(&dest->X, &src->X);
    host_copy_bigint(&dest->Y, &src->Y);
    host_copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}

// Host-side point addition and doubling (you can use your existing CPU implementations)
// These are simplified placeholders - replace with your actual host implementations
void host_add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    // TODO: Implement host-side point addition
    // This should be equivalent to your device add_point_jac function
    // but using host-side modular arithmetic
}

void host_double_point_jac(ECPointJac *R, const ECPointJac *P) {
    // TODO: Implement host-side point doubling
    // This should be equivalent to your device double_point_jac function
    // but using host-side modular arithmetic
}

// GPU kernel to compute precomputed table
__global__ void compute_precomputed_table_gpu(ECPointJac *temp_table) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PRECOMP_SIZE) return;
    
    if (idx == 0) {
        // table[0] = infinity (0*G)
        point_set_infinity_jac(&temp_table[0]);
        return;
    }
    
    // Each thread computes idx*G using binary representation
    ECPointJac result, base;
    point_set_infinity_jac(&result);
    point_copy_jac(&base, &const_G_jacobian);
    
    int temp_idx = idx;
    while (temp_idx > 0) {
        if (temp_idx & 1) {
            add_point_jac(&result, &result, &base);
        }
        double_point_jac(&base, &base);
        temp_idx >>= 1;
    }
    
    point_copy_jac(&temp_table[idx], &result);
}

// New kernel for batch computation
__global__ void compute_batch_precomp(ECPointJac *batch, int start_idx, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    if (idx == 0 && start_idx == 2) {
        // First point in first batch: 2*G = G + G
        add_point_jac(&batch[1], &batch[0], &const_G_jacobian);
    } else if (idx > 0) {
        // Each subsequent point: (start_idx + idx)*G = (start_idx + idx - 1)*G + G
        add_point_jac(&batch[idx], &batch[idx - 1], &const_G_jacobian);
    }
}


// Alternative: More robust kernel that processes in smaller chunks
__global__ void compute_batch_precomp_kernel_robust(ECPointJac *table, int start_idx, int end_idx) {
    // Process in chunks of 64 to avoid timeouts
    const int CHUNK_SIZE = 64;
    
    for (int chunk_start = start_idx; chunk_start < end_idx; chunk_start += CHUNK_SIZE) {
        int chunk_end = min(chunk_start + CHUNK_SIZE, end_idx);
        
        // Compute this chunk
        for (int i = chunk_start; i < chunk_end; i++) {
            add_point_jac(&table[i], &table[i - 1], &const_G_jacobian);
        }
        
        // Allow other kernels to run (prevent timeout)
        __syncthreads();
    }
}
void init_precomputed_table_host() {
    printf("Initializing precomputed table for base point G...\n");
    printf("Table size: %d entries (%.2f MB)\n", PRECOMP_SIZE, 
           (PRECOMP_SIZE * sizeof(ECPointJac)) / (1024.0 * 1024.0));
    
    // Allocate global memory for the table
    ECPointJac *h_precomp_G_ptr;
    CHECK_CUDA(cudaMalloc(&h_precomp_G_ptr, PRECOMP_SIZE * sizeof(ECPointJac)));
    
    // Copy the pointer to device symbol
    CHECK_CUDA(cudaMemcpyToSymbol(d_precomp_G, &h_precomp_G_ptr, sizeof(ECPointJac*)));
    
    // Compute table on GPU using simple sequential approach
    ECPointJac *h_table = (ECPointJac*)malloc(PRECOMP_SIZE * sizeof(ECPointJac));
    
    // Initialize ALL entries to infinity first (for safety)
    for (int i = 0; i < PRECOMP_SIZE; i++) {
        host_point_set_infinity_jac(&h_table[i]);
    }
    
    // Set up first two entries
    ECPointJac G_host;
    cudaMemcpyFromSymbol(&G_host, const_G_jacobian, sizeof(ECPointJac));
    host_point_copy_jac(&h_table[1], &G_host);
    
    // Copy initial entries to GPU
    CHECK_CUDA(cudaMemcpy(h_precomp_G_ptr, h_table, 2 * sizeof(ECPointJac), cudaMemcpyHostToDevice));
    
    // Compute remaining entries in batches
    const int BATCH_SIZE = (PRECOMP_SIZE > 16384) ? 256 : 1024;  // Smaller batches for large tables
    int total_computed = 2;  // Track actual progress
    
    for (int i = 2; i < PRECOMP_SIZE; i += BATCH_SIZE) {
        int batch_end = min(i + BATCH_SIZE, PRECOMP_SIZE);
        int batch_size = batch_end - i;
        
        printf("Processing batch: %d to %d (size: %d)\n", i, batch_end, batch_size);
        
        // Copy current state to GPU if needed
        if (i > 2) {
            CHECK_CUDA(cudaMemcpy(&h_precomp_G_ptr[i-1], &h_table[i-1], 
                                 sizeof(ECPointJac), cudaMemcpyHostToDevice));
        }
        
        // Compute batch on GPU
        compute_batch_precomp_kernel_robust<<<1, 1>>>(h_precomp_G_ptr, i, batch_end);
        
        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy results back
        CHECK_CUDA(cudaMemcpy(&h_table[i], &h_precomp_G_ptr[i], 
                             batch_size * sizeof(ECPointJac), cudaMemcpyDeviceToHost));
        
        // Verify batch was actually computed
        bool batch_valid = true;
        for (int j = i; j < batch_end && batch_valid; j++) {
            if (h_table[j].infinity) {
                printf("ERROR: Entry %d is still infinity after computation!\n", j);
                batch_valid = false;
            }
        }
        
        if (!batch_valid) {
            printf("Batch computation failed at entry %d\n", i);
            total_computed = i;
            break;
        }
        
        total_computed = batch_end;
        
        //if (i % 1024 == 0 || i == 2) {
            printf("Computed %d entries (%.1f%%)\n", total_computed, 
                   100.0 * total_computed / PRECOMP_SIZE);
        //}
    }
    
    printf("\nActual entries computed: %d out of %d\n", total_computed, PRECOMP_SIZE);
    
    if (total_computed < PRECOMP_SIZE) {
        printf("ERROR: Table incomplete! Only computed %d entries (%.1f%%)\n", 
               total_computed, 100.0 * total_computed / PRECOMP_SIZE);
        printf("Reducing PRECOMP_BITS is recommended.\n");
        exit(1);
    }
    
    // Copy final table to GPU
    CHECK_CUDA(cudaMemcpy(h_precomp_G_ptr, h_table, PRECOMP_SIZE * sizeof(ECPointJac), 
                         cudaMemcpyHostToDevice));
    
    free(h_table);
    printf("Precomputed table initialized successfully in global memory\n");
}

// Utility function to initialize secp256k1 constants
void init_secp256k1_constants() {
    // secp256k1 prime p = 2^256 - 2^32 - 977
    BigInt p;
    uint32_t p_data[8] = {
        0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };
    memcpy(p.data, p_data, sizeof(p_data));
    cudaMemcpyToSymbol(const_p, &p, sizeof(BigInt));
    
    // secp256k1 order n
    BigInt n;
    uint32_t n_data[8] = {
        0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
        0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };
    memcpy(n.data, n_data, sizeof(n_data));
    cudaMemcpyToSymbol(const_n, &n, sizeof(BigInt));
    
    // secp256k1 base point G in Jacobian coordinates
    ECPointJac G_jac;
    uint32_t Gx_data[8] = {
        0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
        0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
    };
    uint32_t Gy_data[8] = {
        0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
        0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
    };
    memcpy(G_jac.X.data, Gx_data, sizeof(Gx_data));
    memcpy(G_jac.Y.data, Gy_data, sizeof(Gy_data));
    host_init_bigint(&G_jac.Z, 1);
    G_jac.infinity = false;
    
    cudaMemcpyToSymbol(const_G_jacobian, &G_jac, sizeof(ECPointJac));
    
    printf("secp256k1 constants initialized on GPU\n");
}

// Complete initialization function to call from main
void initialize_secp256k1_gpu() {
    printf("Initializing secp256k1 for GPU...\n");
    
    // Initialize constants first
    init_secp256k1_constants();
    
    // Then initialize precomputed table
    init_precomputed_table_host();
    
    printf("secp256k1 GPU initialization complete!\n");
}

// Example usage kernel
__global__ void test_scalar_multiplication(ECPoint *results, BigInt *scalars, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    ECPointJac result_jac;
    
    // Use optimized scalar multiplication (automatically chooses best method)
    scalar_multiply_optimized(&result_jac, &const_G_jacobian, &scalars[idx]);
    
    // Convert to affine coordinates for output
    jacobian_to_affine_fast(&results[idx], &result_jac);
}

//Author telegram: https://t.me/nmn5436