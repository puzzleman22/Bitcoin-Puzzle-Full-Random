#include "secp256k1.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <fstream>
#pragma once
#include <stdint.h>
#include <curand_kernel.h>

// Optimized rotate right for SHA-256
__device__ inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256(const uint8_t* data, int len, uint8_t hash[32]) {
    const uint32_t K[] = {
        0x428a2f98ul,0x71374491ul,0xb5c0fbcful,0xe9b5dba5ul,
        0x3956c25bul,0x59f111f1ul,0x923f82a4ul,0xab1c5ed5ul,
        0xd807aa98ul,0x12835b01ul,0x243185beul,0x550c7dc3ul,
        0x72be5d74ul,0x80deb1feul,0x9bdc06a7ul,0xc19bf174ul,
        0xe49b69c1ul,0xefbe4786ul,0x0fc19dc6ul,0x240ca1ccul,
        0x2de92c6ful,0x4a7484aaul,0x5cb0a9dcul,0x76f988daul,
        0x983e5152ul,0xa831c66dul,0xb00327c8ul,0xbf597fc7ul,
        0xc6e00bf3ul,0xd5a79147ul,0x06ca6351ul,0x14292967ul,
        0x27b70a85ul,0x2e1b2138ul,0x4d2c6dfcul,0x53380d13ul,
        0x650a7354ul,0x766a0abbul,0x81c2c92eul,0x92722c85ul,
        0xa2bfe8a1ul,0xa81a664bul,0xc24b8b70ul,0xc76c51a3ul,
        0xd192e819ul,0xd6990624ul,0xf40e3585ul,0x106aa070ul,
        0x19a4c116ul,0x1e376c08ul,0x2748774cul,0x34b0bcb5ul,
        0x391c0cb3ul,0x4ed8aa4aul,0x5b9cca4ful,0x682e6ff3ul,
        0x748f82eeul,0x78a5636ful,0x84c87814ul,0x8cc70208ul,
        0x90befffaul,0xa4506cebul,0xbef9a3f7ul,0xc67178f2ul
    };

    uint32_t h[8] = {
        0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
        0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul
    };

    // Optimized for 33-byte input (compressed pubkey)
    uint8_t full[64] = {0};
    
    // Copy input data
    #pragma unroll
    for (int i = 0; i < len; ++i) full[i] = data[i];
    full[len] = 0x80;
    
    // Add length in bits (big-endian) at the end
    uint64_t bit_len = (uint64_t)len * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        full[63 - i] = bit_len >> (8 * i);
    }

    // Process single block (we know it's only one block for 33 bytes)
    uint32_t w[64];
    
    // Load message schedule with proper byte order
    #pragma unroll 16
    for (int i = 0; i < 16; ++i) {
        w[i] = (full[4 * i] << 24) | (full[4 * i + 1] << 16) |
               (full[4 * i + 2] << 8) | full[4 * i + 3];
    }
    
    // Extend message schedule
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hval = h[7];

    // Main compression loop
    #pragma unroll 8
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = hval + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        hval = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hval;

    // Output hash (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        hash[4 * i + 0] = (h[i] >> 24) & 0xFF;
        hash[4 * i + 1] = (h[i] >> 16) & 0xFF;
        hash[4 * i + 2] = (h[i] >> 8) & 0xFF;
        hash[4 * i + 3] = (h[i] >> 0) & 0xFF;
    }
}

__device__ void ripemd160(const uint8_t* msg, uint8_t* out) {
    // RIPEMD-160 constants
    const uint32_t K1[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
    const uint32_t K2[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};
    
    // Message schedule for left and right lines
    const int ZL[80] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
    };
    
    const int ZR[80] = {
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
    };
    
    // Shift amounts for left and right lines
    const int SL[80] = {
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
    };
    
    const int SR[80] = {
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
    };
    
    // Initialize hash values
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;
    
    // Prepare message: add padding and length
    uint8_t buffer[64];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        buffer[i] = msg[i];
    }
    
    // Add padding
    buffer[32] = 0x80;
    #pragma unroll
    for (int i = 33; i < 56; i++) {
        buffer[i] = 0x00;
    }
    
    // Add length (256 bits = 32 bytes)
    uint64_t bitlen = 256;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        buffer[56 + i] = (bitlen >> (i * 8)) & 0xFF;
    }
    
    // Convert buffer to 32-bit data (little-endian)
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        X[i] = ((uint32_t)buffer[i*4]) | 
               ((uint32_t)buffer[i*4 + 1] << 8) | 
               ((uint32_t)buffer[i*4 + 2] << 16) | 
               ((uint32_t)buffer[i*4 + 3] << 24);
    }
    
    // Working variables
    uint32_t AL = h0, BL = h1, CL = h2, DL = h3, EL = h4;
    uint32_t AR = h0, BR = h1, CR = h2, DR = h3, ER = h4;
    
    // Process message in 5 rounds of 16 operations each
    #pragma unroll 10
    for (int j = 0; j < 80; j++) {
        uint32_t T;
        
        // Left line
        if (j < 16) {
            T = AL + (BL ^ CL ^ DL) + X[ZL[j]] + K1[0];
        } else if (j < 32) {
            T = AL + ((BL & CL) | (~BL & DL)) + X[ZL[j]] + K1[1];
        } else if (j < 48) {
            T = AL + ((BL | ~CL) ^ DL) + X[ZL[j]] + K1[2];
        } else if (j < 64) {
            T = AL + ((BL & DL) | (CL & ~DL)) + X[ZL[j]] + K1[3];
        } else {
            T = AL + (BL ^ (CL | ~DL)) + X[ZL[j]] + K1[4];
        }
        T = ((T << SL[j]) | (T >> (32 - SL[j]))) + EL;
        AL = EL; EL = DL; DL = (CL << 10) | (CL >> 22); CL = BL; BL = T;
        
        // Right line
        if (j < 16) {
            T = AR + (BR ^ (CR | ~DR)) + X[ZR[j]] + K2[0];
        } else if (j < 32) {
            T = AR + ((BR & DR) | (CR & ~DR)) + X[ZR[j]] + K2[1];
        } else if (j < 48) {
            T = AR + ((BR | ~CR) ^ DR) + X[ZR[j]] + K2[2];
        } else if (j < 64) {
            T = AR + ((BR & CR) | (~BR & DR)) + X[ZR[j]] + K2[3];
        } else {
            T = AR + (BR ^ CR ^ DR) + X[ZR[j]] + K2[4];
        }
        T = ((T << SR[j]) | (T >> (32 - SR[j]))) + ER;
        AR = ER; ER = DR; DR = (CR << 10) | (CR >> 22); CR = BR; BR = T;
    }
    
    // Add results
    uint32_t T = h1 + CL + DR;
    h1 = h2 + DL + ER;
    h2 = h3 + EL + AR;
    h3 = h4 + AL + BR;
    h4 = h0 + BL + CR;
    h0 = T;
    
    // Convert hash to bytes (little-endian)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i]      = (h0 >> (i * 8)) & 0xFF;
        out[i + 4]  = (h1 >> (i * 8)) & 0xFF;
        out[i + 8]  = (h2 >> (i * 8)) & 0xFF;
        out[i + 12] = (h3 >> (i * 8)) & 0xFF;
        out[i + 16] = (h4 >> (i * 8)) & 0xFF;
    }
}

__device__ __forceinline__ void hash160(const uint8_t* data, int len, uint8_t out[20]) {
    uint8_t sha[32];
    sha256(data, len, sha);
    ripemd160(sha, out);
}

void init_gpu_constants() {
    // 1) 定义 p_host
    const BigInt p_host = {
        { 0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
          0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };
    // 2) 定义 G_jacobian_host
    const ECPointJac G_jacobian_host = {
        {{ 0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
                0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E }},
        {{ 0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
                0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77 }},
        {{ 1, 0, 0, 0, 0, 0, 0, 0 }}
    };
    // 3) 定义 n_host
    const BigInt n_host = {
        { 0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
          0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };

    // 然后再复制到 __constant__ 内存
    CHECK_CUDA(cudaMemcpyToSymbol(const_p, &p_host, sizeof(BigInt)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_G_jacobian, &G_jacobian_host, sizeof(ECPointJac)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_n, &n_host, sizeof(BigInt)));
}

std::string generate_random_hex(size_t length) {
    const char hex_chars[] = "0123456789abcdef";
    std::string hex_string;
    hex_string.reserve(length); // Reserve space to avoid reallocations

    for (size_t i = 0; i < length; ++i) {
        hex_string += hex_chars[rand() % 16];
    }

    return hex_string;
}

std::string zfill(const std::string& input, size_t total_length) {
    if (input.length() >= total_length) {
        return input;
    }
    return std::string(total_length - input.length(), '0') + input;
}

__device__ __forceinline__ uint8_t get_byte(const BigInt& a, int i) {
    // Convert to big-endian byte order
    int word_index = 7 - (i / 4);       // reverse word order
    int byte_index = 3 - (i % 4);       // reverse byte order within word
    return (a.data[word_index] >> (8 * byte_index)) & 0xFF;
}

__device__ __forceinline__ void coords_to_compressed_pubkey(const BigInt& x, const BigInt& y, uint8_t* pubkey) {
    // Prefix: 0x02 if y is even, 0x03 if y is odd
    pubkey[0] = (y.data[0] & 1) ? 0x03 : 0x02;

    // Copy x coordinate (32 bytes) with unrolling
    #pragma unroll 8
    for (int i = 0; i < 32; i++) {
        pubkey[1 + i] = get_byte(x, i);
    }
}

// Convert a hex character to its numeric value
__device__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0; // Invalid character
}

// Convert hex string to bytes
__device__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    #pragma unroll 8
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}

// Compare two hash160 arrays
__device__ __forceinline__ bool compare_hash160(const uint8_t* hash1, const uint8_t* hash2) {
    // Use 32-bit comparisons for speed
    uint32_t* h1 = (uint32_t*)hash1;
    uint32_t* h2 = (uint32_t*)hash2;
    
    return (h1[0] == h2[0]) && (h1[1] == h2[1]) && 
           (h1[2] == h2[2]) && (h1[3] == h2[3]) && 
           (h1[4] == h2[4]);
}

// Compare hash160 with hex string directly - optimized version
__device__ __forceinline__ bool compare_hash160_with_hex(const uint8_t* hash, const char* hex_str) {
    // Early exit on first mismatch
    #pragma unroll 5
    for (int i = 0; i < 20; i++) {
        uint8_t byte = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                       hex_char_to_byte(hex_str[i * 2 + 1]);
        if (hash[i] != byte) {
            return false;
        }
    }
    return true;
}

// Compare two BigInts: returns -1 if a < b, 0 if a == b, 1 if a > b
__device__ __forceinline__ int bigint_compare(const BigInt* a, const BigInt* b) {
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

// Subtract b from a (a must be >= b), result stored in result
__device__ void bigint_subtract(const BigInt* a, const BigInt* b, BigInt* result) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t temp = (uint64_t)a->data[i] - b->data[i] - borrow;
        result->data[i] = (uint32_t)temp;
        borrow = (temp >> 32) & 1;
    }
}

// Add b to a, result stored in result
__device__ void bigint_add(const BigInt* a, const BigInt* b, BigInt* result) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t temp = (uint64_t)a->data[i] + b->data[i] + carry;
        result->data[i] = (uint32_t)temp;
        carry = temp >> 32;
    }
}

// Fast PRNG with better mixing
__device__ __forceinline__ uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}
__device__ void generate_random_bigint_range_optimized(uint64_t* rng_state, const BigInt* min, const BigInt* max, BigInt* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate range = max - min + 1
    BigInt range, range_plus_one;
    bigint_subtract(max, min, &range);
    BigInt one = {1, 0, 0, 0, 0, 0, 0, 0};
    bigint_add(&range, &one, &range_plus_one);
    
    // Find the bit length of range_plus_one
    int highest_word = 7;
    while (highest_word > 0 && range_plus_one.data[highest_word] == 0)
        highest_word--;
    
    int bit_length = highest_word * 32;
    if (range_plus_one.data[highest_word] != 0) {
        bit_length += 32 - __clz(range_plus_one.data[highest_word]);
    }
    
    // Enhanced multi-source entropy mixing
    uint64_t entropy = *rng_state;
    
    // Mix in thread-specific entropy sources
    entropy ^= ((uint64_t)tid << 32) | tid;
    entropy ^= ((uint64_t)blockIdx.x << 48) | ((uint64_t)blockIdx.y << 32) | 
               ((uint64_t)threadIdx.x << 16) | threadIdx.y;
    entropy ^= clock64();  // Hardware cycle counter for temporal entropy
    
    // Apply cryptographic-quality mixing (based on SplitMix64)
    entropy += 0x9e3779b97f4a7c15ULL;
    entropy = (entropy ^ (entropy >> 30)) * 0xbf58476d1ce4e5b9ULL;
    entropy = (entropy ^ (entropy >> 27)) * 0x94d049bb133111ebULL;
    entropy = entropy ^ (entropy >> 31);
    
    // Update RNG state with mixed entropy
    *rng_state = entropy;
    
    // Rejection sampling with optimized bit generation
    BigInt candidate;
    int max_attempts = 100;  // Prevent infinite loops
    
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        // Generate random bits efficiently
        uint64_t current_entropy = entropy;
        
        // Clear candidate
        for (int i = 0; i < 8; i++) {
            candidate.data[i] = 0;
        }
        
        // Generate only the needed number of bits
        int bits_generated = 0;
        int word_idx = 0;
        
        while (bits_generated < bit_length && word_idx < 8) {
            // Generate next 64-bit entropy chunk
            current_entropy += 0x9e3779b97f4a7c15ULL;
            current_entropy = (current_entropy ^ (current_entropy >> 30)) * 0xbf58476d1ce4e5b9ULL;
            current_entropy = (current_entropy ^ (current_entropy >> 27)) * 0x94d049bb133111ebULL;
            current_entropy = current_entropy ^ (current_entropy >> 31);
            
            // Fill current word
            if (bits_generated + 32 <= bit_length) {
                candidate.data[word_idx] = (uint32_t)current_entropy;
                bits_generated += 32;
            } else {
                // Partial word - mask unused bits
                int remaining_bits = bit_length - bits_generated;
                uint32_t mask = (1U << remaining_bits) - 1;
                candidate.data[word_idx] = (uint32_t)current_entropy & mask;
                bits_generated = bit_length;
            }
            
            word_idx++;
            current_entropy >>= 32;  // Use upper 32 bits for next iteration
        }
        
        // Check if candidate < range_plus_one
        bool valid = true;
        for (int i = 7; i >= 0; i--) {
            if (candidate.data[i] > range_plus_one.data[i]) {
                valid = false;
                break;
            } else if (candidate.data[i] < range_plus_one.data[i]) {
                break;
            }
        }
        
        if (valid) {
            // Add min to get final result
            bigint_add(&candidate, min, result);
            return;
        }
        
        // Update entropy for next attempt
        entropy = current_entropy;
    }
    
    // Fallback: use simple reduction (introduces slight bias but ensures termination)
    // This should rarely be reached with proper bit_length calculation
    
    // Generate a full random BigInt
    uint64_t fallback_entropy = entropy;
    for (int i = 0; i < 8; i++) {
        fallback_entropy += 0x9e3779b97f4a7c15ULL;
        fallback_entropy = (fallback_entropy ^ (fallback_entropy >> 30)) * 0xbf58476d1ce4e5b9ULL;
        fallback_entropy = (fallback_entropy ^ (fallback_entropy >> 27)) * 0x94d049bb133111ebULL;
        fallback_entropy = fallback_entropy ^ (fallback_entropy >> 31);
        
        candidate.data[i] = (uint32_t)fallback_entropy;
        fallback_entropy >>= 32;
    }
    
    // Simple modular reduction using repeated subtraction
    // While candidate >= range_plus_one, subtract range_plus_one
    while (true) {
        bool candidate_ge_range = false;
        
        // Check if candidate >= range_plus_one
        for (int i = 7; i >= 0; i--) {
            if (candidate.data[i] > range_plus_one.data[i]) {
                candidate_ge_range = true;
                break;
            } else if (candidate.data[i] < range_plus_one.data[i]) {
                break;
            }
            // If equal, continue to next word
            if (i == 0) candidate_ge_range = true; // All words equal
        }
        
        if (!candidate_ge_range) break;
        
        // Subtract range_plus_one from candidate
        bigint_subtract(&candidate, &range_plus_one, &candidate);
    }
    
    // Add min to get final result
    bigint_add(&candidate, min, result);
}

// Convert hex string to BigInt - optimized
__device__ void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    // Initialize all data to 0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;
    
    // Process hex string from right to left
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        uint8_t val = hex_char_to_byte(hex_str[i]);
        
        bigint->data[word_idx] |= ((uint32_t)val << bit_offset);
        
        bit_offset += 4;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

// Convert BigInt to hex string - optimized
__device__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 28; j >= 0; j -= 4) {
            uint8_t nibble = (bigint->data[i] >> j) & 0xF;
            if (nibble != 0 || !leading_zero || (i == 0 && j == 0)) {
                hex_str[idx++] = hex_chars[nibble];
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}

// Optimized byte to hex conversion
__device__ __forceinline__ void byte_to_hex(uint8_t byte, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[(byte >> 4) & 0xF];
    out[1] = hex_chars[byte & 0xF];
}

__device__ void hash160_to_hex(uint8_t* hash, char* hex_str) {
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        byte_to_hex(hash[i], &hex_str[i * 2]);
    }
    hex_str[40] = '\0';
}

#define HEX_LENGTH 64  // 64 hex characters

// Optimized hex rotation functions
__device__ __forceinline__ void hex_rotate_right_by_one(char* hex_str) {
    int actual_length = 0;
    #pragma unroll 8
    for (int i = 0; i < HEX_LENGTH; i++) {
        if (hex_str[i] == '\0') {
            actual_length = i;
            break;
        }
    }
    if (actual_length == 0) {
        actual_length = HEX_LENGTH;
    }
    
    if (actual_length <= 1) return;
    
    // Find the first occurrence of '1'
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (hex_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    int rotation_start = first_one + 1;
    int rotation_length = actual_length - rotation_start;
    
    if (rotation_length <= 1) return;
    
    char last_char = hex_str[rotation_start + rotation_length - 1];
    
    // Use memmove for better performance
    for (int i = rotation_length - 1; i > 0; i--) {
        hex_str[rotation_start + i] = hex_str[rotation_start + i - 1];
    }
    
    hex_str[rotation_start] = last_char;
}

__device__ __forceinline__ void hex_rotate_left_by_one(char* hex_str) {
    int actual_length = 0;
    #pragma unroll 8
    for (int i = 0; i < HEX_LENGTH; i++) {
        if (hex_str[i] == '\0') {
            actual_length = i;
            break;
        }
    }
    if (actual_length == 0) {
        actual_length = HEX_LENGTH;
    }
    
    if (actual_length <= 1) return;
    
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (hex_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    int rotation_start = first_one + 1;
    int rotation_length = actual_length - rotation_start;
    
    if (rotation_length <= 1) return;
    
    char first_char = hex_str[rotation_start];
    
    for (int i = 0; i < rotation_length - 1; i++) {
        hex_str[rotation_start + i] = hex_str[rotation_start + i + 1];
    }
    
    hex_str[rotation_start + rotation_length - 1] = first_char;
}

// Use lookup table for hex increment/decrement
__constant__ char hex_inc_table[16] = {'1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','0'};
__constant__ char hex_dec_table[16] = {'f','0','1','2','3','4','5','6','7','8','9','a','b','c','d','e'};

__device__ __forceinline__ char hex_increment(char c) {
    if (c >= '0' && c <= '9') return hex_inc_table[c - '0'];
    if (c >= 'a' && c <= 'f') return hex_inc_table[c - 'a' + 10];
    if (c >= 'A' && c <= 'F') return hex_inc_table[c - 'A' + 10];
    return c;
}

__device__ __forceinline__ char hex_decrement(char c) {
    if (c >= '0' && c <= '9') return hex_dec_table[c - '0'];
    if (c >= 'a' && c <= 'f') return hex_dec_table[c - 'a' + 10];
    if (c >= 'A' && c <= 'F') return hex_dec_table[c - 'A' + 10];
    return c;
}

__device__ __forceinline__ void hex_vertical_rotate_up(char* hex_str) {
    int actual_length = 0;
    #pragma unroll 8
    for (int i = 0; i < HEX_LENGTH; i++) {
        if (hex_str[i] == '\0') {
            actual_length = i;
            break;
        }
    }
    if (actual_length == 0) actual_length = HEX_LENGTH;
    
    if (actual_length <= 1) return;
    
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (hex_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    // Rotate all characters after the first '1' vertically up
    #pragma unroll 8
    for (int i = first_one + 1; i < actual_length; i++) {
        hex_str[i] = hex_increment(hex_str[i]);
    }
}

__device__ __forceinline__ void hex_vertical_rotate_down(char* hex_str) {
    int actual_length = 0;
    #pragma unroll 8
    for (int i = 0; i < HEX_LENGTH; i++) {
        if (hex_str[i] == '\0') {
            actual_length = i;
            break;
        }
    }
    if (actual_length == 0) actual_length = HEX_LENGTH;
    
    if (actual_length <= 1) return;
    
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (hex_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    #pragma unroll 8
    for (int i = first_one + 1; i < actual_length; i++) {
        hex_str[i] = hex_decrement(hex_str[i]);
    }
}

__device__ void leftPad64(char* output, const char* suffix) {
    int suffix_len = 0;
    // Get length of suffix
    while (suffix[suffix_len] != '\0' && suffix_len < 64) {
        ++suffix_len;
    }

    int pad_len = 64 - suffix_len;

    // Fill left padding with '0' using memset
    #pragma unroll 8
    for (int i = 0; i < pad_len; ++i) {
        output[i] = '0';
    }

    // Copy suffix to the right
    #pragma unroll 8
    for (int i = 0; i < suffix_len; ++i) {
        output[pad_len + i] = suffix[i];
    }

    output[64] = '\0';
}

__device__ __forceinline__ int str_len(const char* str) {
    int len = 0;
    while (str[len] != '\0') {
        ++len;
    }
    return len;
}

__device__ void reverseAfterFirst1(char* hex) {
    // Find first '1'
    char* first1 = hex;
    while (*first1 && *first1 != '1') first1++;
    
    if (*first1 == '\0' || *(first1 + 1) == '\0') return;
    
    // Find end
    char* end = first1 + 1;
    while (*end) end++;
    end--;
    
    // Reverse after '1'
    char* start = first1 + 1;
    while (start < end) {
        char temp = *start;
        *start = *end;
        *end = temp;
        start++;
        end--;
    }
}

__device__ void invertHexAfterFirst1(char* hex) {
    bool foundFirst1 = false;
    
    for (int i = 0; hex[i] != '\0'; i++) {
        if (!foundFirst1 && hex[i] == '1') {
            foundFirst1 = true;
            continue;
        }
        
        if (foundFirst1) {
            char c = hex[i];
            int val = hex_char_to_byte(c);
            
            // Invert all 4 bits of this hex digit
            val = (~val) & 0xF;
            
            // Convert back to hex char
            hex[i] = (val < 10) ? ('0' + val) : ('a' + (val - 10));
        }
    }
}

__device__ __forceinline__ int d_strlen(const char* str) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

__device__ void incrementBigInt(BigInt* num) {
    // Start from the least significant word (data[0])
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        num->data[i]++;
        
        // If no overflow, we're done
        if (num->data[i] != 0) {
            break;
        }
        // If overflow (wrapped to 0), continue to next word
    }
}

__device__ void clearLowest8Bits(BigInt* num) {
    // Clear the lowest 8 bits of the least significant word
    num->data[0] &= 0xFFFFFF00;
}

__device__ __forceinline__ uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// Convert BigInt to binary string
__device__ void bigint_to_binary(const BigInt* bigint, char* binary_str) {
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 31; j >= 0; j--) {
            uint8_t bit = (bigint->data[i] >> j) & 1;
            if (bit != 0 || !leading_zero || (i == 0 && j == 0)) {
                binary_str[idx++] = bit ? '1' : '0';
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        binary_str[idx++] = '0';
    }
    
    binary_str[idx] = '\0';
}

// Convert binary string to BigInt
__device__ void binary_to_bigint(const char* binary_str, BigInt* bigint) {
    // Initialize all data to 0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (binary_str[len] != '\0' && len < 256) len++; // Max 256 bits
    
    // Process binary string from right to left
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        if (binary_str[i] == '1') {
            bigint->data[word_idx] |= (1U << bit_offset);
        }
        
        bit_offset++;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

// Binary rotate left by one (after first '1')
__device__ __forceinline__ void binary_rotate_left_by_one(char* binary_str) {
    int actual_length = 0;
    while (binary_str[actual_length] != '\0' && actual_length < 256) {
        actual_length++;
    }
    
    if (actual_length <= 1) return;
    
    // Find first '1'
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (binary_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    int rotation_start = first_one + 1;
    int rotation_length = actual_length - rotation_start;
    
    if (rotation_length <= 1) return;
    
    char first_char = binary_str[rotation_start];
    
    for (int i = 0; i < rotation_length - 1; i++) {
        binary_str[rotation_start + i] = binary_str[rotation_start + i + 1];
    }
    
    binary_str[rotation_start + rotation_length - 1] = first_char;
}

// Binary vertical rotate up (increment each 4-bit nibble after first '1')
__device__ __forceinline__ void binary_vertical_rotate_up(char* binary_str) {
    int actual_length = 0;
    while (binary_str[actual_length] != '\0' && actual_length < 256) {
        actual_length++;
    }
    
    if (actual_length <= 1) return;
    
    // Find first '1'
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (binary_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    // Pad length to multiple of 4 for nibble processing
    int start_pos = first_one + 1;
    
    // Process each 4-bit nibble after the first '1'
    for (int nibble_start = start_pos; nibble_start < actual_length; nibble_start += 4) {
        // Extract current nibble (up to 4 bits)
        uint8_t nibble_val = 0;
        int nibble_size = 0;
        
        // Read nibble value
        for (int bit = 0; bit < 4 && (nibble_start + bit) < actual_length; bit++) {
            if (binary_str[nibble_start + bit] == '1') {
                nibble_val |= (1 << (3 - bit));
            }
            nibble_size++;
        }
        
        // Increment nibble (with wrap-around: F -> 0)
        nibble_val = (nibble_val + 1) & 0xF;
        
        // Write back the incremented nibble
        for (int bit = 0; bit < nibble_size; bit++) {
            binary_str[nibble_start + bit] = ((nibble_val >> (3 - bit)) & 1) ? '1' : '0';
        }
    }
}


// Reverse binary string after first '1'
__device__ void reverseBinaryAfterFirst1(char* binary_str) {
    // Find first '1'
    char* first1 = binary_str;
    while (*first1 && *first1 != '1') first1++;
    
    if (*first1 == '\0' || *(first1 + 1) == '\0') return;
    
    // Find end
    char* end = first1 + 1;
    while (*end) end++;
    end--;
    
    // Reverse after '1'
    char* start = first1 + 1;
    while (start < end) {
        char temp = *start;
        *start = *end;
        *end = temp;
        start++;
        end--;
    }
}

// Invert binary string after first '1'
__device__ void invertBinaryAfterFirst1(char* binary_str) {
    bool foundFirst1 = false;
    
    for (int i = 0; binary_str[i] != '\0'; i++) {
        if (!foundFirst1 && binary_str[i] == '1') {
            foundFirst1 = true;
            continue;
        }
        
        if (foundFirst1) {
            // Invert bit
            binary_str[i] = (binary_str[i] == '0') ? '1' : '0';
        }
    }
}

// Convert binary string to hex string
__device__ void binary_to_hex(const char* binary_str, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int binary_len = 0;
    
    // Get binary string length
    while (binary_str[binary_len] != '\0') binary_len++;
    
    // Pad binary string to multiple of 4 bits
    int padded_len = ((binary_len + 3) / 4) * 4;
    
    int hex_idx = 0;
    bool leading_zero = true;
    
    // Process 4 bits at a time
    for (int i = 0; i < padded_len; i += 4) {
        uint8_t nibble = 0;
        
        // Convert 4 binary digits to hex nibble
        for (int j = 0; j < 4; j++) {
            int bit_pos = i + j;
            int actual_pos = bit_pos - (padded_len - binary_len);
            
            if (actual_pos >= 0 && actual_pos < binary_len && binary_str[actual_pos] == '1') {
                nibble |= (1 << (3 - j));
            }
        }
        
        // Skip leading zeros
        if (nibble != 0 || !leading_zero || i >= padded_len - 4) {
            hex_str[hex_idx++] = hex_chars[nibble];
            leading_zero = false;
        }
    }
    
    // Handle case where result is 0
    if (hex_idx == 0) {
        hex_str[hex_idx++] = '0';
    }
    
    hex_str[hex_idx] = '\0';
}

// Optimization 4: Direct binary to BigInt conversion (implement this helper)
__device__ void binary_to_bigint_direct(const char* binary, BigInt* result) {
    // Initialize result to zero
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result->data[i] = 0;
    }
    
    // Process binary string directly without hex intermediate
    int len = d_strlen(binary);
    for (int i = 0; i < len && i < 256; i++) {
        if (binary[len - 1 - i] == '1') {
            int word_idx = i >> 5; // i / 32
            int bit_idx = i & 31;  // i % 32
            if (word_idx < BIGINT_WORDS) {
                result->data[word_idx] |= (1U << bit_idx);
            }
        }
    }
}

// Optimization 5: Faster hash160 comparison (implement this)
__device__ __forceinline__ bool compare_hash160_fast(const uint8_t* hash1, const uint8_t* hash2) {
    // Use 64-bit comparisons instead of byte-by-byte
    const uint64_t* h1 = (const uint64_t*)hash1;
    const uint64_t* h2 = (const uint64_t*)hash2;
    
    return (h1[0] == h2[0]) && (h1[1] == h2[1]) && 
           (*(uint32_t*)(hash1 + 16) == *(uint32_t*)(hash2 + 16));
}


__device__ volatile int g_found = 0;
__device__ char g_found_hex[65] = {0};        // Original hex
__device__ char g_found_hash160[41] = {0};    // Hash160 result

// Add pair swap function
__device__ void binary_pair_swap(char* binary) {
    int len = d_strlen(binary);
    
    // Find the position of the first '1'
    int first_one_pos = 0;
    for (int i = 0; i < len; i++) {
        if (binary[i] == '1') {
            first_one_pos = i;
            break;
        }
    }
    
    // Swap pairs starting after the first '1'
    // Start from first_one_pos + 1 to preserve the leading bit
    for (int i = first_one_pos + 1; i < len - 1; i += 2) {
        char temp = binary[i];
        binary[i] = binary[i + 1];
        binary[i + 1] = temp;
    }
}

// Add interleave function
__device__ void binary_interleave(char* binary) {
    int len = d_strlen(binary);
    
    // Find the position of the first '1'
    int first_one_pos = 0;
    for (int i = 0; i < len; i++) {
        if (binary[i] == '1') {
            first_one_pos = i;
            break;
        }
    }
    
    // Work with the part after the first '1'
    int working_len = len - first_one_pos;
    if (working_len <= 2) return; // Not enough bits to interleave
    
    // Temporary buffer for the result
    char temp[257];
    
    // Copy the leading part (up to and including first '1')
    for (int i = 0; i <= first_one_pos; i++) {
        temp[i] = binary[i];
    }
    
    // Calculate mid point for the working section
    int mid = (working_len - 1) / 2; // -1 because we don't count the first '1'
    
    // Interleave the bits: take alternating bits from first and second half
    int write_pos = first_one_pos + 1;
    for (int i = 0; i < mid; i++) {
        temp[write_pos++] = binary[first_one_pos + 1 + i];              // From first half
        temp[write_pos++] = binary[first_one_pos + 1 + mid + i];        // From second half
    }
    
    // If odd number of bits, append the last bit
    if ((working_len - 1) % 2 == 1) {
        temp[write_pos++] = binary[len - 1];
    }
    
    // Copy back to original
    for (int i = 0; i < len; i++) {
        binary[i] = temp[i];
    }
}

// Add bit shuffle function - shuffles bits in groups of 8
__device__ void binary_bit_shuffle(char* binary) {
    int len = d_strlen(binary);
    
    // Find the position of the first '1'
    int first_one_pos = 0;
    for (int i = 0; i < len; i++) {
        if (binary[i] == '1') {
            first_one_pos = i;
            break;
        }
    }
    
    // Work with the part after the first '1'
    int start_pos = first_one_pos + 1;
    int working_len = len - start_pos;
    if (working_len < 8) return; // Not enough bits to shuffle
    
    // Shuffle pattern: [0,1,2,3,4,5,6,7] -> [3,6,1,4,7,2,5,0]
    // This pattern ensures good bit mixing
    const int shuffle_pattern[8] = {3, 6, 1, 4, 7, 2, 5, 0};
    
    // Process in groups of 8 bits
    for (int group_start = start_pos; group_start + 7 < len; group_start += 8) {
        char temp[8];
        
        // Copy current group
        for (int i = 0; i < 8; i++) {
            temp[i] = binary[group_start + i];
        }
        
        // Apply shuffle pattern
        for (int i = 0; i < 8; i++) {
            binary[group_start + i] = temp[shuffle_pattern[i]];
        }
    }
    
    // Handle remaining bits (less than 8) with a simple reverse
    int remaining_start = (working_len / 8) * 8 + start_pos;
    int remaining_count = len - remaining_start;
    for (int i = 0; i < remaining_count / 2; i++) {
        char temp = binary[remaining_start + i];
        binary[remaining_start + i] = binary[len - 1 - i];
        binary[len - 1 - i] = temp;
    }
}

// Fast RNG function
__device__ uint64_t fast_rng(uint64_t* state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state;
}

// Optimized hash160 comparison using vectorized loads
__device__ bool compare_hash160_fast_vectorized(const uint8_t* hash1, const uint8_t* target) {
    const uint4 h1 = *((uint4*)hash1);
    const uint4 h2 = *((uint4*)target);
    const uint32_t h1_tail = *((uint32_t*)(hash1 + 16));
    const uint32_t h2_tail = *((uint32_t*)(target + 16));
    
    return (h1.x == h2.x) && (h1.y == h2.y) && 
           (h1.z == h2.z) && (h1.w == h2.w) && (h1_tail == h2_tail);
}
__global__ void start_optimized(const char* minRangePure, const char* maxRangePure, const char* target) {
    // Shared memory for frequently accessed data
    __shared__ uint8_t shared_target[20];
    __shared__ BigInt shared_min, shared_max;
    
    // Initialize shared memory once per block
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
            printf("Starting search...\n");
        }
        
        // Parse min/max once per block
        char minRange[65], maxRange[65];
        leftPad64(minRange, minRangePure);
        leftPad64(maxRange, maxRangePure);
        hex_to_bigint(minRange, &shared_min);
        hex_to_bigint(maxRange, &shared_max);
        
        // Parse target once per block
        hex_string_to_bytes(target, shared_target, 20);
    }
    __syncthreads();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Fast RNG - use better mixing
    uint64_t rng_state = mix(clock64() ^ ((uint64_t)tid << 32) ^ ((uint64_t)blockIdx.x << 48));
    
    // Process multiple keys per thread for better efficiency
    const int KEYS_PER_THREAD = 16;
    
    // Allocate working memory once
    BigInt random_value, priv;
    ECPointJac result_jac;
    ECPoint public_key;
    uint8_t pubkey[33];
    uint8_t hash160_out[20];
    
    // Remove all debug output from hot loop
    uint32_t iterations = 0;
    
    while (!g_found) {
        // Process batch of keys
        for (int k = 0; k < KEYS_PER_THREAD; k++) {
            // Generate random value
            generate_random_bigint_range_optimized(&rng_state, &shared_min, &shared_max, &random_value);
            
            // Ensure within curve order (branchless)
            int needs_reduction = (compare_bigint(&random_value, &const_n) >= 0);
            if (needs_reduction) {
                ptx_u256Sub(&priv, &random_value, &const_n);
            } else {
                copy_bigint(&priv, &random_value);
            }
            
            // Scalar multiplication
            scalar_multiply_optimized(&result_jac, &const_G_jacobian, &priv);
            jacobian_to_affine_fast(&public_key, &result_jac);
            
            // Compress public key
            coords_to_compressed_pubkey(public_key.x, public_key.y, pubkey);
            
            // Hash
            hash160(pubkey, 33, hash160_out);
            
            // Fast comparison using 32-bit operations
            uint32_t* hash32 = (uint32_t*)hash160_out;
            uint32_t* target32 = (uint32_t*)shared_target;
            
            int match = ((hash32[0] == target32[0]) & 
                        (hash32[1] == target32[1]) & 
                        (hash32[2] == target32[2]) & 
                        (hash32[3] == target32[3]) & 
                        (hash32[4] == target32[4]));
            
            if (match) {
                // Only do expensive operations when we find a match
                if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                    // Convert to strings only for the found key
                    char binary[257];
                    bigint_to_binary(&random_value, binary);
                    char temp_hex[65], hash160_str[41];
                    binary_to_hex(binary, temp_hex);
                    hash160_to_hex(hash160_out, hash160_str);
                    
                    memcpy(g_found_hex, temp_hex, 65);
                    memcpy(g_found_hash160, hash160_str, 41);
                    
                    printf("\n*** FOUND! ***\n");
                    printf("Private Key: %s\n", temp_hex);
                    printf("Hash160: %s\n", hash160_str);
                }
                return;
            }
        }
        
        iterations += KEYS_PER_THREAD;
        
        if (tid == 0 && iterations % 1024 == 0) {
            // Just report the rate, no string conversions
            printf("Thread 0: %u iterations\n", iterations);
        }
    }
}
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " (required <min> <max> <target>) (optional <blocks> <threads>)" << std::endl;
        return 1;
    }
    
    try {
		
        int device_id = (argc >= 7) ? std::stoi(argv[6]) : 0;

        // Check if device exists
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_id < 0 || device_id >= device_count) {
            std::cerr << "Invalid device ID: " << device_id
                      << ". Available devices: 0 to " << (device_count - 1) << std::endl;
            return 1;
        }

        // Set device
        cudaSetDevice(device_id);

        std::cout << "Using CUDA device " << device_id << std::endl;
		initialize_secp256k1_gpu();
        
        // Allocate device memory for 3 strings
        char *d_param1, *d_param2, *d_param3;
        
        // Get string lengths
        size_t len1 = strlen(argv[1]) + 1;
        size_t len2 = strlen(argv[2]) + 1;
        size_t len3 = strlen(argv[3]) + 1;
        
        // Allocate and copy in one operation each
        cudaMalloc(&d_param1, len1);
        cudaMemcpy(d_param1, argv[1], len1, cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_param2, len2);
        cudaMemcpy(d_param2, argv[2], len2, cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_param3, len3);
        cudaMemcpy(d_param3, argv[3], len3, cudaMemcpyHostToDevice);
        
        // Parse grid configuration
        int blocks = (argc >= 5) ? std::stoi(argv[4]) : 32;
        int threads = (argc >= 6) ? std::stoi(argv[5]) : 32;
        
        printf("Launching with %d blocks and %d threads\nTotal parallel threads: %d\n\n", 
               blocks, threads, blocks * threads);
        
        // Launch kernel
        start_optimized<<<blocks, threads>>>(d_param1, d_param2, d_param3);
        
        // Wait for completion
        cudaDeviceSynchronize();
        
        // Check if solution was found
        int found_flag;
        cudaMemcpyFromSymbol(&found_flag, g_found, sizeof(int));
        
        if (found_flag) {
            char found_hex[65];
            char found_hash160[41];
            
            // Copy results from device
            cudaMemcpyFromSymbol(found_hex, g_found_hex, 65);
            cudaMemcpyFromSymbol(found_hash160, g_found_hash160, 41);
            
            // Save to file with timestamp
            std::ofstream outfile("result.txt", std::ios::app);
            if (outfile.is_open()) {
                std::time_t now = std::time(nullptr);
                char timestamp[100];
                std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", 
                             std::localtime(&now));
                
                outfile << "[" << timestamp << "] Found: " << found_hex 
                       << " -> " << found_hash160 << std::endl;
                outfile.close();
                std::cout << "Result appended to result.txt" << std::endl;
            } else {
                std::cerr << "Unable to open file for writing" << std::endl;
            }
        }
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_param1);
            cudaFree(d_param2);
            cudaFree(d_param3);
            return 1;
        }
        
        // Clean up
        cudaFree(d_param1);
        cudaFree(d_param2);
        cudaFree(d_param3);
        
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        cudaDeviceReset();
        return 1;
    }
    
    return 0;
}