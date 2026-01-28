# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Sliding Window Attention (SWA)
#####################################
# SWA는 고정된 윈도우 크기 내의 토큰만 참조하여 메모리와 연산량을 줄이는 어텐션 기법입니다.
# Mistral, Gemma 2 등에서 사용됩니다.
#
# ============================================
# 일반 Attention vs Sliding Window Attention
# ============================================
#
# 일반 Causal Attention (모든 이전 토큰 참조):
# 토큰:    T0  T1  T2  T3  T4  T5  T6  T7
# T7 참조: ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓  (모두 참조)
#
# Sliding Window Attention (윈도우=4):
# 토큰:    T0  T1  T2  T3  T4  T5  T6  T7
# T7 참조: ✗   ✗   ✗   ✗   ✓   ✓   ✓   ✓  (최근 4개만 참조)
#          └─ 윈도우 밖 ─┘  └─ 윈도우 안 ─┘
#
# ============================================
# 핵심 코드
# ============================================
#
# 1. KV 캐시 트리밍 (윈도우 크기 초과 시 오래된 것 삭제):
#    if self.sliding_window_size is not None:
#        if self.cache_k.size(1) > self.sliding_window_size:
#            self.cache_k = self.cache_k[:, -self.sliding_window_size:, :, :]
#            self.cache_v = self.cache_v[:, -self.sliding_window_size:, :, :]
#
#    캐시 (윈도우=4):
#    Before: [K0, K1, K2, K3, K4, K5]  (6개)
#    After:  [K2, K3, K4, K5]          (최근 4개만 유지)
#
# 2. 슬라이딩 윈도우 마스크:
#    W = self.sliding_window_size  # 윈도우 크기
#    diff = q_positions - k_positions
#    mask_bool = (diff < 0) | (diff >= W)  # 윈도우 밖이면 마스킹
#
#    | 조건       | 의미                    |
#    |------------|-------------------------|
#    | diff < 0   | 미래 토큰 (causal mask) |
#    | diff >= W  | 윈도우 밖의 과거 토큰   |
#
# ============================================
# K:1 스케줄링
# ============================================
# 모든 레이어에 SWA를 적용하지 않고, K개 SWA + 1개 일반 어텐션 패턴을 반복합니다.
#
# sliding_window_stride = 2 인 경우 (2:1 스케줄):
# Layer 0: SWA
# Layer 1: SWA
# Layer 2: 일반 (전체 참조)
# Layer 3: SWA
# Layer 4: SWA
# Layer 5: 일반 (전체 참조)
# ...
#
# 이유: 일부 레이어에서 전체 컨텍스트를 참조하여 정보 손실 방지.
#
# ============================================
# 메모리 절약 효과
# ============================================
# | 항목           | 일반 Attention  | SWA (윈도우=1024) |
# |----------------|-----------------|-------------------|
# | KV 캐시 크기   | O(시퀀스 길이)  | O(윈도우 크기) = 고정 |
# | 어텐션 연산    | O(n²)           | O(n × W)          |
#
# 시퀀스가 아무리 길어져도 KV 캐시는 윈도우 크기로 고정됩니다.
#
# ============================================
# 시각화
# ============================================
# 일반 Attention (모든 이전 토큰):
#         K0 K1 K2 K3 K4 K5 K6 K7
#    Q7 [  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓ ]  ← 8개 참조
#
# Sliding Window (W=4):
#         K0 K1 K2 K3 K4 K5 K6 K7
#    Q7 [  ✗  ✗  ✗  ✗  ✓  ✓  ✓  ✓ ]  ← 4개만 참조
#          └─ 마스킹 ─┘
#
# Sliding Window + K:1 스케줄 (일부 레이어는 전체 참조):
# Layer 0 (SWA):  Q7 → [K4, K5, K6, K7]  (윈도우 내)
# Layer 1 (SWA):  Q7 → [K4, K5, K6, K7]  (윈도우 내)
# Layer 2 (일반): Q7 → [K0~K7 전체]       (전체 컨텍스트)
#
# ============================================
# 사용처
# ============================================
# | 모델        | 윈도우 크기 | 스케줄              |
# |-------------|-------------|---------------------|
# | Mistral 7B  | 4096        | 전체 SWA            |
# | Gemma 2     | 4096        | 1:1 (SWA-일반 교대) |
# | Phi-3       | 2048        | K:1 스케줄          |
#
# ============================================
# Trade-off
# ============================================
# | 장점                      | 단점                      |
# |---------------------------|---------------------------|
# | KV 캐시 고정 크기         | 먼 토큰 직접 참조 불가    |
# | 긴 시퀀스도 메모리 일정   | 일부 정보 손실 가능       |
# | 추론 속도 향상            | K:1 스케줄 튜닝 필요      |
#
# SWA는 무한히 긴 시퀀스도 고정 메모리로 처리할 수 있어,
# 스트리밍/실시간 생성에 효과적입니다.


class MultiHeadAttentionWithSWA(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, sliding_window_size=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.sliding_window_size = sliding_window_size

        ####################################################
        # KV cache-related code
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
        ####################################################

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV 캐시 + 슬라이딩 윈도우 트리밍
        ####################################################
        if use_cache:
            # 이전 캐시 길이 저장
            old_len = 0 if self.cache_k is None else self.cache_k.size(1)

            # 캐시에 새 K, V 추가
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)

            ####################################################
            # 슬라이딩 윈도우 트리밍 (핵심!)
            # 캐시가 윈도우 크기를 초과하면 오래된 토큰 삭제
            # 예: 윈도우=4, 캐시=[K0,K1,K2,K3,K4,K5] → [K2,K3,K4,K5]
            ####################################################
            if self.sliding_window_size is not None:
                if self.cache_k.size(1) > self.sliding_window_size:
                    self.cache_k = self.cache_k[:, -self.sliding_window_size:, :, :]
                    self.cache_v = self.cache_v[:, -self.sliding_window_size:, :, :]

            # 절대 위치 계산 (마스크 생성용)
            # 트리밍으로 삭제된 토큰 수를 고려해야 함
            total_len = old_len + num_tokens        # 원래 전체 길이
            k_len_now = self.cache_k.size(1)        # 트리밍 후 캐시 길이
            dropped = max(0, total_len - k_len_now) # 삭제된 토큰 수
            k_start_pos_abs = (self.ptr_current_pos - old_len) + dropped  # K의 시작 절대 위치
            q_start_pos_abs = self.ptr_current_pos  # Q의 시작 절대 위치
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        ####################################################
        # Causal + Sliding Window 마스크 생성
        ####################################################
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device

        # Q와 K의 절대 위치 결정
        if use_cache:
            q_start = q_start_pos_abs
            k_start = k_start_pos_abs
        else:
            q_start = 0
            k_start = 0

        # 절대 위치 배열 생성
        # 예: q_start=7, num_tokens_Q=1 → q_positions=[7]
        # 예: k_start=4, num_tokens_K=4 → k_positions=[4,5,6,7]
        q_positions = torch.arange(q_start, q_start + num_tokens_Q, device=device, dtype=torch.long)
        k_positions = torch.arange(k_start, k_start + num_tokens_K, device=device, dtype=torch.long)

        # 슬라이딩 윈도우 크기 결정
        # None이면 전체 참조 (일반 어텐션)
        W = num_tokens_K + 1 if self.sliding_window_size is None else int(self.sliding_window_size)

        # 마스크 생성: Q위치 - K위치 = 거리
        # diff[i,j] = q_positions[i] - k_positions[j]
        diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0)

        # 마스킹 조건:
        # 1) diff < 0: 미래 토큰 (causal mask)
        # 2) diff >= W: 윈도우 밖의 과거 토큰 (sliding window mask)
        #
        # 예: Q위치=7, K위치=[4,5,6,7], W=4
        # diff = [7-4, 7-5, 7-6, 7-7] = [3, 2, 1, 0]
        # mask = [False, False, False, False] (모두 윈도우 내)
        #
        # 예: Q위치=7, K위치=[0,1,2,3], W=4
        # diff = [7-0, 7-1, 7-2, 7-3] = [7, 6, 5, 4]
        # mask = [True, True, True, True] (모두 diff >= 4)
        mask_bool = (diff < 0) | (diff >= W)

        # 위치 포인터 업데이트
        if use_cache:
            self.ptr_current_pos += num_tokens_Q
        else:
            self.ptr_current_pos = 0

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionWithSWA(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            sliding_window_size=cfg["sliding_window_size"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        ####################################################
        # K:1 스케줄링으로 Transformer 블록 생성
        ####################################################
        # K:1 스케줄 = K개의 SWA 레이어 + 1개의 일반 어텐션 레이어 반복
        #
        # sliding_window_stride = 2 인 경우 (2:1 스케줄):
        # Layer 0: SWA (i=0, 0 % 3 = 0 < 2 → SWA)
        # Layer 1: SWA (i=1, 1 % 3 = 1 < 2 → SWA)
        # Layer 2: 일반 (i=2, 2 % 3 = 2 >= 2 → 일반)
        # Layer 3: SWA (i=3, 3 % 3 = 0 < 2 → SWA)
        # Layer 4: SWA (i=4, 4 % 3 = 1 < 2 → SWA)
        # Layer 5: 일반 (i=5, 5 % 3 = 2 >= 2 → 일반)
        # ...
        #
        # 이유: 일부 레이어에서 전체 컨텍스트를 참조하여 정보 손실 방지
        ####################################################
        blocks = []
        window_stride = cfg["sliding_window_stride"]
        window_size = cfg["sliding_window_size"] if "sliding_window_size" in cfg else None

        for i in range(cfg["n_layers"]):
            blk = TransformerBlock(cfg)

            K = int(window_stride)
            if K <= 0:
                # K=0: 모든 레이어가 일반 어텐션
                # K<0: 모든 레이어가 SWA
                use_swa = False if K == 0 else True
            else:
                # K:1 스케줄 적용
                group = K + 1              # 그룹 크기 (예: 2+1=3)
                use_swa = (i % group) < K  # 그룹 내 처음 K개는 SWA

            # SWA 레이어면 윈도우 크기 설정, 아니면 None (전체 참조)
            blk.att.sliding_window_size = window_size if use_swa else None
            blocks.append(blk)

        self.trf_blocks = nn.ModuleList(blocks)
        self.current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser(description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--sliding_window_size", type=int, default=1024, help="Window size for sliding window attention.")
    parser.add_argument("--sliding_window_stride", type=int, default=2, help="K:1 frequency sliding window attention is applied. K=5 means 5 sliding window layers follows by a regular layer.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
        "sliding_window_size": args.sliding_window_size,
        "sliding_window_stride": args.sliding_window_stride
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
