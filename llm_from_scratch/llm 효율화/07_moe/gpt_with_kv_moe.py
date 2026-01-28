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

MOE_FF_TIME_MS = []
MOE_FF_MEM_BYTES = []


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
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
        # KV cache-related
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
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
        # causal mask
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q
        else:
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

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
            nn.Linear(cfg["emb_dim"], cfg["hidden_dim"]),
            GELU(),
            nn.Linear(cfg["hidden_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


#####################################
# Mixture of Experts (MoE) FeedForward
#####################################
# MoE는 하나의 큰 FFN 대신 여러 개의 작은 Expert 중 일부만 선택하여 연산합니다.
#
# ============================================
# Dense FFN vs MoE FFN 비교
# ============================================
#
# Dense FFN:
#   입력 → [하나의 큰 FFN] → 출력
#   모든 토큰이 같은 파라미터 사용
#
# MoE FFN:
#   입력 → Gate → [Expert 1, 2, ..., N] 중 Top-K 선택 → 가중합 출력
#   토큰마다 다른 Expert 조합 사용
#
# ============================================
# 구조
# ============================================
#
#         입력 x (768)
#              │
#    ┌─────────┴─────────┐
#    │                   │
#    ▼                   ▼
# [Gate Network]    [Expert 1] [Expert 2] ... [Expert N]
# 768 → N scores       │           │              │
#    │                 └───────────┴──────────────┘
#    ▼                              │
# Top-K 선택 ──────────────────────►│
# (예: K=2)                         │
#    │                              ▼
#    └────► 선택된 Expert만 연산 후 가중합 → 출력
#
# ============================================
# 핵심 코드 설명
# ============================================
#
# 1. Gate Network:
#    self.gate = nn.Linear(emb_dim, num_experts)
#    - 입력을 보고 각 Expert의 점수 계산
#    - scores = gate(x)  # (batch, seq, num_experts)
#
# 2. Top-K 선택:
#    topk_scores, topk_indices = torch.topk(scores, num_experts_per_tok)
#    topk_probs = softmax(topk_scores)  # 선택된 Expert들의 가중치
#
# 3. Expert (SwiGLU 구조):
#    fc1, fc2: emb_dim → hidden_dim (병렬 투영)
#    fc3: hidden_dim → emb_dim
#    hidden = silu(fc1(x)) * fc2(x)  # 게이트 메커니즘
#    output = fc3(hidden)
#
# 4. 가중합:
#    output = Σ (prob_i × Expert_i(x))
#
# ============================================
# 예시 (num_experts=8, num_experts_per_tok=2)
# ============================================
#
# 토큰 "cat":
#   Gate 점수: [0.1, 0.3, 0.05, 0.8, 0.2, 0.15, 0.7, 0.1]
#   Top-2 선택: Expert 3 (0.8), Expert 6 (0.7)
#   softmax([0.8, 0.7]) = [0.52, 0.48]
#   출력 = 0.52 × Expert3(x) + 0.48 × Expert6(x)
#
# ============================================
# 메모리 vs 연산량 Trade-off
# ============================================
#
# | 항목          | Dense FFN | MoE (8 Expert, Top-2) |
# |---------------|-----------|----------------------|
# | 파라미터 수   | 1×        | 8× (Expert 수)       |
# | 실제 연산량   | 100%      | 25% (2/8)            |
# | 모델 용량     | 고정      | 대폭 증가            |
#
# → 파라미터는 8배 늘려도, 연산량은 2배만 증가!
# → 더 큰 모델 용량을 적은 연산 비용으로 달성
#
# ============================================
# 사용처
# ============================================
# - Mixtral 8x7B: 8 Expert, Top-2
# - GPT-4 (추정): MoE 구조 사용
# - Switch Transformer: Top-1 Expert
#
class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.emb_dim = cfg["emb_dim"]

        # Gate: 입력을 보고 어떤 Expert를 사용할지 점수 계산
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False)
        # fc1, fc2, fc3: 각 Expert의 SwiGLU FFN (Expert마다 독립적인 파라미터)
        self.fc1 = nn.ModuleList(
            [
                nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )
        self.fc2 = nn.ModuleList(
            [
                nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )
        self.fc3 = nn.ModuleList(
            [
                nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x):
        # x: (batch, seq_len, emb_dim)

        ####################################################
        # Step 1: Gate 점수 계산 및 Top-K Expert 선택
        ####################################################
        scores = self.gate(x)  # (batch, seq_len, num_experts)
        # 각 토큰에 대해 가장 점수가 높은 K개의 Expert 선택
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        # 선택된 Expert들의 가중치 계산 (합=1)
        topk_probs = torch.softmax(topk_scores, dim=-1)
        # 예: num_experts=8, num_experts_per_tok=2
        # topk_indices = [[3, 6], [1, 4], ...]  # 각 토큰이 선택한 Expert ID
        # topk_probs = [[0.52, 0.48], [0.61, 0.39], ...]  # 가중치

        ####################################################
        # Step 2: 토큰을 평탄화하여 배치 처리 준비
        ####################################################
        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, -1)  # (batch*seq_len, emb_dim)
        out_flat = torch.zeros(batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        # 실제로 선택된 Expert들만 처리 (Sparse Computation)
        unique_experts = torch.unique(topk_indices_flat)

        ####################################################
        # Step 3: 각 Expert별로 해당 토큰들만 모아서 처리
        ####################################################
        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())

            # 이 Expert가 선택된 토큰 위치 찾기
            mask = topk_indices_flat == expert_id
            if not mask.any():
                continue

            # 이 Expert를 사용하는 토큰들의 인덱스
            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            ####################################################
            # Step 4: SwiGLU Expert 연산
            ####################################################
            # 선택된 토큰들만 추출
            expert_input = x_flat.index_select(0, selected_idx)
            # SwiGLU: silu(fc1(x)) * fc2(x) → fc3
            # silu = x * sigmoid(x) (Swish 활성화 함수)
            hidden = torch.nn.functional.silu(self.fc1[expert_id](expert_input)) * self.fc2[
                expert_id
            ](expert_input)
            expert_out = self.fc3[expert_id](hidden)

            ####################################################
            # Step 5: 가중치를 곱해서 결과에 누적
            ####################################################
            # 이 Expert의 가중치 추출
            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(
                topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices
            ).squeeze(-1)

            # 가중치를 곱한 결과를 누적 (여러 Expert 결과의 가중합)
            # 예: 토큰 "cat"의 출력 = 0.52 × Expert3(x) + 0.48 × Expert6(x)
            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        return out_flat.reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = MoEFeedForward(cfg) if cfg["num_experts"] > 0 else FeedForward(cfg)
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
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
        start = time.perf_counter()
        x = self.ff(x)
        if use_cuda:
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated()
            MOE_FF_MEM_BYTES.append(peak_mem - base_mem)
        MOE_FF_TIME_MS.append((time.perf_counter() - start) * 1000.0)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

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
    batch_size, base_len = idx.shape
    total_len = base_len + max_new_tokens
    generated = torch.empty(
        batch_size, total_len, dtype=idx.dtype, device=idx.device
    )
    generated[:, :base_len] = idx
    cur_len = base_len
    use_cuda = torch.cuda.is_available()
    MOE_FF_TIME_MS.clear()
    MOE_FF_MEM_BYTES.clear()

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            prompt_start = max(0, cur_len - ctx_len)
            logits = model(generated[:, prompt_start:cur_len], use_cache=True)

            if use_cuda:
                torch.cuda.synchronize()

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1)
                # b) append it to the running sequence (in-place)
                generated[:, cur_len] = next_idx
                cur_len += 1
                # c) feed model only the new token
                logits = model(generated[:, cur_len - 1 : cur_len], use_cache=True)

                if use_cuda:
                    torch.cuda.synchronize()
        else:
            if use_cuda:
                torch.cuda.synchronize()

            for _ in range(max_new_tokens):
                start_ctx = max(0, cur_len - ctx_len)
                logits = model(generated[:, start_ctx:cur_len], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1)
                generated[:, cur_len] = next_idx
                cur_len += 1

                if use_cuda:
                    torch.cuda.synchronize()

    if MOE_FF_TIME_MS:
        avg_ffn_time = sum(MOE_FF_TIME_MS) / len(MOE_FF_TIME_MS)
        print(f"Avg MoE FF time/call: {avg_ffn_time:.3f} ms")
    if MOE_FF_MEM_BYTES:
        avg_ffn_mem = sum(MOE_FF_MEM_BYTES) / len(MOE_FF_MEM_BYTES)
        max_ffn_mem = max(MOE_FF_MEM_BYTES)

        def to_mb(bytes_val):
            return bytes_val / (1024 ** 2)
        print(f"Avg MoE FF mem delta/call: {to_mb(avg_ffn_mem):.2f} MB (max {to_mb(max_ffn_mem):.2f} MB)")

    return generated[:, :cur_len]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=768*4, help="Intermediate FFN or MoE size.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV caching during generation.",
    )

    parser.add_argument(
        "--num_experts",
        type=int,
        default=0,
        help="Number of experts. If 0, use dense FFN. If >0, use MoE.",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=2,
        help="Top-k experts per token when using MoE (ignored if num_experts=0).",
    )

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,            # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        "emb_dim": args.emb_dim,        # Embedding dimension
        "hidden_dim": args.hidden_dim,  # Intermediate size
        "n_heads": args.n_heads,        # Number of attention heads
        "n_layers": args.n_layers,      # Number of layers
        "drop_rate": 0.0,               # Dropout rate
        "qkv_bias": False,              # Query-Key-Value bias
        "num_experts": args.num_experts,
        "num_experts_per_tok": args.num_experts_per_tok if args.num_experts > 0 else 0,
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
        use_cache=not args.no_kv_cache,
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
