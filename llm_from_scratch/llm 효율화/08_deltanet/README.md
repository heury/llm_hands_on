# Gated DeltaNet for Linear Attention

Recently, [Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) and [Kimi Linear](https://arxiv.org/abs/2510.26692) proposed hybrid transformers that implement alternatives to the attention mechanism that scale linearly instead of quadratically with respect to the context length.

Both Qwen3-Next and Kimi Linear use a 3:1 ratio, meaning for every three transformer blocks employing the linear Gated DeltaNet variant, there’s one block that uses full attention, as shown in the figure below.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next versus Kimi Linear">



&nbsp;

## Introduction and Overview

Gated DeltaNet is a linear attention variant with inspiration from recurrent neural networks, including a gating mechanism from the [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) paper. In a sense, Gated DeltaNet is a DeltaNet with Mamba-style gating, and DeltaNet is a linear attention mechanism.

Kimi Linear modifies the linear attention mechanism of Qwen3-Next by the Kimi Delta Attention (KDA) mechanism, which is essentially a refinement of Gated DeltaNet. Whereas Qwen3-Next applies a scalar gate (one value per attention head) to control the memory decay rate, Kimi Linear replaces it with a channel-wise gating for each feature dimension. According to the authors, this gives more control over the memory, and this, in turn, improves long-context reasoning.

In addition, for the full attention layers, Kimi Linear replaces Qwen3-Next’s gated attention layers (which are essentially standard multi-head attention layers with output gating) with Multi-Head Latent Attention (MLA). This is the same MLA mechanism we discussed earlier in the DeepSeek V3/R1 section, but with an additional gate. (To recap, MLA compresses the key/value space to reduce the KV cache size.)

The MLA in Kimi Linear does not use the gate, which was intentional so that the authors could compare the architecture more directly to standard MLA, however, they [stated](https://x.com/yzhang_cs/status/1984631714464088563) that they plan to add it in the future.

Since we already implemented MLA in [../05_mla](../05_mla), this bonus material focuses on the Gated DeltaNet aspect.


&nbsp;
## Gated Attention

Before we get to the Gated DeltaNet itself, let's briefly talk about the gate. As you can see in the upper part of the Qwen3-Next architecture in the previous figure, Qwen3-Next uses "gated attention". This is essentially regular full attention with an additional sigmoid gate.

This gating is a simple modification that I added to the `MultiHeadAttention`  code from chapter 3 below for illustration purposes:

---

### GatedMultiHeadAttention 코드 설명 (한글)

**GatedMultiHeadAttention**은 일반 MHA(Multi-Head Attention)에 출력 게이트를 추가한 간단한 변형입니다.

#### 일반 MHA vs Gated MHA 비교

```
일반 MHA:                           Gated MHA:

   Q  K  V                             Q  K  V  Gate
   │  │  │                             │  │  │   │
   └──┼──┘                             └──┼──┘   │
      │                                   │      │
   Attention                           Attention │
      │                                   │      │
      ▼                                   ▼      ▼
   Output                              Output × sigmoid(Gate)
                                          │
                                          ▼
                                       Output
```

#### 핵심 아이디어

| 구분 | 일반 MHA | Gated MHA |
|------|----------|-----------|
| 어텐션 계산 | softmax(QK^T / √d) @ V | **동일** |
| 게이트 | 없음 | `sigmoid(W_gate @ x)` |
| 출력 | context | `context × sigmoid(gate)` |
| 복잡도 | O(n²) | **O(n²) (동일)** |

#### 게이트의 역할

- **sigmoid(gate)**: 0~1 사이 값으로 출력 스케일링
- **장점**: Attention Sink, Massive Activation 문제 해결
- **목적**: 학습 안정성 향상 (수치적 안정성)

#### Qwen3-Next에서의 사용

```
3:1 하이브리드 아키텍처:
┌─────────────────────────────────────┐
│ Layer 0-2: Gated DeltaNet (선형)    │
│ Layer 3:   Gated MHA (전체 컨텍스트)│  ← 여기서 사용!
│ Layer 4-6: Gated DeltaNet           │
│ Layer 7:   Gated MHA                │
│ ...                                 │
└─────────────────────────────────────┘
```

---

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    """
    Gated Multi-Head Attention: 일반 MHA + 출력 게이트

    일반 MHA와 동일한 O(n²) 복잡도이지만,
    출력 게이트로 학습 안정성을 향상시킴.
    Qwen3-Next의 3:1 하이브리드 구조에서 Full Attention 레이어로 사용.
    """
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 예: 2048 / 16 = 128

        # Q, K, V 프로젝션 (일반 MHA와 동일)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: 출력 게이트 프로젝션 추가
        # 입력 x를 받아 게이트 값을 생성
        # sigmoid 적용 후 [0, 1] 범위로 출력 스케일링
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # 최종 출력 프로젝션
        self.dropout = nn.Dropout(dropout)

        # Causal mask: 미래 토큰 참조 방지 (autoregressive)
        # 상삼각 행렬: 대각선 위는 1, 나머지는 0
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Step 1: Q, K, V 프로젝션 (일반 MHA와 동일)
        queries = self.W_query(x)
        ####################################################
        ### NEW: 게이트 값 계산
        # gate는 나중에 sigmoid 적용 후 출력을 스케일링
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        # Step 2: Multi-Head 형태로 reshape
        # (batch, tokens, d_out) → (batch, tokens, heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Step 3: Head 차원을 앞으로 이동
        # (batch, tokens, heads, head_dim) → (batch, heads, tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Step 4: Attention Score 계산 (O(n²) 연산!)
        # (batch, heads, n, d) @ (batch, heads, d, n) → (batch, heads, n, n)
        attn_scores = queries @ keys.transpose(2, 3)

        # Step 5: Causal Masking - 미래 토큰 참조 방지
        # 마스크된 위치에 -inf를 넣어 softmax 후 0이 되도록 함
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        # Step 6: Scaled Softmax
        # √d로 나눠서 스케일링 → 그래디언트 안정화
        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # Step 7: Value와 가중합
        # (batch, heads, n, n) @ (batch, heads, n, d) → (batch, heads, n, d)
        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### NEW: 출력 게이팅
        # sigmoid(gate): [0, 1] 범위로 출력 스케일링
        # 장점:
        #   1. Attention Sink 방지 (특정 토큰에 과도한 어텐션 집중)
        #   2. Massive Activation 방지 (비정상적으로 큰 활성화 값)
        #   3. 수치적 안정성 향상 (학습 안정화)
        context = context * torch.sigmoid(gate)
        ####################################################

        # Step 8: 최종 출력 프로젝션
        out = self.out_proj(context)
        return out
```



As we can see, after computing attention as usual, the model uses a separate gating signal from the same input, applies a sigmoid to keep it between 0 and 1, and multiplies it with the attention output. This allows the model to scale up or down certain features dynamically. The Qwen3-Next developers [state](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) that this helps with training stability:

> [...] the attention output gating mechanism helps eliminate issues like Attention Sink and Massive Activation, ensuring numerical stability across the model.


&nbsp;
## Gated DeltaNet

Now, what is Gated DeltaNet? Gated DeltaNet (short for *Gated Delta Network*) is Qwen3-Next's linear-attention layer, which is intended as an alternative to standard softmax attention. It was adopted from the [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) paper as mentioned earlier.

Gated DeltaNet was originally proposed as an improved version of Mamba2, where it combines the gated decay mechanism of Mamba2 with a delta rule.

Mamba is a state-space model (an alternative to transformers), a big topic that deserves separate coverage in the future.

The delta rule part refers to computing the difference (delta, Δ) between new and predicted values to update a hidden state that is used as a memory state (more on that later).

(Side note: Readers with classic machine learning literature can think of this as similar to Hebbian learning inspired by biology: "Cells that fire together wire together." It's basically a precursor of the perceptron update rule and gradient descent-based learning, but without supervision.)

Gated DeltaNet has a gate similar to the gate in gated attention discussed earlier, except that it uses a SiLU instead of logistic sigmoid activation, as illustrated below. (The SiLU choice is likely to improve gradient flow and stability over the standard sigmoid.)

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="Gated DeltaNet" width=500px>

However, as shown in the figure above, the "gated" in the Gated DeltaNet also refers to several additional gates:

- `α` (decay gate) controls how fast the memory decays or resets over time,
- `β` (update gate) controls how strongly new inputs modify the state.

In code, a simplified version of the Gated DeltaNet depicted above (without the convolutional mixing) can be implemented as follows (the code is inspired by the [official implementation](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835) by the Qwen3 team).

(Note that some implementations refer to the decay gate as `gk` (gate for step k), where `exp(gk)` matches the paper's $\alpha_t$. To keep this relationship explicit, the snippet below separates the log-space gate `alpha_log` from the exponentiated decay `alpha`.)

---

### GatedDeltaNet 코드 설명 (한글)

**GatedDeltaNet**은 일반 어텐션의 O(n²) 복잡도를 O(n)으로 줄이는 선형 어텐션 기법입니다.

#### 핵심 구조

```
         입력 x
            │
  ┌─────────┼─────────┬─────────┬─────────┐
  ▼         ▼         ▼         ▼         ▼
W_Q       W_K       W_V      W_gate    W_alpha, W_beta
  │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼
  Q         K         V       gate      α, β (게이트)
  │         │         │
  └────┬────┴────┬────┘
       │         │
       ▼         ▼
  ┌────────────────────────────────────────┐
  │  Delta Rule 상태 업데이트 (RNN 방식)   │
  │                                        │
  │  S = S × α          (메모리 감쇠)      │
  │  kv_mem = S × k     (메모리 조회)      │
  │  delta = (v - kv_mem) × β  (Delta 계산)│
  │  S = S + k × delta  (메모리 업데이트)  │
  │  y = S × q          (출력 생성)        │
  └────────────────────────────────────────┘
            │
            ▼
       RMSNorm + SiLU(gate)
            │
            ▼
          출력
```

#### 핵심 게이트 3가지

| 게이트 | 변수 | 역할 | 활성화 함수 |
|--------|------|------|-------------|
| **α (decay)** | `alpha` | 이전 메모리 유지율 (망각률) | `exp(-A * softplus(...))` |
| **β (update)** | `beta` | 새 정보 반영 강도 | `sigmoid()` |
| **output** | `gate` | 최종 출력 통과율 | `SiLU()` |

#### KV 캐시 vs DeltaNet 상태

| 항목 | MHA (일반 어텐션) | GatedDeltaNet |
|------|-------------------|---------------|
| 메모리 | `batch × n_tokens × heads × d_head × 2` | `batch × heads × d_head × d_head` |
| 컨텍스트 의존성 | **선형 증가** | **고정 (무관)** |
| 복잡도 | O(n²) | O(n) |

---


```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    """L2 정규화: 안정적인 학습을 위해 Q, K를 단위 벡터로 정규화"""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet: 선형 어텐션 + Delta Rule + Gating

    일반 어텐션: softmax(Q @ K.T) @ V  →  O(n²)
    DeltaNet: 고정 크기 상태 S를 순차 업데이트  →  O(n)
    """
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # 기본 Q, K, V 프로젝션 (일반 어텐션과 동일)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        ####################################################
        ### 게이트 정의 (DeltaNet 핵심!)
        ####################################################
        # 출력 게이트: 최종 출력을 얼마나 통과시킬지 (SiLU 적용)
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        # β (업데이트 게이트): 새 정보를 얼마나 반영할지 (sigmoid → [0,1])
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # α (감쇠 게이트): 이전 메모리를 얼마나 유지할지
        # alpha = exp(-A * softplus(W_alpha(x) + dt_bias))
        # A가 크면 → 빠른 망각, A가 작으면 → 오래 기억
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))

        # RMSNorm: 출력 안정화
        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Step 1: Q, K, V 계산 (일반 어텐션과 동일)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        ####################################################
        ### Step 2: 게이트 계산
        ####################################################
        # β: 업데이트 강도 [0, 1] - 새 정보를 얼마나 반영?
        beta = torch.sigmoid(self.W_beta(x))

        # α: 감쇠율 - 이전 메모리를 얼마나 유지?
        # alpha = exp(-A * softplus(W_alpha(x) + bias))
        # α ≈ 1: 메모리 유지, α ≈ 0: 빠른 망각
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()

        # 출력 게이트 (SiLU 적용 예정)
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # NEW

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)
        gate = gate.transpose(1, 2)  # NEW

        ####################################################
        ### Step 3: L2 정규화 (학습 안정성)
        ####################################################
        # Q, K를 단위 벡터로 정규화 → 내적 값 범위 제한
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)

        ####################################################
        ### Step 4: Delta Rule 상태 업데이트 (핵심!)
        ####################################################
        # 고정 크기 상태 행렬 S: (batch, heads, d_head, d_head)
        # → 컨텍스트 길이와 무관! (일반 어텐션의 n×n 행렬 대신)
        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        # 선형 복잡도 O(n)! (일반 어텐션은 O(n²))
        for t in range(num_tokens):
            k_t = keys[:, :, t]      # 현재 토큰의 Key
            q_t = queries[:, :, t]   # 현재 토큰의 Query
            v_t = values[:, :, t]    # 현재 토큰의 Value
            b_t = beta[:, :, t]      # 현재 토큰의 업데이트 강도
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)  # 감쇠율

            # 4-1. 메모리 감쇠: 이전 정보 일부 망각
            S = S * a_t

            # 4-2. 메모리 조회: 현재 키와 관련된 정보 추출
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)

            # 4-3. Delta 계산: (새 값 - 예측 값) × 업데이트 강도
            # "예측이 틀린 만큼만 학습" (Hebbian learning과 유사)
            delta = (v_t - kv_mem) * b_t

            # 4-4. 메모리 업데이트: k_t 위치에 delta 정보 저장
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

            # 4-5. 출력 생성: 쿼리로 메모리에서 정보 추출
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            outs.append(y_t)

        ####################################################
        ### Step 5: 출력 조합 및 게이팅
        ####################################################
        # 모든 토큰의 출력을 스택
        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        # RMSNorm으로 정규화 + SiLU 게이트로 출력 조절
        # SiLU(x) = x × sigmoid(x) → 부드러운 게이팅
        context = self.norm(context)
        context = context * F.silu(gate)

        # 최종 출력 프로젝션
        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

(Note that for simplicity, I omitted the convolutional mixing that Qwen3-Next and Kimi Linear use to keep the code more readable and focus on the recurrent aspects.)

So, as we can see above, there are lots of differences to standard (or gated) attention.

In gated attention, the model computes normal attention between all tokens (every token attends or looks at every other token). Then, after getting the attention output, a gate (a sigmoid) decides how much of that output to keep. The takeaway is that it's still the the regular scaled-dot product attention that scales quadratically with the context length.

As a refresher, scaled-dot production attention is computed as softmax(QKᵀ)V, where Q and K are *n*-by-*d* matrices, where *n* is the number of input tokens, and *d* is the embedding dimension. So QKᵀ results in an attention *n*-by-*n* matrix, that is multiplied by a *n*-by-*d* dimensional value matrix V:

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="Quadratic attention" width=500px />

In Gated DeltaNet, there's no  *n*-by-*n* attention matrix. Instead, the model processes tokens one by one. It keeps a running memory (a state) that gets updated as each new token comes in. This is what's implemented as, where `S` is the state that gets updated recurrently for each time step *t*.

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

And the gates control how that memory changes:

- α (`alpha`) regulates how much of the old memory to forget (decay).

- β (`beta`) regulates how much the current token at time step *t* updates the memory.

(And the final output gate, not shown in the snippet above, is similar to gated attention; it controls how much of the output is kept.)

So, in a sense, this state update in Gated DeltaNet is similar to how recurrent neural networks (RNNs) work. The advantage is that it scales linearly (via the for-loop) instead of quadratically with context length.

The downside of this recurrent state update is that, compared to regular (or gated) attention, it sacrifices the global context modeling ability that comes from full pairwise attention.

Gated DeltaNet, can, to some extend, still capture context, but it has to go through the memory (*S*) bottleneck. That memory is a fixed size and thus more efficient, but it compresses past context into a single hidden state similar to RNNs.

That's why the Qwen3-Next and Kimi Linear architectures don't replace all attention layers with DeltaNet layers but use the 3:1 ratio mentioned earlier.

&nbsp;
## DeltaNet Memory Savings

In the previous section, we discussed the advantage of the DeltaNet over full attention in terms of linear instead of quadratic compute complexity with respect to the context length.

Next to the linear compute complexity, another big advantage of DeltaNet is the memory savings, as DeltaNet modules don't grow the KV cache. (For more information about KV caching, see [../03_kv-cache](../03_kv-cache)). Instead, as mentioned earlier, they keep a fixed-size recurrent state, so memory stays constant with context length.

For a regular multi-head attention (MHA) layer, we can compute the KV cache size as follows:

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

(The 2 multiplier is there because we have both keys and values that we store in the cache.)

For the simplified DeltaNet version implemented above, we have:


```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

Note that the `KV_cache_DeltaNet` memory size doesn't have a context length (`n_tokens`) dependency. Also, we have only the memory state S that we store instead of separate keys and values, hence `2 × bytes` becomes just `bytes`. However, note that we now have a quadratic `d_head × d_head` in here. This comes from the state :

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

But that's usually nothing to worry about, as the head dimension is usually relatively small. For instance, it's 128 in Qwen3-Next.

The full version with the convolutional mixing is a bit more complex, including the kernel size and so on, but the formulas above should illustrate the main trend and motivation behind the Gated DeltaNet.

We can visualize the memory estimates and savings for different context lengths via the following helper script:

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

Note that the above computes the `head_dim` as `emb_dim / n_heads`. I.e., 2048 / 16  = 128.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="Gated DeltaNet scaling" width=500px>
