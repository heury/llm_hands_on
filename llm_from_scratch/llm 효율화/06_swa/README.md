# 슬라이딩 윈도우 어텐션 (SWA)

이 보너스 자료는 일반적인 멀티 헤드 어텐션(MHA) 대신 슬라이딩 윈도우 어텐션(SWA)을 사용할 때의 메모리 절감 효과를 보여줍니다.



&nbsp;
## 소개 (Introduction)

슬라이딩 윈도우 어텐션(SWA)이란 무엇일까요? 일반적인 셀프 어텐션(self-attention)은 각 시퀀스 요소가 다른 모든 시퀀스 요소에 접근할 수 있기 때문에 *전역(global)* 어텐션 메커니즘으로 볼 수 있습니다. 반면, SWA는 현재 쿼리 위치 주변으로 컨텍스트 크기를 제한하기 때문에 *지역(local)* 어텐션으로 생각할 수 있습니다. 이는 아래 그림에 잘 나타나 있습니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

위 그림에서 볼 수 있듯이, 각 토큰은 이전의 모든 토큰에 주의를 기울이는(attend) 대신, 자신의 위치 주변의 고정된 크기의 로컬 윈도우에만 주의를 기울입니다. 이러한 국소화된 어텐션은 KV 캐시(KV cache)의 크기를 상당히 줄여줍니다.

이 소개의 나머지 부분에서는 [../../ch05/12_gemma3](../../ch05/12_gemma3)에서 처음부터 구현된 [Gemma 3](https://arxiv.org/abs/2503.19786)의 맥락에서 SWA를 논의할 것입니다.

슬라이딩 윈도우 어텐션은 원래 [2020년 LongFormer 논문](https://arxiv.org/abs/2004.05150)에서 소개되었지만, 우리가 구글의 Gemma 모델에 주목하는 이유는 이들이 슬라이딩 윈도우 어텐션이 최신의 고성능 모델에서도 실제로 유용한 접근 방식임을 보여주는 매우 우수한 오픈 웨이트(open-weight) 모델이기 때문입니다.

[Gemma 2](https://arxiv.org/abs/2408.00118)는 로컬(슬라이딩 윈도우) 어텐션 층과 전역 어텐션 층을 1:1 비율로 결합한 하이브리드 접근 방식을 사용했습니다. 각 토큰은 4k(4096) 토큰의 컨텍스트 윈도우에 주의를 기울일 수 있었습니다. 1:1 하이브리드 방식을 쓴 이유는 로컬 어텐션만 사용하는 LLM은 너무 제한적일 수 있어, 효율성과 전역 컨텍스트 모델링 사이의 균형을 맞추기 위함이었습니다.

[Gemma 3](https://arxiv.org/abs/2503.19786)는 설계를 효율성 측면에서 한 단계 더 발전시켰습니다. 슬라이딩 윈도우와 전체 어텐션 층의 비율을 5:1로 사용했는데, 이는 로컬 어텐션 층 5개마다 전역 층 1개가 있다는 뜻입니다. 또한, 슬라이딩 윈도우 크기를 Gemma 2의 4096 토큰에서 Gemma 3에서는 1024 토큰으로 줄였습니다.

흥미롭게도, Gemma 3 기술 보고서의 ablation studies(구성 요소 제거 실험)에 따르면 이러한 변경 사항이 전체 모델 품질에 미치는 영향은 미미한 것으로 나타났습니다. 즉, 슬라이딩 윈도우 어텐션을 통해 달성한 상당한 메모리 및 연산 절감 효과가 모델링 성능의 손실을 거의 가져오지 않는다는 것입니다.



&nbsp;
## 슬라이딩 윈도우 어텐션(SWA)의 메모리 절감 효과

메모리 절감 효과는 주로 KV 저장소(storage)에 반영됩니다. KV 저장소 크기는 다음 공식으로 계산할 수 있습니다:

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

SWA를 사용할 때는 위의 시퀀스 길이(seqlen)를 윈도우 크기 W로 대체합니다. 따라서 슬라이딩 윈도우 어텐션을 사용하면 KV 캐시 크기를 "W / seqlen" 비율만큼 줄일 수 있습니다. (참고로, 이는 단순화를 위해 모든 층에서 슬라이딩 윈도우 어텐션이 사용된다고 가정한 것입니다.)


이 폴더에 있는 [memory_estimator_swa.py](memory_estimator_swa.py) 스크립트를 사용하여 다양한 모델 설정에 적용해 보고, MHA 대비 SWA 사용 시 메모리를 얼마나 절약할 수 있는지 확인할 수 있습니다:

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
MHA + GQA (Ratio: 5:1) : 0.78 GB