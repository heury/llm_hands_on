# 보너스 자료: KV 캐시 (KV Cache)



**이 폴더는 GPT 모델에 KV 캐시를 추가하는 구현을 다룹니다.**

&nbsp;
## 개요 (Overview)

요약하자면, KV 캐시는 추론(inference) 중에 중간 키(Key, K)와 값(Value, V) 계산 결과를 저장하여 재사용함으로써 응답 생성 속도를 상당히 높여줍니다. 단점은 코드 복잡도가 약간 증가하고 메모리 사용량이 늘어나며 훈련 중에는 사용할 수 없다는 것입니다. 하지만 LLM을 배포할 때 추론 속도 향상은 코드 복잡성이나 메모리 트레이드오프를 감수할 가치가 충분한 경우가 많습니다.

&nbsp;
## 작동 원리 (How it works)

LLM이 어떤 텍스트를 생성하고 있다고 상상해 봅시다. 구체적으로, LLM에 "Time flies"라는 프롬프트가 주어졌다고 가정해 보겠습니다.

아래 그림은 3장에서 수정한 그래픽을 사용하여 키(Key)와 값(Value) 벡터가 강조된 어텐션 점수 계산의 일부를 보여줍니다:

<img src="images/llm_from_scratch/bonus/kv-cache/kv-cache-attn-1.png" width=800>

이제 2장과 4장에서 배웠듯이 LLM은 한 번에 하나의 단어(또는 토큰)를 생성합니다. LLM이 "fast"라는 단어를 생성하여 다음 라운드의 프롬프트가 "Time flies fast"가 되었다고 가정해 봅시다. 이는 다음 그림에 설명되어 있습니다:

<img src="images/llm_from_scratch/bonus/kv-cache/kv-cache-attn-2.png" width=800>

이전 두 그림을 비교해 보면 알 수 있듯이, 처음 두 토큰에 대한 키와 값 벡터는 정확히 동일하며, 다음 토큰 텍스트 생성 라운드마다 이를 다시 계산하는 것은 낭비입니다.

따라서 KV 캐시의 아이디어는 이전에 생성된 키와 값 벡터를 저장하여 재사용하는 캐싱 메커니즘을 구현하여 불필요한 재계산을 피하는 것입니다.

&nbsp;

## KV 캐시 구현 (KV cache implementation)

KV 캐시를 구현하는 방법은 여러 가지가 있지만, 핵심 아이디어는 각 생성 단계에서 새로 생성된 토큰에 대한 키와 값 텐서만 계산한다는 것입니다.

저는 코드 가독성을 강조한 간단한 방법을 선택했습니다. 코드 변경 사항을 훑어보는 것이 구현 방법을 이해하는 데 가장 쉬울 것입니다.

이 폴더에는 두 개의 파일이 있습니다:

1. [`gpt_ch04.py`](gpt_ch04.py): LLM을 구현하고 간단한 텍스트 생성 함수를 실행하기 위해 3장과 4장에서 가져온 독립형 코드
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py): 위와 동일하지만 KV 캐시를 구현하기 위해 필요한 변경 사항이 적용된 코드

다음 두 가지 방법 중 하나로 확인할 수 있습니다.

a. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 파일을 열어 새로운 변경 사항을 표시하는 `# NEW` 섹션을 찾아보세요:

<img src="../images/llm_from_scratch/bonus/kv-cache/new-sections.png" width=800>

b. 원하는 파일 비교(diff) 도구를 통해 두 코드 파일을 확인하여 변경 사항을 비교하세요:

<img src="images/llm_from_scratch/bonus/kv-cache/file-diff.png" width=800>

구현 세부 사항을 요약하기 위해, 여기 짧은 단계별 설명이 있습니다.

&nbsp;

### 1. 캐시 버퍼 등록 (Registering the cache buffers)

`MultiHeadAttention` 생성자 내부에 단계별로 연결된(concatenated) 키와 값을 담을 두 개의 버퍼, `cache_k`와 `cache_v`를 추가합니다:

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)