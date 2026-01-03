# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Code to run the exercises; see exercise-solutions.ipynb for more information

from functools import partial
from importlib.metadata import version
import json
import math
import os
import re
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import requests
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from previous_chapters import  load_gpt2_model
# 로컬 파일에서 필요한 함수와 클래스 임포트 (이전 챕터에서 작성된 코드들)
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    load_gpt2_model,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)

# -----------------------------------------------------------------------------
# 1. 데이터셋 클래스 정의
# -----------------------------------------------------------------------------

class InstructionDataset(Dataset):
    """
    기본 Instruction 데이터셋 클래스.
    입력(Instruction + Input)과 출력(Response)을 하나의 텍스트로 합쳐서 토크나이징합니다.
    """
    def __init__(self, data, tokenizer):
        self.data = data

        # 데이터를 미리 토크나이징하여 저장 (학습 속도 향상)
        self.encoded_texts = []
        for entry in data:
            # Alpaca 스타일로 포맷팅 (Instruction + Input)
            instruction_plus_input = format_input(entry)
            # 정답(Response) 부분 포맷팅
            response_text = f"\n\n### Response:\n{entry['output']}"
            # 전체 텍스트 결합
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetWithMasking(Dataset):
    """
    [마스킹 기능 지원] Instruction 데이터셋 클래스.
    모델이 '질문(Instruction)'은 학습하지 않고 '답변(Response)'만 학습하도록 하기 위해,
    질문 부분의 길이를 별도로 저장합니다.
    """
    def __init__(self, data, tokenizer):
        self.data = data

        # 지시문의 길이를 저장할 리스트 (나중에 마스킹에 사용)
        self.instruction_lengths = []
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

            # 지시문 부분(질문)의 길이만 따로 계산하여 저장
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    def __getitem__(self, index):
        # 데이터와 함께 지시문의 길이도 반환
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetPhi(Dataset):
    """
    [Phi-3 프롬프트 스타일] 데이터셋 클래스.
    Microsoft의 Phi-3 모델에서 사용하는 <|user|>, <|assistant|> 태그 형식을 사용합니다.
    """
    def __init__(self, data, tokenizer):
        self.data = data

        self.encoded_texts = []
        for entry in data:
            # Phi-3 스타일로 입력 포맷팅 (<|user|>...)
            instruction_plus_input = format_input_phi(entry)
            # Phi-3 스타일로 답변 포맷팅 (<|assistant|>...)
            response_text = f"\n<|assistant|>:\n{entry['output']}"

            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# -----------------------------------------------------------------------------
# 2. LoRA (Low-Rank Adaptation) 관련 클래스
#    - 거대 모델의 가중치를 얼리고(Freeze), 작은 행렬(A, B)만 학습하는 효율적 튜닝 기법
# -----------------------------------------------------------------------------

class LinearWithLoRA(torch.nn.Module):
    """
    기존 Linear 레이어에 LoRA 레이어를 추가한 래퍼(Wrapper) 클래스.
    출력 = 기존_Linear(x) + LoRA(x)
    """
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # 원래 가중치의 결과에 LoRA 경로의 결과를 더함
        return self.linear(x) + self.lora(x)


class LoRALayer(torch.nn.Module):
    """
    실제 LoRA 로직이 구현된 레이어.
    W' = W + (A @ B) * scaling
    여기서 A와 B는 매우 작은 차원(rank)을 가집니다.
    """
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # A 행렬: 입력 차원 -> Rank (가중치 초기화 적용)
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) 
        # B 행렬: Rank -> 출력 차원 (0으로 초기화 -> 학습 시작 시 영향력 0)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        # LoRA 수식: alpha * (x A B)
        x = self.alpha * (x @ self.A @ self.B)
        return x


def replace_linear_with_lora(model, rank, alpha):
    """
    모델 내의 모든 Linear 레이어를 찾아 LinearWithLoRA로 교체하는 재귀 함수.
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Linear 레이어를 발견하면 LoRA가 적용된 버전으로 교체
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 하위 모듈에 대해 재귀적으로 탐색
            replace_linear_with_lora(module, rank, alpha)

# -----------------------------------------------------------------------------
# 3. 데이터 배치 처리(Collate) 함수
# -----------------------------------------------------------------------------

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    일반 배치 처리 함수.
    - 배치 내 가장 긴 문장에 맞춰 패딩(padding)을 추가합니다.
    - 입력(inputs)과 정답(targets)을 생성합니다. (targets는 inputs를 한 칸 시프트한 것)
    """
    # 배치 내 가장 긴 시퀀스 길이 찾기
    batch_max_length = max(len(item)+1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 문장 끝에 <|endoftext|> 토큰 추가
        new_item += [pad_token_id]
        # 최대 길이에 맞춰 패딩 추가
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        # 입력: 마지막 토큰 제외 / 정답: 첫 번째 토큰 제외 (Next token prediction)
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # 패딩 부분은 손실(Loss) 계산에서 제외하기 위해 ignore_index(-100)로 설정
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 최대 길이 제한 (메모리 관리)
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def custom_collate_with_masking_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    [마스킹 적용] 배치 처리 함수.
    - 기본 기능은 위와 같으나, '지시문(Instruction)' 부분의 타겟을 -100으로 마스킹합니다.
    - 이렇게 하면 모델은 질문을 읽고 답변을 생성할 때 발생하는 오차에 대해서만 학습합니다.
    """
    # 배치 내 가장 긴 길이 찾기 (튜플의 두 번째 요소가 텍스트 데이터임)
    batch_max_length = max(len(item)+1 for instruction_length, item in batch)

    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # 패딩 부분 마스킹
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 핵심: 지시문(Instruction) 길이만큼 타겟을 -100으로 설정하여 학습 제외
        targets[:instruction_length-1] = -100

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# -----------------------------------------------------------------------------
# 4. 유틸리티 함수 (다운로드, 포맷팅, 플로팅)
# -----------------------------------------------------------------------------

def download_and_load_file(file_path, url):
    """파일을 다운로드하거나 이미 있으면 로드합니다."""
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_input_phi(entry):
    """Phi-3 모델 스타일의 입력 포맷 (<|user|>...)"""
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )
    input_text = f"\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def format_input(entry):
    """Alpaca 스타일의 입력 포맷 (### Instruction: ...)"""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_name):
    """학습 손실 그래프를 그리고 저장합니다."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 에포크 기준 손실 그래프
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 처리한 토큰 수 기준 보조 축 생성
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)
    # plt.show()

# -----------------------------------------------------------------------------
# 5. 메인 실행 함수
# -----------------------------------------------------------------------------

def main(mask_instructions=False, alpaca52k=False, phi3_prompt=False, lora=False):
    # 패키지 버전 출력
    print()
    pkgs = ["matplotlib", "tiktoken", "torch", "tqdm", ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")

    # ---------------------------------------------------------
    # 데이터셋 준비
    # ---------------------------------------------------------
    file_path = "instruction-data.json"

    # Alpaca 52k 데이터셋 또는 저자의 소규모 데이터셋 선택
    if alpaca52k:
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    else:
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    # 데이터 분할 (Train 85% / Test 10% / Val 5%)
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    # 토크나이저 및 디바이스 설정
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    # 설정에 따른 최대 길이 제한
    if alpaca52k:
        allowed_max_length = 512
    else:
        allowed_max_length = 1024

    # 예외 처리: 마스킹과 Phi-3 프롬프트 동시 사용 미구현
    if mask_instructions and phi3_prompt:
        raise ValueError("Simultaneous support for instruction masking and the Phi-3 prompt template has not been implemented, yet.")

    # 옵션에 따라 데이터셋 클래스와 콜레이트 함수 선택
    if mask_instructions:
        customized_collate_fn = partial(custom_collate_with_masking_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetWithMasking
    elif phi3_prompt:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetPhi
    else:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDataset

    num_workers = 0
    batch_size = 4 if alpaca52k else 8

    torch.manual_seed(123)

    # 데이터 로더 생성
    train_dataset = CustomDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = CustomDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    # ---------------------------------------------------------
    # 모델 로드 (GPT-2 Medium)
    # ---------------------------------------------------------
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_name = "gpt2-medium-355M.pth"
    model = load_gpt2_model(model_name, BASE_CONFIG)
    model.eval()
    model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    # ---------------------------------------------------------
    # LoRA 적용 (옵션)
    # ---------------------------------------------------------
    if lora:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters before: {total_params:,}")

        # 기존 모델의 모든 파라미터를 Freeze (학습 불가능하게 설정)
        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters after freezing: {total_params:,}")
        
        # Linear 레이어를 LoRA 레이어로 교체
        replace_linear_with_lora(model, rank=16, alpha=16)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LoRA parameters: {total_params:,}")
        model.to(device)

    # ---------------------------------------------------------
    # 모델 학습 (Fine-tuning)
    # ---------------------------------------------------------
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()
    num_epochs = 2
    
    # AdamW 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    torch.manual_seed(123)

    # 학습 시작 시 테스트해볼 문맥 설정
    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input(val_data[0])

    # 실제 학습 루프 실행 (train_model_simple 함수는 이전 챕터 코드)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 결과 플로팅
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_name = "outputs/loss-plot.pdf"
    
    # 파일 이름에 실험 설정 반영
    if mask_instructions:
        plot_name = plot_name.replace(".pdf", "-mask-instructions.pdf")
    if alpaca52k:
        plot_name = plot_name.replace(".pdf", "-alpaca52k.pdf")
    if phi3_prompt:
        plot_name = plot_name.replace(".pdf", "-phi3-prompt.pdf")
    if lora:
        plot_name = plot_name.replace(".pdf", "-lora.pdf")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        plot_name = plot_name.replace(".pdf", "-baseline.pdf")

    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_name)
    print(50*"-")

    # ---------------------------------------------------------
    # 결과 생성 및 저장
    # ---------------------------------------------------------
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input_phi(entry) if phi3_prompt else format_input(entry)

        # 텍스트 생성 (Inference)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        # 생성된 텍스트에서 프롬프트 부분 제거하고 응답만 추출
        if phi3_prompt:
            response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
        else:
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    # 결과 JSON 및 모델 가중치 저장
    test_data_path = "datas/instruction-data-with-response.json"
    file_name = f"outputs/{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"

    if mask_instructions:
        test_data_path = test_data_path.replace(".json", "-mask-instructions.json")
        file_name = file_name.replace(".pth", "-mask-instructions.pth")
    if alpaca52k:
        test_data_path = test_data_path.replace(".json", "-alpaca52k.json")
        file_name = file_name.replace(".pth", "-alpaca52k.pth")
    if phi3_prompt:
        test_data_path = test_data_path.replace(".json", "-phi3-prompt.json")
        file_name = file_name.replace(".pth", "-phi3-prompt.pth")
    if lora:
        test_data_path = test_data_path.replace(".json", "-lora.json")
        file_name = file_name.replace(".pth", "-lora.pth")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        test_data_path = test_data_path.replace(".json", "-baseline.json")
        file_name = file_name.replace(".pth", "-baseline.pth")

    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Responses saved as {test_data_path}")

    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Instruction finetune a GPT model"
    )
    # 실행 가능한 옵션들 정의
    options = {"baseline", "mask_instructions", "alpaca_52k", "phi3_prompt", "lora"}
    parser.add_argument(
        "--exercise_solution",
        type=str,
        default="baseline",
        help=(
            f"Which experiment to run. Options: {options}."
        )
    )
    args = parser.parse_args()

    # 인자에 따라 다른 모드로 main 함수 실행
    if args.exercise_solution == "baseline":
        main()
    elif args.exercise_solution == "mask_instructions":
        main(mask_instructions=True)
    elif args.exercise_solution == "alpaca_52k":
        main(alpaca52k=True)
    elif args.exercise_solution == "phi3_prompt":
        main(phi3_prompt=True)
    elif args.exercise_solution == "lora":
        main(lora=True)
    else:
        raise ValueError(f"{args.exercise_solution} is not a valid --args.exercise_solution option. Options: {options}")