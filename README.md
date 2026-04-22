# Megatron-Bridge RL: SimPO Training for Nemotron Nano V2 VL

SimPO를 활용한 Nemotron Nano V2 VL 선호도 최적화(Preference Optimization) 학습 코드입니다.

> 기반 코드: [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)

---

## SimPO란?

[SimPO](https://arxiv.org/abs/2405.14734) (Simple Preference Optimization)는 reference model이 필요 없는 선호도 최적화 방법입니다. Length-normalised average log-probability를 implicit reward로 사용하여, DPO 대비 메모리와 연산 비용을 절반으로 줄입니다.

**Reward:**

$$r(y) = \frac{\beta}{|y|} \sum_t \log p(y_t \mid y_{<t}, x)$$

**Loss:**

$$\mathcal{L} = -\log \sigma\big(r(y_w) - r(y_l) - \gamma\big)$$

- $y_w$: chosen response, $y_l$: rejected response
- $\beta$: temperature (default: 2.0)
- $\gamma$: target reward margin (default: 0.5)

---

## Prerequisites

1. SFT 체크포인트 (Megatron-Bridge 또는 HuggingFace 형식)
2. Preference-pair 데이터셋:
   - **RLAIF-V** — HuggingFace에서 자동 다운로드
   - **SafeWatch DPO** — `chosen_conversation` / `rejected_conversation` 필드를 가진 JSONL

---

## Quick Start

### 1. RLAIF-V 데이터셋으로 학습 (자동 다운로드)

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/models/vlm/nemotron_vl/simpo_nemotron_nano_v2_vl.py \
    --pretrained-checkpoint /path/to/sft/checkpoint
```

### 2. SafeWatch DPO 데이터셋으로 학습

```bash
python -m torch.distributed.run --nproc_per_node=4 \
    examples/models/vlm/nemotron_vl/simpo_nemotron_nano_v2_vl.py \
    --pretrained-checkpoint /path/to/sft/checkpoint \
    --train-data /path/to/dpo_train.jsonl \
    --eval-data /path/to/dpo_eval.jsonl
```

### 3. 소량 데이터로 빠른 테스트

```bash
python -m torch.distributed.run --nproc_per_node=4 \
    examples/models/vlm/nemotron_vl/simpo_nemotron_nano_v2_vl.py \
    --pretrained-checkpoint /path/to/sft/checkpoint \
    --train-data /path/to/dpo_train.jsonl \
    --max-preference-samples 200
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--pretrained-checkpoint` | **(required)** | SFT 체크포인트 경로 또는 HuggingFace 모델 |
| `--beta` | `2.0` | SimPO temperature / scaling factor |
| `--gamma` | `0.5` | Chosen-rejected 간 target reward margin |
| `--train-data` | `None` | DPO JSONL 파일 경로 (생략 시 RLAIF-V 사용) |
| `--eval-data` | `None` | 평가용 JSONL 파일 경로 |
| `--max-preference-samples` | `None` | 로드할 preference pair 수 제한 |
| `--config-file` | `conf/nemotron_nano_v2_vl_simpo.yaml` | YAML override 파일 |

---

## Default Config

`examples/models/vlm/nemotron_vl/conf/nemotron_nano_v2_vl_simpo.yaml`

| Parameter | Value | Notes |
|---|---|---|
| `seq_length` | 4096 | |
| `train_iters` | 510 | |
| `global_batch_size` | 16 | |
| `micro_batch_size` | 1 | Preference *pair* 수 (실제 배치 = 2x) |
| `lr` | 5e-7 | SFT보다 낮은 학습률 |
| `tensor_model_parallel_size` | 2 | |
| `freeze_vision_model` | `true` | Vision encoder 고정 |
| `cross_entropy_loss_fusion` | `false` | SimPO는 raw logits 필요 |

---

## Logged Metrics

학습 중 다음 메트릭이 기록됩니다:

| Metric | Description |
|---|---|
| **simpo loss** | SimPO preference loss |
| **reward accuracy** | Chosen reward > rejected reward인 비율 |
| **chosen reward** | Chosen response의 평균 reward |
| **rejected reward** | Rejected response의 평균 reward |

---

## Inference & Evaluation

### SimPO 학습 후 추론

```bash
bash inference.sh /path/to/simpo/checkpoint /path/to/output_dir
```

### 평가

```bash
bash eval.sh
```

---

## File Structure

```
examples/models/vlm/nemotron_vl/
├── simpo_nemotron_nano_v2_vl.py              # SimPO 학습 entry point
├── finetune_nemotron_nano_v2_vl.py           # SFT 학습 / 추론 entry point
├── conf/
│   ├── nemotron_nano_v2_vl_simpo.yaml        # SimPO config
│   └── nemotron_nano_v2_vl_safewatch.yaml    # SafeWatch SFT config

src/megatron/bridge/training/
├── simpo_step.py                              # SimPO loss 계산 & forward step
├── llava_step.py                              # VLM SFT forward step
├── eval_dump.py                               # Eval dump 유틸리티

scripts/
├── eval_interval_guardrail_metrics.py         # Guardrail metrics 평가
├── hf_generate_eval.py                        # HF 기반 생성 평가
└── vllm_generate_eval.py                      # vLLM 기반 생성 평가
```
