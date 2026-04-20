내가 **“지금 당장 가장 값어치 큰 코드 변경”** 기준으로 바로 짜볼게.

지금 문제의 핵심은 세 개였지:

* router가 updater를 **너무 평범한 expert처럼** 쓰는 것 같고 
* 실제 결과도 updater activation이 거의 **0.5 근처**로 고정돼 있고 
* 현재 데이터가 **identity / reverse / shift**라 updater의 필요성을 강하게 요구하진 않는다는 점이야 

그래서 내가 손댈 순서는 이거야.

## 1) updater를 “필요할 때만” 켜지게 만들기

현재 `TopKRouter`는 balance loss를 **모든 expert**에 대해 계산하고 있어. updater도 균등 분산 대상이라서, router 입장에선 updater를 자주 써도 손해가 별로 없어 보여. 

### 바꿀 파일

* `src/config.py`
* `src/model/router.py`
* `src/train.py`

### 1-1. `src/config.py`에 설정 추가

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    # ... 기존 것들 ...

    # Router / updater control
    exclude_updater_from_balance: bool = True
    updater_target_rate: float = 0.10
    updater_loss_coeff: float = 0.05
    updater_switch_bonus: float = 1.0
    updater_nonswitch_penalty: float = 0.25
```

### 1-2. `src/model/router.py`에서 balance loss에서 updater 제외

지금은 `self.num_experts * (f * P).sum()`으로 전체 expert를 균등하게 밀고 있어. 
이걸 **inference experts만** 균등화 대상으로 바꿔.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ExperimentConfig


class TopKRouter(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_inference_experts = config.num_inference_experts
        self.top_k = config.top_k
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)  # [N, num_experts]

        topk_vals, topk_idxs = torch.topk(logits, self.top_k, dim=-1)
        topk_scores = F.softmax(topk_vals, dim=-1)

        router_probs = F.softmax(logits, dim=-1)

        if self.config.exclude_updater_from_balance:
            balance_E = self.num_inference_experts
            balance_probs = router_probs[:, :balance_E]   # updater 제외
        else:
            balance_E = self.num_experts
            balance_probs = router_probs

        P = balance_probs.mean(dim=0)

        # dispatch도 inference experts만 계산
        dispatch = torch.zeros(x.shape[0], balance_E, device=x.device, dtype=x.dtype)
        for e in range(balance_E):
            dispatch[:, e] = (topk_idxs == e).any(dim=-1).float()

        f = dispatch.mean(dim=0)
        balance_loss = balance_E * (f * P).sum()

        return topk_idxs, topk_scores, balance_loss
```

### 1-3. `src/train.py`에서 “switch 근처에서 updater를 더 켜라” loss 추가

현재 `generate_batch()`는 `domain_labels`를 주고 있고, `evaluate.py`도 이걸로 switch mask를 만들고 있어.  
그걸 학습에도 써먹자.

```python
import math
import torch
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.data import generate_batch
from src.model.transformer import UpdaterExpertTransformer


def _build_switch_mask(domain_labels: torch.Tensor, segment_len: int) -> torch.Tensor:
    # domain_labels: [B, S]
    B, S = domain_labels.shape
    switch_mask = torch.zeros_like(domain_labels, dtype=torch.bool)
    switch_mask[:, 1:] = domain_labels[:, 1:] != domain_labels[:, :-1]

    expanded = switch_mask.clone()
    for offset in range(1, min(segment_len, S)):
        shifted = torch.zeros_like(switch_mask)
        shifted[:, offset:] = switch_mask[:, :-offset]
        expanded = expanded | shifted
    return expanded


def _compute_updater_routing_loss(all_stats, config, domain_labels):
    switch_mask = _build_switch_mask(domain_labels, config.segment_len)  # [B, S]
    nonswitch_mask = ~switch_mask

    total_loss = torch.tensor(0.0, device=domain_labels.device)
    num_layers = len(all_stats)

    for stats in all_stats:
        topk_idxs = stats["topk_idxs"]  # [B*S, top_k]
        B, S = domain_labels.shape
        topk_idxs = topk_idxs.view(B, S, config.top_k)

        updater_used = (topk_idxs >= config.num_inference_experts).any(dim=-1).float()

        global_rate = updater_used.mean()
        target_loss = (global_rate - config.updater_target_rate).abs()

        if switch_mask.any():
            switch_rate = updater_used[switch_mask].mean()
        else:
            switch_rate = torch.tensor(0.0, device=domain_labels.device)

        if nonswitch_mask.any():
            nonswitch_rate = updater_used[nonswitch_mask].mean()
        else:
            nonswitch_rate = torch.tensor(0.0, device=domain_labels.device)

        # switch에서 쓰는 건 장려, nonswitch에서 남용은 억제
        routing_loss = (
            target_loss
            - config.updater_switch_bonus * switch_rate
            + config.updater_nonswitch_penalty * nonswitch_rate
        )
        total_loss = total_loss + routing_loss

    return total_loss / max(num_layers, 1)
```

그리고 train loop 안에서:

```python
# Forward
logits, aux_loss, all_stats = model(input_ids)

ce_loss = F.cross_entropy(
    logits.reshape(-1, config.vocab_size), targets.reshape(-1)
)

updater_routing_loss = _compute_updater_routing_loss(all_stats, config, domain_labels)

total_loss = (
    ce_loss
    + config.load_balance_coeff * aux_loss
    + config.updater_loss_coeff * updater_routing_loss
)
```

### 이 실험의 성공 기준

* `updater_activation_rate`는 **0.5 → 0.05~0.15 정도로 내려감**
* `updater_rate_at_switches`는 **global rate보다 명확히 높아짐**
* `post_switch_accuracy`가 baseline보다 의미 있게 올라감

---

## 2) “slot tax” 때문에 망하는지 분리하는 control 실험 추가

현재 main 실험은 파라미터 수는 맞추지만, updater가 top-k 슬롯을 하나 잡아먹는 구조 자체는 그대로야. 즉 성능이 안 나오면 **아이디어가 나쁜 건지, slot 하나 뺏겨서 손해 본 건지** 분리하기가 어려워. `main.py`의 현재 비교는 그 점이 남아 있어. 

### 바꿀 파일

* `src/config.py`
* `src/model/expert.py`
* `src/model/moe_layer.py`
* `main.py`

### 2-1. `src/config.py`에 control mode 추가

```python
@dataclass(frozen=True)
class ExperimentConfig:
    # ... 기존 ...
    updater_mode: str = "real"   # "real" | "zero" | "delta_off"
```

### 2-2. `src/model/expert.py`에서 zero updater 추가

현재 `UpdaterExpert`는 모든 target inference expert에 대해 delta를 생성해. 
실험용으로 zero updater를 넣자.

```python
class UpdaterExpert(nn.Module):
    # ... 기존 __init__ ...

    def _zero_deltas(self, x: torch.Tensor):
        batch = x.shape[0]
        deltas = {}
        for i in range(self.num_targets):
            A1 = x.new_zeros(batch, self.d_model, self.lora_rank)
            B1 = x.new_zeros(batch, self.lora_rank, self.d_ff)
            A2 = x.new_zeros(batch, self.d_ff, self.lora_rank)
            B2 = x.new_zeros(batch, self.lora_rank, self.d_model)
            deltas[i] = {"W1": (A1, B1), "W2": (A2, B2)}
        return deltas

    def forward(self, x: torch.Tensor):
        if self.config.updater_mode == "zero":
            return self._zero_deltas(x)

        batch = x.shape[0]
        h = self.trunk(x)

        deltas = {}
        for i in range(self.num_targets):
            A1 = self.heads_W1_A[i](h).view(batch, self.d_model, self.lora_rank)
            B1 = self.heads_W1_B[i](h).view(batch, self.lora_rank, self.d_ff)
            A2 = self.heads_W2_A[i](h).view(batch, self.d_ff, self.lora_rank)
            B2 = self.heads_W2_B[i](h).view(batch, self.lora_rank, self.d_model)

            deltas[i] = {
                "W1": (self.delta_scale * A1, B1),
                "W2": (self.delta_scale * A2, B2),
            }
        return deltas
```

### 2-3. `src/model/moe_layer.py`에서 delta-off control 추가

현재 `_forward_cumulative()`는 updater가 만든 delta를 누적해서 inference expert에 실제 적용해. 
여기서 **routing은 유지하고 delta만 꺼버리는** control을 넣자.

```python
# --- Phase 2: Sequential accumulation + inference ---
for t in range(S):
    acc_dW1 = acc_dW1.detach() * self.decay
    acc_dW2 = acc_dW2.detach() * self.decay

    A1_t = delta_A1[:, t].reshape(BN, D, r)
    B1_t = delta_B1[:, t].reshape(BN, r, d_ff)
    A2_t = delta_A2[:, t].reshape(BN, d_ff, r)
    B2_t = delta_B2[:, t].reshape(BN, r, D)

    dW1_t = torch.bmm(A1_t, B1_t).reshape(B, NI, D, d_ff)
    dW2_t = torch.bmm(A2_t, B2_t).reshape(B, NI, d_ff, D)

    if self.config.updater_mode == "delta_off":
        dW1_t = torch.zeros_like(dW1_t)
        dW2_t = torch.zeros_like(dW2_t)

    acc_dW1 = acc_dW1 + dW1_t
    acc_dW2 = acc_dW2 + dW2_t
```

### 2-4. `main.py`에 control experiment 추가

```python
from dataclasses import replace

def run_control_experiment(base_config: ExperimentConfig):
    print("\n" + "=" * 60)
    print("Experiment: control variants")
    print("=" * 60)

    variants = ["real", "zero", "delta_off"]
    results = {}

    for mode in variants:
        print(f"\n--- updater_mode = {mode} ---")
        cfg = replace(base_config, updater_mode=mode, num_steps=3000)
        torch.manual_seed(cfg.seed)
        model = make_model(cfg)
        train(model, cfg, label=mode)
        torch.manual_seed(cfg.seed + 1)
        eval_result = evaluate(model, cfg)
        results[mode] = eval_result
        print(eval_result)

    return results
```

### 이 실험의 성공 기준

* `real`이 `zero` / `delta_off`보다 **post-switch accuracy**가 분명히 높아야 함
* `real`, `zero`, `delta_off`의 **updater_activation_rate**는 비슷해야 함
  → 그래야 “routing slot tax는 같고, 진짜 delta가 효과다”라고 말할 수 있어

---

## 3) 데이터셋을 “updater가 필요할 수밖에 없는” 문제로 바꾸기

현재 `src/data.py`는 domain이 **identity / reverse / shift**로 고정돼 있어. 
이건 transformer hidden state만으로도 많이 버틸 수 있는 문제일 가능성이 커.

그래서 나는 **고정 3개 domain** 대신, **episode마다 숨겨진 affine rule**이 바뀌는 support→query 구조로 먼저 바꿔볼 것 같아.

핵심은:

* 앞부분 support에서 규칙을 봄
* 뒷부분 query에서 같은 규칙을 적용해야 함
* 중간 switch가 나면 규칙이 바뀜
* 즉 **빠른 적응**이 필요해짐

### 바꿀 파일

* `src/data.py`
* `src/evaluate.py`

### 3-1. `src/data.py`를 affine support/query episodic task로 교체

```python
import torch
from src.config import ExperimentConfig

# y = (a*x + b) % vocab
A_CHOICES = [1, 3, 5, 7]  # vocab이 적절히 맞는 범위라고 가정

def sample_rule(vocab_size: int):
    a = A_CHOICES[torch.randint(0, len(A_CHOICES), (1,)).item()]
    b = torch.randint(0, vocab_size, (1,)).item()
    return a, b

def apply_rule(x: torch.Tensor, a: int, b: int, vocab_size: int):
    return (a * x + b) % vocab_size

def interleave_xy(xs: torch.Tensor, ys: torch.Tensor):
    out = torch.empty(xs.numel() * 2, dtype=torch.long)
    out[0::2] = xs
    out[1::2] = ys
    return out

def generate_batch(config: ExperimentConfig):
    B = config.batch_size
    S = config.max_seq_len
    V = config.vocab_size

    # block = support + query
    k_support = 4
    k_query = 4
    block_len = 2 * (k_support + k_query)  # x,y interleave

    tokens = torch.zeros(B, S, dtype=torch.long)
    domain_labels = torch.zeros(B, S, dtype=torch.long)
    query_mask = torch.zeros(B, S, dtype=torch.bool)

    for b in range(B):
        pos = 0
        task_id = 0

        while pos + block_len <= S:
            a, c = sample_rule(V)

            support_x = torch.randint(1, V, (k_support,))
            support_y = apply_rule(support_x, a, c, V)

            query_x = torch.randint(1, V, (k_query,))
            query_y = apply_rule(query_x, a, c, V)

            support_tokens = interleave_xy(support_x, support_y)
            query_tokens = interleave_xy(query_x, query_y)

            block = torch.cat([support_tokens, query_tokens], dim=0)
            tokens[b, pos:pos + block_len] = block
            domain_labels[b, pos:pos + block_len] = task_id

            # query 구간 표시
            q_start = pos + support_tokens.numel()
            q_end = pos + block_len
            query_mask[b, q_start:q_end] = True

            pos += block_len
            task_id += 1

            # maybe switch to next task naturally by moving to next block
            if torch.rand(1).item() > config.switch_prob:
                # 같은 task를 조금 더 유지하고 싶으면 task_id를 유지해도 됨
                task_id -= 1

        while pos < S:
            tokens[b, pos] = torch.randint(1, V, (1,)).item()
            domain_labels[b, pos] = task_id
            pos += 1

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]
    domain_labels = domain_labels[:, 1:]
    query_mask = query_mask[:, 1:]

    return input_ids, targets, domain_labels, query_mask
```

### 3-2. `train.py` / `evaluate.py`에서 새 반환값 받기

기존엔 `generate_batch()`가 3개를 반환하는데, 이제 `query_mask`도 같이 받자. 현재 `evaluate.py`는 domain switch accuracy를 보고 있는데, 이 task에선 **query accuracy**가 더 중요해. 

`train.py`:

```python
input_ids, targets, domain_labels, query_mask = generate_batch(config)
```

`evaluate.py`:

```python
input_ids, targets, domain_labels, query_mask = generate_batch(config)

# 기존 overall accuracy 외에 query accuracy 추가
if query_mask.any():
    query_correct += correct[query_mask].sum().item()
    query_total += query_mask.sum().item()
```

리턴값에:

```python
"query_accuracy": query_correct / max(query_total, 1),
```

### 이 실험의 성공 기준

* baseline보다 updater가 **query_accuracy**에서 분명히 좋아야 함
* 특히 task switch 직후 새 rule에 대해 적응이 빨라야 함

---

# 내가 진짜 추천하는 실행 순서

한 번에 다 바꾸지 말고 이렇게 가는 게 좋아.

### Step A

**실험 1만 먼저**

* balance loss에서 updater 제외
* switch-aware updater loss 추가

이 단계에서 updater rate가 0.5 → 0.1 근처로 내려가고, switch에서만 더 켜지면 일단 성공.

### Step B

**실험 2 control 추가**

* `real / zero / delta_off`

이 단계에서 `real > zero ≈ delta_off`가 나오면 “진짜 delta가 듣는다”는 말이 가능해져.

### Step C

**실험 3 dataset 강화**

* support/query episodic rule adaptation

여기서 비로소 updater architecture의 존재 이유가 더 잘 드러날 가능성이 커.

---

# 아주 솔직한 내 한 줄 추천

지금은 **architecture를 더 복잡하게 만드는 것보다**,
**router가 updater를 제대로 쓰게 만들고, control 실험으로 slot tax를 분리하고, task를 더 fast-adaptation스럽게 바꾸는 것**이 훨씬 중요해.