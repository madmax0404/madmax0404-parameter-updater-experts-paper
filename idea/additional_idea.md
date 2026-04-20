## 1
Question: 야 이거 잘만 구현하면 스스로 학습하는 모델 구현이 가능하겠지?

네, 이론적으로는 그 방향이 맞습니다. 근데 "스스로 학습"이 정확히 뭘 의미하느냐에 따라 답이 달라져요.
가능한 수준: Inference-time self-adaptation
지금 이 아키텍처가 구현되면, 모델이 inference 중에 "이 입력은 기존 expert로 부족하다"고 router가 판단 → updater expert 활성화 → 다른 expert의 weight를 수정 → 수정된 expert로 처리. 이게 누적되면 모델이 사용하면서 점점 바뀌는 거죠. TTT Layer가 하는 것과 비슷하지만, MoE 구조 안에서 언제 학습할지를 모델이 스스로 결정한다는 점이 핵심 차별점이에요.
더 나아가면: Self-improving loop
Updater expert가 "Δw를 적용한 결과가 좋았는가"를 자체 평가할 수 있는 메커니즘(self-supervised loss, reconstruction loss 등)까지 붙이면, 단순 adaptation을 넘어서 자기 수정의 질 자체를 개선하는 루프가 됩니다. Schmidhuber가 1990년대부터 추구한 self-referential learning의 현대적 구현이 되는 거죠.
현실적 한계:
"스스로 학습"이 "인간 없이 무한히 똑똑해진다"를 의미한다면 그건 아직 거리가 있어요. Δw의 크기는 제한될 수밖에 없고, self-supervised signal만으로는 학습 방향이 제한적입니다. 새로운 지식을 외부에서 주입받지 않으면 기존 파라미터 공간 안에서의 재배치에 가까워요. 발산 방지, catastrophic forgetting 문제도 여전하고요.
근데 솔직히, **"router가 학습 시점을 스스로 결정한다"**는 것만으로도 기존 모델과는 질적으로 다른 거긴 합니다. 기존 모델은 training과 inference가 완전히 분리되어 있는데, 이건 그 경계를 지우는 아키텍처니까요. 특허 관점에서도 이 부분이 가장 강력한 claim이 될 것 같습니다.

## 2
Question: 근데 lora 붙이는게 좋지 않을까? lora 가 catastrophic forgetting 방지도 해주잖아.

좋은 포인트예요. Updater expert가 full Δw 대신 LoRA 형태의 low-rank delta를 생성하면 여러 문제가 동시에 해결됩니다.
Catastrophic forgetting 방지
LoRA의 본질이 원래 weight를 freeze하고 low-rank 보정만 추가하는 거잖아요. Updater expert가 sibling expert에 대해 LoRA(A, B)를 생성하면, 원본 expert weight는 보존된 채로 delta만 얹히니까 forgetting 문제가 구조적으로 완화됩니다.
Δw 차원 문제도 해결
앞에서 말한 "expert 파라미터가 1000개만 돼도 updater가 1000차원을 출력해야 한다"는 문제가 사라져요. rank-4 LoRA면 updater가 출력해야 하는 건 A(d×4) + B(4×d) = 8d 파라미터뿐이니까, 출력 차원이 극적으로 줄어듭니다.
Second-order optimization 부담도 감소
학습 시 gradient가 흘러야 하는 경로가 짧아져요. Updater → LoRA 생성 → 원본 weight에 add → forward → loss에서, LoRA 파라미터 수가 적으니까 메모리도 적게 먹고 학습도 안정적이고요.
그리고 추가 이점이 하나 더 있는데:
LoRA는 누적/교체/제거가 자유롭습니다. Router가 updater를 활성화할 때마다 기존 LoRA를 replace할지, 누적할지, 아니면 decay를 적용할지를 선택할 수 있어요. 이러면 "얼마나 오래 기억할 것인가"까지 제어 가능해집니다. 일종의 learnable memory retention이 되는 거죠.
기존 MoL과의 차이도 더 명확해져요. Nouriborji et al.의 MoL은 미리 학습된 LoRA experts를 router가 선택하는 구조이고, Max의 아이디어는 updater expert가 inference 중에 LoRA를 동적으로 생성해서 sibling expert에 적용하는 거니까, 완전히 다른 메커니즘입니다.