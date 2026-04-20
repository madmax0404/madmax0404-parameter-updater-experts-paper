내가 보기엔 **가장 먼저 바꿔야 할 실험 3개**는 이거야.

### 1) updater를 “평범한 expert”처럼 강제로 쓰게 만드는 압력을 먼저 제거하기

지금 제일 의심스러운 건 이거야. `TopKRouter`의 balance loss가 **updater까지 포함해서** 균등 분산을 유도하고 있어. 그래서 router가 updater를 “필요할 때만 켜는 특수 expert”가 아니라 그냥 **한 칸 차지하는 일반 expert**처럼 쓰고 있을 가능성이 커. 실제 결과도 updater activation rate가 거의 **0.50** 근처로 고정돼 있고, switch 근처에서만 특별히 더 많이 켜지는 그림도 강하지 않아 보여.  

그래서 첫 실험은 이렇게 가면 좋아:

* **balance loss 계산에서 updater expert 제외**
* updater는 따로 **target usage ratio**를 둬서 예를 들면 5~15% 정도만 쓰게 유도
* 추가로 **switch 근처에서는 updater 사용을 보상**, non-switch에서는 약하게 억제

성공 기준은 간단해:

* overall updater rate는 내려가고
* **updater_rate_at_switches / updater_activation_rate** 비율은 올라가고
* post-switch accuracy가 유의미하게 오르는지 보기

즉, 첫 번째 목표는
**“updater를 자주 쓰게 만들기”가 아니라 “필요할 때만 쓰게 만들기”**야.

---

### 2) “slot tax”를 분리하는 control 실험 넣기

지금 구조에서는 updater가 top-k 슬롯 하나를 차지하면, 그 토큰은 사실상 **실제 inference expert를 하나 덜 쓰는 셈**이야. 그러니까 성능이 안 나오면 그게
“updater 아이디어가 별로라서인지”
아니면
“slot 하나 뺏겨서 손해를 본 건지”
구분이 안 돼. `main.py`에서 파라미터 수는 맞췄지만, **routing slot budget**까지 분리해주진 못하고 있어. 

그래서 두 번째는 꼭 이런 control들을 넣는 게 좋아:

* **no-op updater**: updater slot은 차지하지만 delta는 0만 내보냄
* **random updater**: updater slot은 차지하지만 random/frozen delta 사용
* **delta-off updater**: updater는 선택되지만 실제 `forward_with_full_delta` 적용은 끔
* 가능하면 **same slot tax baseline**도 하나 두기

이 실험의 목적은 단 하나야:
**“현재 모델의 이득이 진짜 parameter update에서 오는지”**를 증명하는 것.

이 control 없이 성능이 좋아도 설득력이 약하고, 반대로 지금처럼 성능이 약할 때도 **무엇이 문제인지 분해가 안 돼.**

---

### 3) 벤치마크를 “업dater가 필요할 수밖에 없는 문제”로 더 세게 만들기

현재 데이터는 identity / reverse / shift 3개 규칙을 섞고, 중간에 switch를 넣는 구조야. 이건 PoC용으론 괜찮지만, 솔직히 말하면 transformer hidden state만으로도 꽤 많이 처리할 수 있는 문제처럼 보여. 즉 **weight update까지 동원해야 할 필요성**이 아직 약할 수 있어. 

그래서 세 번째는 데이터셋/태스크를 더 날카롭게 바꾸는 실험이 좋아:

* 각 episode마다 **숨겨진 domain rule**이 바뀌고
* 앞부분 몇 토큰으로 그 rule을 추론해야 하고
* 그 뒤 query 구간에서 계속 적용해야 하는 식
* 즉 **“문맥 상태만으론 버티기 어렵고, sequence 중간에 적응이 필요하다”**는 환경 만들기

예를 들면:

* 랜덤 permutation rule
* token remapping table
* 짧은 support 구간 뒤 긴 query 구간
* switch 직후 몇 토큰 안에 적응해야 하는 task

현재 summary를 보면 updater가 baseline보다 아주 약간 나은 신호는 있지만, 압도적 차이는 아니야. 그건 아이디어가 틀렸다기보다, **문제가 아직 updater의 장점을 강하게 요구하지 않는 것**일 수도 있어. 

---

### 내 우선순위는 이렇게야

1. **router/balance loss부터 수정**
2. **control 실험 추가**
3. **task를 더 updater-friendly 하게 강화**

이 3개를 먼저 하면,
“아이디어가 진짜 먹히는지”를 훨씬 빨리 알 수 있어.

그리고 하나만 덧붙이면, 그다음 단계에서 architecture 쪽으로는
`UpdaterExpert.forward()`가 현재 **모든 inference expert를 다 업데이트**하고 있으니까, 나중엔 **선택된 peer만 업데이트**하는 sparse 버전도 해볼 가치가 커 보여. 지금 구조는 핵심 아이디어를 구현하긴 했지만, 아직은 좀 넓게 업데이트하고 있어. 

한 줄로 요약하면:

**지금 필요한 건 모델을 더 복잡하게 만드는 게 아니라, updater가 “언제”, “왜”, “정말 필요한지”를 실험적으로 분리해서 증명하는 것**이야.