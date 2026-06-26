# 주요 실험 요약

본 문서는 live birth prediction 프로젝트에서 최종 제출 후보를 만들기까지의 주요 실험 흐름을 요약한 기록입니다.

## 주요 실험표

| 실험명 | 모델 | Feature Engineering | 주요 파라미터 / 설정 | OOF / CV AUC | Public LB | 비고 |
|---|---|---|---|---:|---:|---|
| `baseline_catboost_all_features` | CatBoost | 기본 ART process feature + 전체 feature | depth=6, long training | 0.740185 | 0.7417 | 초기 강한 baseline. 첫 유효 제출 후보 |
| `lgbm_all_features_diversity` | LightGBM | CatBoost와 동일한 all features 계열 | LGBM 기본 튜닝 | 0.739621 | - | 단일 성능은 CatBoost보다 낮지만 blend diversity source로 활용 |
| `branch_specialist_diagnostics` | CatBoost / LGBM branch models | IVF/DI, fresh/frozen, transfer 여부 등 branch별 진단 | branch별 specialist 모델 | - | - | frozen/DI 일부 branch 개선 가능성 확인. global OOF 개선 부족 |
| `evidence_guided_depth7_all_evidence` | CatBoost | all_evidence features: age, egg source, SET/day5/blastocyst, time-code proxy 등 | depth=7, lr=0.045, l2=6, seed=42 | 0.740650 | 0.74239 | 기존 단일 best. ART 과정 기반 feature 정리한 핵심 모델 |
| `literature_gated_extension` | CatBoost | 문헌 기반 frozen/oocyte/PGT/process extension | current all_evidence에 문헌 feature 추가 | 0.740456-0.740584 | - | 문헌 기반 feature 확장은 기존 모델을 넘지 못함 |
| `team_discovered_branch_features` | CatBoost | history trajectory, frozen-thaw, donor/own, DI branch, cause-history | 팀원 설계 기반 branch feature family 비교 | best 0.740660 | - | history 계열만 미세 개선. 나머지는 global 성능 개선 부족 |
| `history_safe_compact_plus_rates` | CatBoost | all_evidence + compact history trajectory + prior rate features | depth=7, lr=0.045, l2=6 | 0.740754 | - | history feature가 기존 currentCB 대비 +0.000104 개선 |
| `history_cb_controlled_blend_040_020` | Ensemble | currentCB + historyCB + LGBM | 0.4 currentCB + 0.4 historyCB + 0.2 LGBM | 0.740923 | 0.74248 | 현재 유지 중인 best 제출 후보 |
| `constrained_oof_blend_optimizer` | Ensemble | 기존 OOF 모델 weight grid | 제한된 OOF-only weight search | 0.740939 | - | base 대비 +0.000016. 개선 폭이 작아 제출 보류 |
| `history_cb_local_tuning` | CatBoost | history_safe_compact_plus_rate feature set 고정 | best: depth=7, lr=0.045, l2=8, random_strength=2 | 0.740797 | - | historyCB 대비 +0.000043. 튜닝 폭 확인 |
| `embedding_mlp_diversity` | Embedding MLP | all_evidence + history feature, categorical embedding + numeric scaling | high dropout MLP | 0.738889 | - | 단일 모델은 낮지만 tree 모델과 다른 ranking source 확인 |
| `mlp_blend_w200` | Ensemble | base history blend + high dropout MLP | 0.8 base blend + 0.2 MLP | 0.741072 | 0.7424699989 | OOF는 +0.000148 개선. Public LB에서 미세하게 낮아 최종 채택하지 않음 |
| `ft_transformer_diversity` | FT-Transformer | all_evidence + history feature, token-based tabular transformer | 진행 중 | - | - | MLP보다 강한 neural diversity source 탐색 중 |

## 최종 제출 후보

최종 제출물은 `candidate_10fold_conservative_avg.csv`입니다.

이 후보는 새로 학습한 10fold CatBoost만 사용하는 방식이 아니라, 기존 5fold CatBoost 계열과 10fold CatBoost 계열을 보수적으로 평균한 제출물입니다.

```text
0.2 * evidence CatBoost 5fold
+ 0.2 * evidence CatBoost 10fold
+ 0.2 * history CatBoost 5fold
+ 0.2 * history CatBoost 10fold
+ 0.2 * LGBM
```

최종 제출 패키지에서는 위 component prediction을 다시 읽어 산식을 검증하고 `/data/submission.csv`를 생성합니다.

## 최종 제출 코드 패키지

- 패키지: `final_submission_minimal_blend_package.zip`
- 실행 파일: `final_submission.py`
- 출력 경로: `/data/submission.csv`
- 검증 결과: 최종 제출 산식 재계산값과 기존 `candidate_10fold_conservative_avg.csv`의 최대 절대 차이 `1.110e-16`
- 실제 저장 파일: 기존 최종 제출물과 바이트 단위 일치 확인
