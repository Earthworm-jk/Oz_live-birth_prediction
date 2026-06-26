# Oz Live Birth Prediction

난임 시술 cycle-level 데이터를 바탕으로 임신 성공 여부를 예측한 프로젝트입니다.

최종 제출물은 `candidate_10fold_conservative_avg`이며, 최종 제출 패키지는 component prediction을 다시 읽어 고정 가중치 산식을 검증한 뒤 `/data/submission.csv`를 생성합니다.

## 최종 제출 코드

최신 제출 패키지:

```text
final_submission_minimal_blend_package.zip
```

패키지 실행 파일:

```text
final_submission.py
```

입출력 경로:

```text
입력 검증 파일: /data/sample_submission.csv
출력 파일    : /data/submission.csv
```

최종 제출 산식:

```text
0.2 * evidence CatBoost 5fold
+ 0.2 * evidence CatBoost 10fold
+ 0.2 * history CatBoost 5fold
+ 0.2 * history CatBoost 10fold
+ 0.2 * LGBM
```

검증 결과:

```text
최종 제출 산식 재계산값과 기존 candidate_10fold_conservative_avg.csv의 최대 절대 차이: 1.110e-16
생성된 /data/submission.csv는 기존 최종 제출 CSV와 바이트 단위 일치 확인
```

## 주요 실험 요약

| 실험명 | 모델 | 핵심 아이디어 | OOF / CV AUC | Public LB | 비고 |
|---|---|---|---:|---:|---|
| `baseline_catboost_all_features` | CatBoost | 기본 ART process feature + 전체 feature | 0.740185 | 0.7417 | 초기 강한 baseline |
| `lgbm_all_features_diversity` | LightGBM | CatBoost와 다른 boosting 구조의 diversity source | 0.739621 | - | 단일 성능은 낮지만 blend에 사용 |
| `evidence_guided_depth7_all_evidence` | CatBoost | age, egg source, SET/day5/blastocyst, time-code proxy | 0.740650 | 0.74239 | ART 과정 기반 핵심 모델 |
| `history_safe_compact_plus_rates` | CatBoost | all_evidence + compact history trajectory + prior rate features | 0.740754 | - | history feature로 개선 |
| `history_cb_controlled_blend_040_020` | Ensemble | 0.4 currentCB + 0.4 historyCB + 0.2 LGBM | 0.740923 | 0.74248 | 기준 제출 blend |
| `catboost_10fold_stability_check_v1` | CatBoost / Ensemble | 10fold CatBoost 안정성 확인 | 0.741031 | - | conservative average 후보 근거 |
| `catboost_cv_stability_submission_variants_v1` | Ensemble | 5fold/10fold CatBoost와 LGBM conservative average | - | 최종 제출 후보 | `candidate_10fold_conservative_avg` 생성 |
| `mlp_blend_w200` | Ensemble | 0.8 base blend + 0.2 high dropout MLP | 0.741072 | 0.7424699989 | OOF 개선은 있었지만 최종 채택하지 않음 |

더 자세한 실험표는 [docs/major_experiment_summary.md](docs/major_experiment_summary.md)에 정리했습니다.

## 최종 제출 패키지 구성

```text
final_submission_minimal_blend_package/
  final_submission.py
  requirements.txt
  README.md
  artifacts/
    candidate_10fold_conservative_avg.csv
    test_component_predictions.csv
    test_pred_cbv2_depth7_all_evidence.csv
    test_pred_cbv2_history_safe_compact_plus_rates.csv
    test_pred_lgbm_a_all_features.csv
  pipeline_src/src/
    features/art_features.py
    models/make_evidence_submission_v1.py
    models/make_history_blend_submission_v1.py
    models/catboost_10fold_stability_check_v1.py
    models/catboost_cv_stability_submission_variants_v1.py
    ...
```

`pipeline_src/src`에는 최종 제출 흐름에 필요한 source file만 포함했습니다.

## 실행 방법

```bash
pip install -r requirements.txt
python final_submission.py
```

심사 환경에서는 `/data/sample_submission.csv`가 있어야 하며, 실행 결과 `/data/submission.csv`가 생성됩니다.

## 개발 환경

```text
OS: Windows 10
Python: 3.13.14
numpy: 2.4.3
pandas: 3.0.1
scikit-learn: 1.8.0
catboost: 1.2.10
lightgbm: 4.6.0
```

## 참고

- 원본 데이터와 중간 학습 산출물은 저장소에 포함하지 않는 것을 원칙으로 했습니다.
- 최종 제출 패키지에는 최종 제출 산식 검증에 필요한 component prediction artifact를 포함했습니다.
- `sample_submission.csv`는 ID 순서 검증과 제출 형식 확인에만 사용합니다.
