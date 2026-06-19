# Oz Live Birth Prediction

난임 환자의 임신 성공 여부를 예측하는 해커톤 프로젝트입니다.

## 폴더 구조

```text
ivf/
├─ data/
│  ├─ raw/          # 원본 데이터
│  └─ processed/    # 전처리한 데이터
├─ notebooks/       # EDA, 실험용 노트북
├─ src/             # 반복해서 사용할 Python 코드
├─ models/          # 학습한 모델 파일
├─ submissions/     # 제출 파일
├─ README.md
├─ requirements.txt
└─ .gitignore
```

## 시작 방법

1. `data/raw/`에 제공받은 원본 데이터를 둡니다.
2. 처음 분석은 `notebooks/`에서 진행합니다.
3. 반복해서 쓰는 코드는 필요할 때 `src/`에 옮깁니다.
4. 제출 파일은 `submissions/`에 저장합니다.

## 공유 규칙

- GitHub에는 코드와 문서 위주로 올립니다.
- 원본 데이터, 전처리 데이터, 모델 파일은 올리지 않습니다.
- `sample_submission.csv`는 제출 형식 확인용으로만 둡니다.
