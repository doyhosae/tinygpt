# TinyGPT (타이니GPT)

간단한 한국어 텍스트 생성 AI 실험 프로젝트입니다. GPT의 기본 원리를 이해하고 실험해볼 수 있도록 제작되었습니다.

## 🚀 주요 기능

- 한국어 텍스트 생성
- 다양한 학습 모드 지원 (빠른/일반/깊은 학습)
- 대화형 실험 모드
- 모델 평가 도구
- 학습 과정 시각화
- 자동 체크포인트 저장

## 📋 요구사항

```
torch>=1.9.0
matplotlib>=3.4.3
numpy>=1.20.0
```

## 🛠️ 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/tinyGPT.git
cd tinyGPT
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 💻 사용 방법

### 1. 모델 학습

```bash
python tinygpt.py [옵션]
```

주요 옵션:
- `--learning_mode`: 학습 모드 선택 (quick/normal/deep)
- `--block_size`: 문맥 범위 크기 (기본값: 8)
- `--temperature`: 생성 텍스트의 다양성 (기본값: 0.5)
- `--batch_size`: 배치 크기 (기본값: 32)
- `--learning_rate`: 학습률 (기본값: 0.001)
- `--text_file`: 학습 데이터 파일 경로 (기본값: text.txt)
- `--no_interactive`: 대화형 모드 비활성화

예시:
```bash
# 일반 학습 모드로 실행
python tinygpt.py --learning_mode normal --block_size 12

# 빠른 학습 모드로 실행
python tinygpt.py --learning_mode quick --temperature 0.7

# 깊은 학습 모드로 실행
python tinygpt.py --learning_mode deep --batch_size 64
```

### 2. 모델 평가

```bash
python evaluate_models.py [옵션]
```

주요 옵션:
- `--temperature`: 텍스트 생성 온도 (기본값: 0.7)
- `--num_tests`: 각 프롬프트당 테스트 횟수 (기본값: 3)
- `--max_tokens`: 생성할 최대 토큰 수 (기본값: 30)

## 📊 결과 확인

학습 결과는 `results` 디렉토리에 저장됩니다:
- `best_model.pt`: 최고 성능 모델
- `final_model.pt`: 최종 학습된 모델
- `learning_curve.png`: 학습 곡선 그래프
- `config.json`: 실험 설정 정보

## 🔍 모델 평가 결과

평가 결과는 다음 항목들을 포함합니다:
- 일관성 점수
- 자연스러움 점수
- 문맥 유지 점수
- 다양성 점수
- 모델 비교 그래프

## 📝 학습 데이터 준비

1. `text.txt` 파일에 학습하고 싶은 한국어 텍스트를 저장합니다.
2. 텍스트는 문장 단위로 구성하며, 각 문장은 마침표(.)로 끝나야 합니다.
3. 데이터가 많을수록 더 좋은 결과를 얻을 수 있습니다.

## ⚙️ 하이퍼파라미터 조정

더 나은 결과를 위한 권장 설정:
- `block_size`: 8-16 사이 권장
- `temperature`: 0.5-1.0 사이 권장
- `batch_size`: 32-128 사이 권장
- `learning_rate`: 0.001-0.0001 사이 권장

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

## 📜 라이선스

MIT License