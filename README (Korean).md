🚀 AI 학습 코치 포트폴리오 앱

이 프로젝트는 개인화된 학습 경험을 제공하는 AI 코치 애플리케이션입니다. RAG(Retrieval-Augmented Generation) 기술과 LSTM 예측 모델을 결합하여 사용자 맞춤형 학습을 지원합니다.

주요 기능

RAG 지식 챗봇: 사용자가 업로드한 PDF, TXT, HTML 문서 기반으로 질문에 답변합니다.

맞춤형 콘텐츠 생성: 특정 주제와 난이도에 맞춰 요약, 퀴즈, 실습 예제 아이디어를 Gemini LLM을 통해 생성합니다.

LSTM 성취도 예측: 가상의 과거 학습 데이터를 바탕으로 미래 학습 성취도를 예측하여 동기 부여를 제공합니다.

기술 스택

Frontend/App Framework: Streamlit

Backend/ML/LLM: Google Gemini API, LangChain

Vector Database: FAISS (Firestore에 인덱스 저장)

Persistent Storage: Google Cloud Firestore (Admin SDK를 통한 영구 데이터 관리)

Deployment: Streamlit Cloud

🔑 배포를 위한 설정 (Secrets)

이 앱은 Streamlit Cloud 배포 시 다음 세 가지 Secrets 키를 필요로 합니다. (민감 정보는 이 저장소에 포함되어 있지 않습니다.)

GEMINI_API_KEY: Gemini API 호출을 위한 개인 키.

FIREBASE_SERVICE_ACCOUNT_JSON: Firestore Admin SDK 접근을 위한 서비스 계정 JSON 문자열 전체.
