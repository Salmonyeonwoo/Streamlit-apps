# ========================================
# Streamlit AI 학습 코치 (최종 Firebase 영구 저장소 통합 및 시뮬레이터 확장)
# ========================================
import streamlit as st
import os
import tempfile
import time
import json
import re
import base64
import io
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta # 날짜/시간 처리를 위해 추가

# ⭐ Admin SDK 관련 라이브러리 임포트
from firebase_admin import credentials, firestore, initialize_app, get_app
# Admin SDK의 firestore와 Google Cloud SDK의 firestore를 구분하기 위해 alias 사용
from google.cloud import firestore as gcp_firestore
from google.cloud.firestore import Query # Firestore 쿼리용 import 추가

# ConversationChain 사용을 위해 import 추가
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate # ⭐ PromptTemplate 임포트


# ================================
# 1. Firebase Admin SDK 초기화 및 Secrets 처리 함수
# ================================

def _get_admin_credentials():
    """Secrets에서 서비스 계정 정보를 안전하게 로드하고 딕셔너리로 반환합니다."""
    if "FIREBASE_SERVICE_ACCOUNT_JSON" not in st.secrets:
        return None, "FIREBASE_SERVICE_ACCOUNT_JSON Secret이 누락되었습니다."
    
    service_account_data = st.secrets["FIREBASE_SERVICE_ACCOUNT_JSON"]
    sa_info = None

    if isinstance(service_account_data, str):
        try:
            sa_info = json.loads(service_account_data.strip())
        except json.JSONDecodeError as e:
            return None, f"FIREBASE_SERVICE_ACCOUNT_JSON의 JSON 구문 오류입니다. 값을 확인하세요. 상세 오류: {e}"
    elif hasattr(service_account_data, 'get'):
        try:
            sa_info = dict(service_account_data) # AttrDict를 표준 dict로 변환
        except Exception:
             return None, f"FIREBASE_SERVICE_ACCOUNT_JSON의 딕셔너리 변환 실패. 타입: {type(service_account_data)}"
    else:
        return None, f"FIREBASE_SERVICE_ACCOUNT_JSON의 형식이 올바르지 않습니다. (Type: {type(service_account_data)})"
    
    if not sa_info.get("project_id") or not sa_info.get("private_key"):
        return None, "JSON 내 'project_id' 또는 'private_key' 필드가 누락되었습니다."

    return sa_info, None

@st.cache_resource(ttl=None)
def initialize_firestore_admin():
    """Secrets에서 로드된 정보를 사용하여 Firebase Admin SDK를 초기화합니다."""
    sa_info, error_message = _get_admin_credentials()

    if error_message:
        st.error(f"❌ Firebase Secret 오류: {error_message}")
        return None

    try:
        get_app()
    except ValueError:
        pass 
    else:
        try:
            return firestore.client()
        except Exception as e:
            st.error(f"🔥 Firebase 클라이언트 로드 실패: {e}")
            return None

    try:
        cred = credentials.Certificate(sa_info) 
        initialize_app(cred)
        
        db_client = firestore.client()
        st.session_state["db"] = db_client
        st.success("✅ Firebase Admin SDK 초기화 완료! (Secrets 기반)")
        return db_client
    except Exception as e:
        st.error(f"🔥 Firebase 초기화 실패: 서비스 계정 정보 문제. 오류: {e}")
        return None


def save_index_to_firestore(db, vector_store, index_id="user_portfolio_rag"):
    """FAISS 인덱스를 Firestore에 Base64 형태로 직렬화하여 저장합니다."""
    if not db: return False
    temp_dir = tempfile.mkdtemp()
    
    try:
        vector_store.save_local(folder_path=temp_dir, index_name="index")
        
        with open(f"{temp_dir}/index.faiss", "rb") as f: faiss_bytes = f.read()
        with open(f"{temp_dir}/index.pkl", "rb") as f: metadata_bytes = f.read()
        
        encoded_data = {
            "faiss_data": base64.b64encode(faiss_bytes).decode('utf-8'),
            "metadata_data": base64.b64encode(metadata_bytes).decode('utf-8'),
            "timestamp": gcp_firestore.SERVER_TIMESTAMP 
        }
        
        db.collection("rag_indices").document(index_id).set(encoded_data)
        return True
    
    except Exception as e:
        st.error(f"DB 저장 시도 중 오류 발생: {e}")
        print(f"Error saving index to Firestore: {e}")
        return False

def load_index_from_firestore(db, embeddings, index_id="user_portfolio_rag"):
    """Firestore에서 Base64 문자열을 로드하여 FAISS 인덱스로 역직렬화합니다."""
    if not db: return False

    try:
        doc = db.collection("rag_indices").document(index_id).get()
        if not doc.exists:
            return None 

        encoded_data = doc.to_dict()
        
        faiss_bytes = base64.b64decode(encoded_data["faiss_data"])
        metadata_bytes = base64.b64decode(encoded_data["metadata_data"])
        
        temp_dir = tempfile.mkdtemp()
        with open(f"{temp_dir}/index.faiss", "wb") as f: f.write(faiss_bytes)
        with open(f"{temp_dir}/index.pkl", "wb") as f: f.write(metadata_bytes)
        
        vector_store = FAISS.load_local(folder_path=temp_dir, embeddings=embeddings, index_name="index")
        return vector_store
        
    except Exception as e:
        print(f"Error loading index from Firestore: {e}")
        return None

# ⭐ 상담 이력 저장 함수 추가
def save_simulation_history(db, initial_query, customer_type, messages):
    """Firestore에 상담 이력을 저장합니다."""
    if not db: 
        st.sidebar.warning("❌ DB 연결 실패: 상담 이력 저장 불가")
        return False
    
    # 메시지 리스트를 JSON 직렬화 가능한 형태로 변환
    history_data = [{k: v for k, v in msg.items()} for msg in messages]

    data = {
        "initial_query": initial_query,
        "customer_type": customer_type,
        "messages": history_data,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    
    try:
        db.collection("simulation_histories").add(data)
        st.sidebar.success("✅ 상담 이력이 저장되었습니다.")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ 상담 이력 저장 실패: {e}")
        return False

# ⭐ 상담 이력 로드 함수 추가
def load_simulation_histories(db):
    """Firestore에서 최근 상담 이력을 로드합니다 (최대 10개)."""
    if not db: return []
    
    try:
        # 최근 10개 이력을 시간 순으로 정렬하여 가져옴
        histories = (
            db.collection("simulation_histories")
            .order_by("timestamp", direction=Query.DESCENDING)
            .limit(10)
            .stream()
        )
        
        results = []
        for doc in histories:
            data = doc.to_dict()
            data['id'] = doc.id
            
            # 메시지 데이터가 직렬화된 리스트인지 확인
            if 'messages' in data and isinstance(data['messages'], list) and data['messages']:
                results.append(data)

        return results
    except Exception as e:
        st.error(f"❌ 이력 로드 실패: {e}")
        return []

# ================================
# 2. JSON/RAG/LSTM/TTS 함수 정의
# ================================
def clean_and_load_json(text):
    """LLM 응답 텍스트에서 JSON 객체만 정규표현식으로 추출하여 로드"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def synthesize_and_play_audio(current_lang_key):
    """TTS API 대신 Web Speech API를 위한 JS 유틸리티를 Streamlit에 삽입합니다."""
    
    # 템플릿 리터럴 내부에서 L 딕셔너리를 직접 참조할 수 없으므로, 하드코딩된 값 사용
    ko_ready = "음성으로 듣기 준비됨"
    en_ready = "Ready to listen"
    ja_ready = "音声再生の準備ができました"

    tts_js_code = f"""
    <script>
    if (!window.speechSynthesis) {{
        document.getElementById('tts_status').innerText = '❌ TTS Not Supported';
    }}

    window.speakText = function(text, langKey) {{
        if (!window.speechSynthesis || !text) return;

        const statusElement = document.getElementById('tts_status');
        const utterance = new SpeechSynthesisUtterance(text);
        
        // 동적으로 언어 코드 설정
        const langCode = {{ "ko": "ko-KR", "en": "en-US", "ja": "ja-JP" }}[langKey] || "en-US";
        utterance.lang = langCode; 

        // 동적으로 준비 상태 메시지 설정 (L 딕셔너리 값을 직접 사용)
        const getReadyText = (key) => {{
            if (key === 'ko') return '{ko_ready}';
            if (key === 'en') return '{en_ready}';
            if (key === 'ja') return '{ja_ready}';
            return '{en_ready}';
        }};

        let voicesLoaded = false;
        const setVoiceAndSpeak = () => {{
            const voices = window.speechSynthesis.getVoices();
            if (voices.length > 0) {{
                // 현재 언어 코드와 일치하는 음성을 찾거나, 첫 번째 음성을 사용
                utterance.voice = voices.find(v => v.lang.startsWith(langCode.substring(0, 2))) || voices[0];
                voicesLoaded = true;
                window.speechSynthesis.speak(utterance);
            }} else if (!voicesLoaded) {{
                // 음성이 아직 로드되지 않은 경우, 잠시 후 재시도 (비동기 로드 문제 해결)
                setTimeout(setVoiceAndSpeak, 100);
            }}
        }};
        
        // 이벤트 핸들러 설정
        utterance.onstart = () => {{
            statusElement.innerText = '{LANG[current_lang_key].get("tts_status_generating", "오디오 생성 중...")}';
            statusElement.style.backgroundColor = '#fff3e0';
        }};
        
        utterance.onend = () => {{
            statusElement.innerText = '{LANG[current_lang_key].get("tts_status_success", "✅ 오디오 재생 완료!")}';
            statusElement.style.backgroundColor = '#e8f5e9';
             setTimeout(() => {{ 
                 statusElement.innerText = getReadyText(langKey);
                 statusElement.style.backgroundColor = '#f0f0f0';
             }}, 3000);
        }};
        
        utterance.onerror = (event) => {{
            statusElement.innerText = '{LANG[current_lang_key].get("tts_status_error", "❌ TTS 오류 발생")}';
            statusElement.style.backgroundColor = '#ffebee';
            console.error("SpeechSynthesis Error:", event);
             setTimeout(() => {{ 
                 statusElement.innerText = getReadyText(langKey);
                 statusElement.style.backgroundColor = '#f0f0f0';
             }}, 3999);
        }};

        window.speechSynthesis.cancel(); // Stop any current speech
        setVoiceAndSpeak(); // 재생 시작

    }};
    </script>
    """
    # JS 유틸리티를 Streamlit 앱에 컴포넌트로 삽입 (높이 조정하여 상태창만 보이도록)
    st.components.v1.html(tts_js_code, height=5, width=0)

def render_tts_button(text_to_speak, current_lang_key):
    """TTS 버튼 UI를 렌더링하고 클릭 시 JS 함수를 호출합니다."""
    
    # 줄 바꿈을 공백으로 변환하고, 따옴표를 이스케이프 처리
    safe_text = text_to_speak.replace('\n', ' ').replace('"', '\\"').replace("'", "\\'")
    
    # ⭐ JS 함수에 언어 키도 함께 전달
    js_call = f"window.speakText('{safe_text}', '{current_lang_key}')"

    st.markdown(f"""
        <button onclick="{js_call}"
                style="background-color: #4338CA; color: white; padding: 10px 20px; border-radius: 5px; cursor: pointer; border: none; width: 100%; font-weight: bold; margin-bottom: 10px;">
            {LANG[current_lang_key].get("button_listen_audio", "음성으로 듣기")} 🎧
        </button>
    """, unsafe_allow_html=True)


def get_mock_response_data(lang_key, customer_type):
    """API Key가 없을 때 사용할 가상 응대 데이터 (다국어 지원)"""
    
    L = LANG[lang_key]
    
    if lang_key == 'ko':
        # ⭐ 수정된 중립적인 목업 데이터 템플릿
        initial_check = "고객님의 성함, 전화번호, 이메일 등 정확한 연락처 정보를 확인해 주시면 감사하겠습니다."
        tone = "공감 및 해결 중심"
        advice = "이 고객은 {customer_type} 성향이지만, 문제 해결을 간절히 원합니다. 공감과 함께, 문제 해결에 필수적인 정보를 명확하게 요청해야 합니다. 불필요한 사족을 피하고 신뢰를 주도록 하세요."
        draft = f"""
{initial_check}

> 고객님, 불편을 겪게 해드려 죄송합니다. 고객님의 상황을 충분히 이해하고 있습니다.
> 문제 해결을 위해, 아래 세 가지 필수 정보를 확인해 주시면 감사하겠습니다. 이 정보가 있어야 고객님 상황에 맞는 정확한 해결책을 제시할 수 있습니다.
> 1. 문제 발생과 관련된 상품/서비스의 **정확한 명칭 및 예약 번호** (예: 파리 eSIM, 예약번호 1234567)
> 2. 현재 **문제 상황**에 대한 구체적인 설명 (예: 휴대폰이 안 됨, 환불 요청, 정보 문의)
> 3. 이미 **시도하신 해결 단계** (예: 기기 재부팅, 설정 확인 등)

> 고객님과의 원활한 소통을 통해 신속하게 문제 해결을 돕겠습니다. 답변 기다리겠습니다.
"""
    elif lang_key == 'en':
        initial_check = "Could you please confirm your accurate contact details, such as your full name, phone number, and email address?"
        tone = "Empathy and Solution-Focused"
        advice = "This customer is {customer_type} but desperately wants a solution. Show empathy, but clearly request the essential information needed for troubleshooting. Be direct and build trust."
        draft = f"""
{initial_check}

> Dear Customer, I sincerely apologize for the inconvenience you are facing. I completely understand your frustration.
> To proceed with troubleshooting, please confirm the three essential pieces of information below. This data is critical for providing you with the correct, tailored solution:
> 1. The **exact name and booking number** of the product/service concerned (e.g., Paris eSIM, Booking #1234567).
> 2. A specific description of the **current issue** (e.g., phone not connecting, refund request, information inquiry).
> 3. Any **troubleshooting steps already attempted** (e.g., device rebooted, settings checked, etc.).

> We aim to resolve your issue as quickly as possible with your cooperation. We await your response.
"""
    elif lang_key == 'ja':
        initial_check = "お客様の氏名、お電話番号、Eメールアドレスなど、正確な連絡先情報を確認させていただけますでしょうか。"
        tone = "共感と解決中心"
        advice = "このお客様は{customer_type}傾向ですが、問題の解決を強く望んでいます。共感を示しつつも、問題解決に不可欠な情報を明確に尋ねる必要があります。冗長な説明を避け、信頼感を与える対応を心がけてください。"
        draft = f"""
{initial_check}

> お客様、ご不便をおかけし、誠に申し訳ございません。現在の状況、十分承知いたしました。
> 問題を迅速に解決するため、恐れ入りますが、以下の3点の必須情報についてご確認いただけますでしょうか。この情報がないと、お客様の状況に合わせた的確な解決策をご案内できません。
> 1. 問題の対象となる**商品・サービスの正確な名称と予約番号** (例: パリeSIM、予約番号1234567)
> 2. 現在の**具体的な問題状況** (例: 携帯電話が使えない、返金を希望する、情報が知りたい)
> 3. 既に**お試しいただいた解決手順** (例: 端末の再起動、設定確認など)

> お客様との円滑なコミュニケーションを通じて、迅速に問題解決をサポートさせていただきます。ご返信をお待ちしております。
"""
    
    # advice 문자열 내부의 {customer_type}을 실제 선택 값으로 대체
    advice_text = advice.replace("{customer_type}", customer_type)

    return {
        "advice_header": f"{L['simulation_advice_header']}",
        "advice": advice_text,
        "draft_header": f"{L['simulation_draft_header']} ({tone})",
        "draft": draft
    }

def get_closing_messages(lang_key):
    """고객 응대 종료 시 사용하는 다국어 메시지 딕셔너리를 반환합니다."""
    
    if lang_key == 'ko':
        return {
            "additional_query": "또 다른 문의 사항은 없으신가요?",
            "chat_closing": "고객님의 추가 문의 사항이 없어, 이 상담 채팅을 종료하겠습니다. 고객 문의 센터에 연락 주셔서 감사드리며, 추가로 저희 응대 솔루션에 대한 설문 조사에 응해 주시면 감사하겠습니다. 추가 문의 사항이 있으시면 언제든지 연락 주십시오."
        }
    elif lang_key == 'en':
        return {
            "additional_query": "Is there anything else we can assist you with today?",
            "chat_closing": "As there are no further inquiries, we will now end this chat session. Thank you for contacting our Customer Support Center. We would be grateful if you could participate in a short survey about our service solution. Please feel free to contact us anytime if you have any additional questions."
        }
    elif lang_key == 'ja':
        return {
            "additional_query": "また、お客様にお手伝いさせて頂けるお問い合わせは御座いませんか？",
            "chat_closing": "お客様からの追加のお問い合わせがないため、本チャットサポートを終了させていただきます。お問い合わせいただき、誠にありがとうございました。弊社の対応ソリューションに関する簡単なアンケートにご協力いただければ幸いです。追加のご質問がございましたらいつでもご連絡ください。"
        }
    return get_closing_messages('ko') # 기본값


def get_document_chunks(files):
    """업로드된 파일에서 텍스트를 로드하고 청킹합니다."""
    documents = []
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in files:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == "pdf":
            with open(temp_filepath, "wb") as f: f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            documents.extend(loader.load())
        elif file_extension == "html":
            raw_html = uploaded_file.getvalue().decode('utf-8')
            soup = BeautifulSoup(raw_html, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)
            documents.append(Document(page_content=text_content, metadata={"source": uploaded_file.name}))
        elif file_extension == "txt":
            with open(temp_filepath, "wb") as f: f.write(uploaded_file.getvalue())
            loader = TextLoader(temp_filepath, encoding="utf-8")
            documents.extend(loader.load())
        else:
            print(f"File '{uploaded_file.name}' not supported.")
            continue
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def get_vector_store(text_chunks):
    """텍스트 청크를 임베딩하고 Vector Store를 생성합니다."""
    cache_key = tuple(doc.page_content for doc in text_chunks)
    if cache_key in st.session_state.embedding_cache: return st.session_state.embedding_cache[cache_key]
    if not st.session_state.is_llm_ready: return None
    try:
        vector_store = FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)
        st.session_state.embedding_cache[cache_key] = vector_store
        return vector_store
    except Exception as e:
        if "429" in str(e): return None
        else:
            print(f"Vector Store creation failed: {e}") 
            return None

def get_rag_chain(vector_store):
    """검색 체인(ConversationalRetrievalChain)을 생성합니다."""
    if vector_store is None: return None
    # ⭐ RAG 체인에 memory_key를 명시적으로 전달
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )

@st.cache_resource
def load_or_train_lstm():
    """가상의 학습 성취도 예측을 위한 LSTM 모델을 생성하고 학습합니다."""
    np.random.seed(int(time.time())) # ⭐ LSTM 결과를 랜덤화하기 위해 시드에 현재 시간을 사용
    data = np.cumsum(np.random.normal(loc=5, scale=5, size=50)) + 60
    data = np.clip(data, 50, 95)
    def create_dataset(dataset, look_back=3):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back)])
            Y.append(dataset[i + look_back])
        return np.array(X), np.array(Y)
    look_back = 5
    X, Y = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=10, batch_size=1, verbose=0)
    return model, data

def force_rerun_lstm():
    """캐시된 LSTM 모델을 무효화하고 새로 실행합니다."""
    # st.cache_resource 함수의 캐시를 직접 지울 수 없으므로, 
    # Streamlit의 재실행 메커니즘을 사용하여 load_or_train_lstm이
    # time.time() 시드로 새 결과를 생성하도록 유도합니다.
    st.session_state.lstm_rerun_trigger = time.time()
    st.rerun()


def clean_and_load_json(text):
    """LLM 응답 텍스트에서 JSON 객체만 정규표현식으로 추출하여 로드"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def render_interactive_quiz(quiz_data, current_lang):
    """생성된 퀴즈 데이터를 Streamlit UI로 렌더링하고 피드백을 제공합니다."""
    L = LANG[current_lang]
    if not quiz_data or 'quiz_questions' not in quiz_data: return

    questions = quiz_data['quiz_questions']
    num_questions = len(questions)

    if "current_question" not in st.session_state or st.session_state.current_question >= num_questions:
        st.session_state.current_question = 0
        st.session_state.quiz_results = [None] * num_questions
        st.session_state.quiz_submitted = False
        
    q_index = st.session_state.current_question
    q_data = questions[q_index]
    
    st.subheader(f"{q_index + 1}. {q_data['question']}")
    
    options_dict = {}
    try:
        options_dict = {f"{opt['option']}": f"{opt['option']}) {opt['text']}" for opt in q_data['options']}
    except KeyError:
        st.error(L["quiz_fail_structure"])
        if 'quiz_data_raw' in st.session_state: st.code(st.session_state.quiz_data_raw, language="json")
        return

    options_list = list(options_dict.values())
    
    selected_answer = st.radio(
        L.get("select_answer", "정답을 선택하세요"),
        options=options_list,
        key=f"q_radio_{q_index}"
    )

    col1, col2 = st.columns(2)

    if col1.button(L.get("check_answer", "정답 확인"), key=f"check_btn_{q_index}", disabled=st.session_state.quiz_submitted):
        user_choice_letter = selected_answer.split(')')[0] if selected_answer else None
        correct_answer_letter = q_data['correct_answer']

        is_correct = (user_choice_letter == correct_answer_letter)
        
        st.session_state.quiz_results[q_index] = is_correct
        st.session_state.quiz_submitted = True
        
        if is_correct:
            st.success(L.get("correct_answer", "정답입니다! 🎉"))
        else:
            st.error(L.get("incorrect_answer", "오답입니다.😞"))
        
        st.markdown(f"**{L.get('correct_is', '정답')}: {correct_answer_letter}**")
        st.info(f"**{L.get('explanation', '해설')}:** {q_data['explanation']}")

    if st.session_state.quiz_submitted:
        if q_index < num_questions - 1:
            if col2.button(L.get("next_question", "다음 문항"), key=f"next_btn_{q_index}"):
                st.session_state.current_question += 1
                st.session_state.quiz_submitted = False
                st.rerun()
        else:
            total_correct = st.session_state.quiz_results.count(True)
            total_questions = len(st.session_state.quiz_results)
            st.success(f"**{L.get('quiz_complete', '퀴즈 완료!')}** {L.get('score', '점수')}: {total_correct}/{total_questions}")
            if st.button(L.get("retake_quiz", "퀴즈 다시 풀기"), key="retake"):
                st.session_state.current_question = 0
                st.session_state.quiz_results = [None] * num_questions
                st.session_state.quiz_submitted = False
                st.rerun()

# ================================
# 3. 다국어 지원 딕셔너리 (Language Dictionary)
# ================================
LANG = {
    "ko": {
        "title": "개인 맞춤형 AI 학습 코치",
        "sidebar_title": "📚 AI Study Coach 설정",
        "file_uploader": "학습 자료 업로드 (PDF, TXT, HTML)",
        "button_start_analysis": "자료 분석 시작 (RAG Indexing)",
        "rag_tab": "RAG 지식 챗봇",
        "content_tab": "맞춤형 학습 콘텐츠 생성",
        "lstm_tab": "LSTM 성취도 예측 대시보드",
        "simulator_tab": "AI 고객 응대 시뮬레이터", 
        "rag_header": "RAG 지식 챗봇 (문서 기반 Q&A)",
        "rag_desc": "업로드된 문서 기반으로 질문에 답변합니다。",
        "rag_input_placeholder": "학습 자료에 대해 질문해 보세요",
        "llm_error_key": "⚠️ 경고: GEMINI API 키가 설정되지 않았습니다. Streamlit Secrets에 'GEMINI_API_KEY'를 설정해주세요。",
        "llm_error_init": "LLM 초기화 오류: API 키를 확인해 주세요。",
        "content_header": "맞춤형 학습 콘텐츠 생성",
        "content_desc": "학습 주제와 난이도에 맞춰 콘텐츠 생성",
        "topic_label": "학습 주제",
        "level_label": "난이도",
        "content_type_label": "콘텐츠 형식",
        "level_options": ["초급", "중급", "고급"],
        "content_options": ["핵심 요약 노트", "객관식 퀴즈 10문항", "실습 예제 아이디어"],
        "button_generate": "콘텐츠 생성",
        "warning_topic": "학습 주제를 입력해 주세요。",
        "lstm_header": "LSTM 기반 학습 성취도 예측 대시보드",
        "lstm_desc": "가상의 과거 퀴즈 점수 데이터를 바탕으로 LSTM 모델을 훈련하고 미래 성취도를 예측하여 보여줍니다。",
        "lstm_disabled_error": "The LSTM feature is temporarily disabled due to build environment issues. Please use the 'Custom Content Generation' feature first.",
        "lang_select": "언어 선택",
        "embed_success": "총 {count}개 청크로 학습 DB 구축 완료!",
        "embed_fail": "임베딩 실패: 무료 티어 한도 초과 또는 네트워크 문제。",
        "warning_no_files": "먼저 학습 자료를 업로드하세요。",
        "warning_rag_not_ready": "RAG가 준비되지 않았습니다. 학습 자료를 업로드하고 분석하세요。",
        "quiz_fail_structure": "퀴즈 데이터 구조가 올바르지 않습니다。",
        "select_answer": "정답을 선택하세요",
        "check_answer": "정답 확인",
        "next_question": "다음 문항",
        "correct_answer": "정답입니다! 🎉",
        "incorrect_answer": "오답입니다. 😞",
        "correct_is": "정답",
        "explanation": "해설",
        "quiz_complete": "퀴즈 완료!",
        "score": "점수",
        "retake_quiz": "퀴즈 다시 풀기",
        "quiz_error_llm": "퀴즈 생성 실패: LLM이 올바른 JSON 형식을 반환하지 않았습니다. LLM 응답 원본을 확인하세요。",
        "quiz_original_response": "LLM 원본 응답",
        "firestore_loading": "데이터베이스에서 RAG 인덱스 로드 중...",
        "firestore_no_index": "데이터베이스에서 기존 RAG 인덱스를 찾을 수 없습니다. 파일을 업로드하여 새로 만드세요。", 
        "db_save_complete": "(DB 저장 완료)", # ⭐ 다국어 키 추가
        "data_analysis_progress": "자료 분석 및 학습 DB 구축 중...", # ⭐ 다국어 키 추가
        "response_generating": "답변 생성 중...", # ⭐ 다국어 키 추가
        "lstm_result_header": "학습 성취도 예측 결과",
        "lstm_score_metric": "현재 예측 성취도",
        "lstm_score_info": "다음 퀴즈 예상 점수는 약 **{predicted_score:.1f}점**입니다. 학습 성과를 유지하거나 개선하세요!",
        "lstm_rerun_button": "새로운 가상 데이터로 예측",
        
        # ⭐ 시뮬레이터 관련 텍스트
        "simulator_header": "AI 고객 응대 시뮬레이터",
        "simulator_desc": "까다로운 고객 문의에 대해 AI의 응대 초안 및 가이드라인을 제공합니다.",
        "customer_query_label": "고객 문의 내용 (링크 포함 가능)",
        "customer_type_label": "고객 성향",
        "customer_type_options": ["일반적인 문의", "까다로운 고객", "매우 불만족스러운 고객"],
        "button_simulate": "응대 조언 요청",
        "simulation_warning_query": "고객 문의 내용을 입력해주세요。",
        "simulation_no_key_warning": "⚠️ API Key가 없는 경우, 응답 생성은 실행되지 않습니다. (UI 구성은 완료되었습니다.)",
        "simulation_advice_ready": "AI의 응대 조언이 준비되었습니다!",
        "simulation_advice_header": "AI의 응대 가이드라인",
        "simulation_draft_header": "추천 응대 초안",
        "button_listen_audio": "음성으로 듣기",
        "tts_status_ready": "음성으로 듣기 준비됨",
        "tts_status_generating": "오디오 생성 중...",
        "tts_status_success": "✅ 오디오 재생 완료!",
        "tts_status_fail": "❌ TTS 생성 실패 (데이터 없음)",
        "tts_status_error": "❌ TTS 오류 발생",
        "history_expander_title": "📝 이전 상담 이력 로드 (최근 10개)", 
        "initial_query_sample": "프랑스 파리에 도착했는데, 클룩에서 구매한 eSIM이 활성화가 안 됩니다. 연결이 안 돼서 너무 곤란합니다. 어떻게 해야 하나요?", 
        
        # ⭐ 대화형/종료 메시지
        "button_mic_input": "음성 입력",
        "prompt_customer_end": "고객님의 추가 문의 사항이 없어, 이 상담 채팅을 종료하겠습니다.",
        "prompt_survey": "고객 문의 센터에 연락 주셔서 감사드리며, 추가로 저희 응대 솔루션에 대한 설문 조사에 응해 주시면 감사하겠습니다. 추가 문의 사항이 있으시면 언제든지 연락 주십시오。",
        "customer_closing_confirm": "또 다른 문의 사항은 없으신가요?",
        "customer_positive_response": "좋은 말씀/친절한 상담 감사드립니다。",
        "button_end_chat": "응대 종료 (설문 조사 요청)",
        "agent_response_header": "✍️ 에이전트 응답",
        "agent_response_placeholder": "고객에게 응답하세요 (고객의 필수 정보를 요청/확인하거나, 문제 해결책을 제시하세요)",
        "send_response_button": "응답 전송",
        "request_rebuttal_button": "고객의 다음 반응 요청",
        "new_simulation_button": "새 시뮬레이션 시작",
        "history_selectbox_label": "로드할 이력을 선택하세요:",
        "history_load_button": "선택된 이력 로드",
        "delete_history_button": "❌ 모든 이력 삭제", 
        "delete_confirm_message": "정말로 모든 상담 이력을 삭제하시겠습니까? 되돌릴 수 없습니다.", 
        "delete_confirm_yes": "예, 삭제합니다", 
        "delete_confirm_no": "아니오, 유지합니다", 
        "delete_success": "✅ 모든 상담 이력 삭제 완료!",
        "deleting_history_progress": "이력 삭제 중...", 
        "search_history_label": "이력 키워드 검색", 
        "date_range_label": "날짜 범위 필터", 
        "no_history_found": "검색 조건에 맞는 이력이 없습니다." 
    },
    "en": {
        "title": "Personalized AI Study Coach",
        "sidebar_title": "📚 AI Study Coach Settings",
        "file_uploader": "Upload Study Materials (PDF, TXT, HTML)",
        "button_start_analysis": "Start Analysis (RAG Indexing)",
        "rag_tab": "RAG Knowledge Chatbot",
        "content_tab": "Custom Content Generation",
        "lstm_tab": "LSTM Achievement Prediction",
        "simulator_tab": "AI Customer Response Simulator", 
        "rag_header": "RAG Knowledge Chatbot (Document Q&A)",
        "rag_desc": "Answers questions based on the uploaded documents.",
        "rag_input_placeholder": "Ask a question about your study materials",
        "llm_error_key": "⚠️ Warning: GEMINI API Key is not set. Please set 'GEMINI_API_KEY' in Streamlit Secrets.",
        "llm_error_init": "LLM initialization error: Please check your API key.",
        "content_header": "Custom Learning Content Generation",
        "content_desc": "Generate content tailored to your topic and difficulty.",
        "topic_label": "Learning Topic",
        "level_label": "Difficulty",
        "content_type_label": "Content Type",
        "level_options": ["Beginner", "Intermediate", "Advanced"],
        "content_options": ["Key Summary Note", "10 Multiple-Choice Questions", "Practical Example Idea"],
        "button_generate": "Generate Content",
        "warning_topic": "Please enter a learning topic.",
        "lstm_header": "LSTM Based Achievement Prediction",
        "lstm_desc": "Trains an LSTM model on hypothetical past quiz scores to predict future achievement.",
        "lstm_disabled_error": "The LSTM feature is temporarily disabled due to build environment issues. Please use the 'Custom Content Generation' feature first.",
        "lang_select": "Select Language",
        "embed_success": "Learning DB built with {count} chunks!",
        "embed_fail": "Embedding failed: Free tier quota exceeded or network issue.",
        "warning_no_files": "Please upload study materials first.",
        "warning_rag_not_ready": "RAG is not ready. Upload materials and click Start Analysis.",
        "quiz_fail_structure": "Quiz data structure is incorrect.",
        "select_answer": "Select answer",
        "check_answer": "Confirm answer",
        "next_question": "Next Question",
        "correct_answer": "Correct! 🎉",
        "incorrect_answer": "Incorrect. 😞",
        "correct_is": "Correct answer",
        "explanation": "Explanation",
        "quiz_complete": "Quiz completed!",
        "score": "Score",
        "retake_quiz": "Retake Quiz",
        "quiz_error_llm": "Quiz generation failed: LLM did not return a valid JSON format. Check the original LLM response.",
        "quiz_original_response": "Original LLM Response",
        "firestore_loading": "Loading RAG index from database...",
        "firestore_no_index": "Could not find existing RAG index in database. Please upload files and create a new one.", 
        "db_save_complete": "(DB Save Complete)", 
        "data_analysis_progress": "Analyzing materials and building learning DB...", 
        "response_generating": "Generating response...", 
        "lstm_result_header": "Prediction Results",
        "lstm_score_metric": "Current Predicted Achievement",
        "lstm_score_info": "Your next estimated quiz score is **{predicted_score:.1f}**. Maintain or improve your learning progress!",
        "lstm_rerun_button": "Predict with New Hypothetical Data",

        # ⭐ 시뮬레이터 관련 텍스트
        "simulator_header": "AI Customer Response Simulator",
        "simulator_desc": "Provides AI-generated response drafts and guidelines for handling challenging customer inquiries.",
        "customer_query_label": "Customer Query (Link optional)",
        "customer_type_label": "Customer Sentiment",
        "customer_type_options": ["General Inquiry", "Challenging Customer", "Highly Dissatisfied Customer"],
        "button_simulate": "Request Response Advice",
        "simulation_warning_query": "Please enter the customer's query.",
        "simulation_no_key_warning": "⚠️ API Key is missing. Response generation cannot proceed. (UI configuration is complete.)",
        "simulation_advice_ready": "AI's response advice is ready!",
        "simulation_advice_header": "AI Response Guidelines",
        "simulation_draft_header": "Recommended Response Draft",
        "button_listen_audio": "Listen to Audio",
        "tts_status_ready": "Ready to listen",
        "tts_status_generating": "Generating audio...",
        "tts_status_success": "✅ Audio playback complete!",
        "tts_status_fail": "❌ TTS generation failed (No data)",
        "tts_status_error": "❌ TTS API error occurred",
        "history_expander_title": "📝 Load Previous Simulation History (Last 10)", 
        "initial_query_sample": "I arrived in Paris, France, but the eSIM I bought from Klook won't activate. I'm really struggling to get connected. What should I do?", 

        # ⭐ 대화형/종료 메시지
        "button_mic_input": "Voice Input",
        "prompt_customer_end": "As there are no further inquiries, we will now end this chat session.",
        "prompt_survey": "Thank you for contacting our Customer Support Center. We would be grateful if you could participate in a short survey about our service solution. Please feel free to contact us anytime if you have any additional questions.",
        "customer_closing_confirm": "Is there anything else we can assist you with today?",
        "customer_positive_response": "Thank you for your kind understanding/friendly advice.",
        "button_end_chat": "End Chat (Request Survey)",
        "agent_response_header": "✍️ Agent Response",
        "agent_response_placeholder": "Respond to the customer (Request/confirm essential information or provide solution steps)",
        "send_response_button": "Send Response",
        "request_rebuttal_button": "Request Customer's Next Reaction",
        "new_simulation_button": "Start New Simulation",
        "history_selectbox_label": "Select history to load:",
        "history_load_button": "Load Selected History",
        "delete_history_button": "❌ Delete All History", # ⭐ 다국어 키 추가
        "delete_confirm_message": "Are you sure you want to delete ALL simulation history? This action cannot be undone.", # ⭐ 다국어 키 추가
        "delete_confirm_yes": "Yes, Delete", # ⭐ 다국어 키 추가
        "delete_confirm_no": "No, Keep", # ⭐ 다국어 키 추가
        "delete_success": "✅ Successfully deleted!", # ⭐ 다국어 키 추가
        "deleting_history_progress": "Deleting history...", # ⭐ 다국어 키 추가
        "search_history_label": "Search History by Keyword", # ⭐ 다국어 키 추가
        "date_range_label": "Date Range Filter", # ⭐ 다국어 키 추가
        "no_history_found": "No history found matching the criteria." # ⭐ 다국어 키 추가
    },
    "ja": {
        "title": "パーソナライズAI学習コーチ",
        "sidebar_title": "📚 AI学習コーチ設定",
        "file_uploader": "学習資料をアップロード (PDF, TXT, HTML)",
        "button_start_analysis": "資料分析開始 (RAGインデックス作成)",
        "rag_tab": "RAG知識チャットボット",
        "content_tab": "カスタムコンテンツ生成",
        "lstm_tab": "LSTM達成度予測ダッシュボード",
        "simulator_tab": "AI顧客対応シミュレーター", 
        "rag_header": "RAG知識チャットボット (ドキュメントQ&A)",
        "rag_desc": "アップロードされたドキュメントに基づいて質問に回答します。",
        "rag_input_placeholder": "学習資料について質問してください",
        "llm_error_key": "⚠️ 警告: GEMINI APIキーが設定されていません。Streamlit Secretsに'GEMINI_API_KEY'を設置してください。",
        "llm_error_init": "LLM初期化エラー：APIキーを確認してください。",
        "content_header": "カスタム学習コンテンツ生成",
        "content_desc": "学習テーマと難易度に合わせてコンテンツを生成します。",
        "topic_label": "学習テーマ",
        "level_label": "難易度",
        "content_type_label": "コンテンツ形式",
        "level_options": ["初級", "中級", "上級"],
        "content_options": ["核心要約ノート", "選択式クイズ10問", "実践例のアイデア"],
        "button_generate": "コンテンツ生成",
        "warning_topic": "学習テーマを入力してください。",
        "lstm_header": "LSTMベース達成度予測ダッシュボード",
        "lstm_desc": "仮想の過去クイズスコアデータに基づき、LSTMモデルを訓練して将来の達成度を予測し表示します。",
        "lstm_disabled_error": "現在、ビルド環境の問題によりLSTM機能は一時的に無効化されています。「カスタムコンテンツ生成」機能を先にご利用ください。」",
        "lang_select": "言語選択",
        "embed_success": "全{count}チャンクで学習DB構築完了!",
        "embed_fail": "埋め込み失敗: フリーティアのクォータ超過またはネットワークの問題。",
        "warning_no_files": "まず学習資料をアップロードしてください。",
        "warning_rag_not_ready": "RAGが準備されていません。資料をアップロードし、分析開始ボタンを押してください。",
        "quiz_fail_structure": "クイズのデータ構造が正しくありません。",
        "select_answer": "正解を選択してください",
        "check_answer": "正解を確認",
        "next_question": "次の質問",
        "correct_answer": "正解です! 🎉",
        "incorrect_answer": "不正解です。😞",
        "correct_is": "正解",
        "explanation": "解説",
        "quiz_complete": "クイズ完了!",
        "score": "スコア",
        "retake_quiz": "クイズを再挑戦",
        "quiz_error_llm": "LLMが正しいJSONの形式を読み取れませんでしたので、クイズの生成が失敗しました。",
        "quiz_original_response": "LLM 原本応答",
        "firestore_loading": "データベースからRAGインデックスをロード中...",
        "firestore_no_index": "データベースで既存のRAGインデックスが見つかりません。ファイルをアップロードして新しく作成してください。", 
        "db_save_complete": "(DB保存完了)", # ⭐ 다국어 키 추가
        "data_analysis_progress": "資料分析および学習DB構築中...", # ⭐ 다국어 키 추가
        "response_generating": "応答生成中...", # ⭐ 다국어 키 추가
        "lstm_result_header": "達成度予測結果",
        "lstm_score_metric": "現在の予測達成度",
        "lstm_score_info": "次のクイズの推定スコアは約 **{predicted_score:.1f}点**です。学習の成果を維持または向上させてください！",
        "lstm_rerun_button": "新しい仮想データで予測",

        # ⭐ 시뮬레이터 관련 텍스트
        "simulator_header": "AI顧客対応シミュレーター",
        "simulator_desc": "難しい顧客の問い合わせに対して、AIによる対応案とガイドラインを提供します。",
        "customer_query_label": "顧客の問い合わせ内容（リンク任意）",
        "customer_type_label": "顧客の傾向",
        "customer_type_options": ["一般的な問い合わせ", "手ごわい顧客", "非常に不満な顧客"],
        "button_simulate": "対応アドバイスを要求",
        "simulation_warning_query": "顧客の問い合わせ内容を入力してください。",
        "simulation_no_key_warning": "⚠️ APIキーが不足しています。応答の生成は続行できません。（UI設定は完了しています。）",
        "simulation_advice_ready": "AIの対応アドバイスが利用可能です！",
        "simulation_advice_header": "AI対応ガイドライン",
        "simulation_draft_header": "推奨される対応草案",
        "button_listen_audio": "音声で聞く",
        "tts_status_ready": "音声再生の準備ができました",
        "tts_status_generating": "音声生成中...",
        "tts_status_success": "✅ 音声再生完了!",
        "tts_status_fail": "❌ TTS生成失敗（データなし）",
        "tts_status_error": "❌ TTS APIエラーが発生しました",
        "history_expander_title": "📝 以前の対応履歴をロード (最新 10件)", 
        "initial_query_sample": "フランスのパリに到着しましたが、Klookで購入したeSIMがアクティベートできません。接続できなくて困っています。どうすればいいですか？", 

        # ⭐ 대화형/종료 메시지
        "button_mic_input": "音声入力",
        "prompt_customer_end": "お客様からの追加のお問い合わせがないため、本チャットサポートを終了させていただきます。",
        "prompt_survey": "お問い合わせいただき、誠にありがとうございました。弊社の対応ソリューションに関する簡単なアンケートにご協力いただければ幸いです。追加のご質問がございましたらいつでもご連絡ください。",
        "customer_closing_confirm": "また、お客様にお手伝いさせて頂けるお問い合わせは御座いませんか？",
        "customer_positive_response": "親切なご対応ありがとうございました。",
        "button_end_chat": "対応終了 (アンケートを依頼)",
        "agent_response_header": "✍️ エージェント応答",
        "agent_response_placeholder": "顧客に返信 (必須情報の要求/確認、または解決策の提示)",
        "send_response_button": "応答送信",
        "request_rebuttal_button": "顧客の次の反応を要求", 
        "new_simulation_button": "新しいシミュレーションを開始",
        "history_selectbox_label": "履歴を選択してロード:",
        "history_load_button": "選択された履歴をロード",
        "delete_history_button": "❌ 全履歴を削除", 
        "delete_confirm_message": "本当にすべてのシミュレーション履歴を削除してもよろしいですか？この操作は元に戻せません。", 
        "delete_confirm_yes": "はい、削除します", 
        "delete_confirm_no": "いいえ、維持します", 
        "delete_success": "✅ 削除が完了されました!" 
    }
}


# ================================
# 4. Streamlit 핵심 Config 설정 및 Session State 초기화 (CRITICAL ZONE)
# ================================

if 'language' not in st.session_state: st.session_state.language = 'ko'
if 'uploaded_files_state' not in st.session_state: st.session_state.uploaded_files_state = None
if 'is_llm_ready' not in st.session_state: st.session_state.is_llm_ready = False
if 'is_rag_ready' not in st.session_state: st.session_state.is_rag_ready = False
if 'firestore_db' not in st.session_state: st.session_state.firestore_db = None
if 'llm_init_error_msg' not in st.session_state: st.session_state.llm_init_error_msg = None
if 'firestore_load_success' not in st.session_state: st.session_state.firestore_load_success = False

# ⭐ 시뮬레이터 전용 상태 초기화 추가
if "simulator_memory" not in st.session_state:
    # ConversationChain에서 사용할 메모리 초기화
    st.session_state.simulator_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "simulator_messages" not in st.session_state:
    st.session_state.simulator_messages = []
if "initial_advice_provided" not in st.session_state:
    st.session_state.initial_advice_provided = False
if "simulator_chain" not in st.session_state:
    st.session_state.simulator_chain = None
# ⭐ 시뮬레이터 진행 상태 추가
if "is_chat_ended" not in st.session_state:
    st.session_state.is_chat_ended = False

# ⭐ 이력 삭제 확인 모달 상태
if "show_delete_confirm" not in st.session_state:
    st.session_state.show_delete_confirm = False

# 언어 설정 로드 (UI 출력 전 필수)
L = LANG[st.session_state.language] 
API_KEY = os.environ.get("GEMINI_API_KEY")

# =======================================================
# 5. Streamlit UI 페이지 설정 (스크립트 내 첫 번째 ST 명령)
# =======================================================
st.set_page_config(page_title=L["title"], layout="wide")

# =======================================================
# 6. 서비스 초기화 및 LLM/DB 로직 (페이지 설정 후 안전하게 실행)
# =======================================================

if 'llm' not in st.session_state: 
    llm_init_error = None # ⭐ safety initialization
    if not API_KEY:
        llm_init_error = L["llm_error_key"]
    else:
        try:
            # LLM 및 Embeddings 초기화
            st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=API_KEY)
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
            st.session_state.is_llm_ready = True
            
            # Admin SDK 클라이언트 초기화 
            sa_info, error_message = _get_admin_credentials()
            
            if error_message:
                llm_init_error = f"{L['llm_error_init']} (DB Auth Error: {error_message})" 
            elif sa_info:
                db = initialize_firestore_admin() 
                st.session_state.firestore_db = db
                
                if not db:
                    llm_init_error = f"{L['llm_error_init']} (DB Client Error: Firebase Admin Init Failed)" 
                else:
                    # DB 로딩 로직 (RAG 챗봇용)
                    if 'conversation_chain' not in st.session_state:
                        # DB 로딩 시도
                        loaded_index = load_index_from_firestore(st.session_state.firestore_db, st.session_state.embeddings)
                        
                        if loaded_index:
                            st.session_state.conversation_chain = get_rag_chain(loaded_index)
                            st.session_state.is_rag_ready = True
                            st.session_state.firestore_load_success = True
                        else:
                            st.session_state.firestore_load_success = False
            
            # ⭐ 시뮬레이터 체인 초기화
            SIMULATOR_PROMPT = PromptTemplate(
                template="The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.\n\n{chat_history}\nHuman: {input}\nAI:",
                input_variables=["input", "chat_history"]
            )
            
            st.session_state.simulator_chain = ConversationChain(
                llm=st.session_state.llm,
                memory=st.session_state.simulator_memory,
                prompt=SIMULATOR_PROMPT,
                input_key="input", 
            )


        except Exception as e:
            # LLM 초기화 오류 처리 
            llm_init_error = f"{L['llm_error_init']} {e}" 
            st.session_state.is_llm_ready = False
    
    if llm_init_error:
        st.session_state.is_llm_ready = False
        st.session_state.llm_init_error_msg = llm_init_error 

# 나머지 세션 상태 초기화
if "memory" not in st.session_state:
    # RAG 체인용 메모리 초기화
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}

# ⭐ LSTM 리런 트리거 초기화 (추가)
if "lstm_rerun_trigger" not in st.session_state:
    st.session_state.lstm_rerun_trigger = time.time()

# ================================
# 7. 초기화 오류 메시지 출력 및 DB 상태 알림
# ================================

if st.session_state.llm_init_error_msg:
    st.error(st.session_state.llm_init_error_msg)
    
if st.session_state.get('firestore_db'):
    if st.session_state.get('firestore_load_success', False):
        st.success("✅ RAG 인덱스가 데이터베이스에서 성공적으로 로드되었습니다!")
    elif not st.session_state.get('is_rag_ready', False):
        st.info(L["firestore_no_index"]) # ⭐ 다국어 적용


# ================================
# 8. Streamlit UI 시작
# ================================

with st.sidebar:
    selected_lang_key = st.selectbox(
        L["lang_select"],
        options=['ko', 'en', 'ja'],
        index=['ko', 'en', 'ja'].index(st.session_state.language),
        format_func=lambda x: {"ko": "한국어", "en": "English", "ja": "日本語"}[x],
    )
    
    if selected_lang_key != st.session_state.language:
        st.session_state.language = selected_lang_key
        st.rerun() 
    
    L = LANG[st.session_state.language] 
    
    st.title(L["sidebar_title"])
    
    st.markdown("---")
    
    uploaded_files_widget = st.file_uploader(
        L["file_uploader"],
        type=["pdf","txt","html"],
        accept_multiple_files=True
    )
    
    if uploaded_files_widget:
        st.session_state.uploaded_files_state = uploaded_files_widget
    elif 'uploaded_files_state' not in st.session_state:
        st.session_state.uploaded_files_state = None
    
    files_to_process = st.session_state.uploaded_files_state if st.session_state.uploaded_files_state else []
    
    if files_to_process and st.session_state.is_llm_ready:
        if st.button(L["button_start_analysis"], key="start_analysis"):
            with st.spinner(L["data_analysis_progress"]): # ⭐ 다국어 적용
                text_chunks = get_document_chunks(files_to_process)
                vector_store = get_vector_store(text_chunks)
                
                if vector_store:
                    # RAG 인덱스가 성공적으로 생성되면 Firestore에 저장 시도
                    db = st.session_state.firestore_db
                    save_success = False
                    if db:
                        save_success = save_index_to_firestore(db, vector_store)
                    
                    if save_success:
                        st.success(L["embed_success"].format(count=len(text_chunks)) + " " + L["db_save_complete"]) # ⭐ 다국어 적용
                    else:
                        st.success(L["embed_success"].format(count=len(text_chunks)) + " (DB 저장 실패)")

                    st.session_state.conversation_chain = get_rag_chain(vector_store)
                    st.session_state.is_rag_ready = True
                else:
                    st.session_state.is_rag_ready = False
                    st.error(L["embed_fail"])

    else:
        st.session_state.is_rag_ready = False
        st.warning(L.get("warning_no_files", "먼저 학습 자료를 업로드하세요.")) 

    st.markdown("---")
    # ⭐ 새로운 탭(시뮬레이터)을 포함하여 라디오 버튼 업데이트
    feature_selection = st.radio(
        L["content_tab"], 
        [L["rag_tab"], L["content_tab"], L["lstm_tab"], L["simulator_tab"]]
    )

st.title(L["title"])

# ================================
# 9. 기능별 페이지 구현
# ================================

# ⭐ 이력 삭제 함수 (Firestore 연동)
def delete_all_history(db):
    """Firestore의 모든 상담 이력을 삭제합니다."""
    L = LANG[st.session_state.language] # 함수 내에서 L을 다시 정의
    
    if not db:
        st.error(L["firestore_no_index"])
        return
    
    try:
        # 이터레이션을 위해 스트림 사용
        docs = db.collection("simulation_histories").stream()
        
        # 삭제 작업 실행
        with st.spinner(L["deleting_history_progress"]): 
            for doc in docs:
                doc.reference.delete()
        
        # 세션 상태도 초기화
        st.session_state.simulator_messages = []
        st.session_state.simulator_memory.clear()
        st.session_state.initial_advice_provided = False
        st.session_state.show_delete_confirm = False
        st.success(L["delete_success"]) # ⭐ 다국어 적용
        st.rerun()
        
    except Exception as e:
        st.error(f"이력 삭제 중 오류 발생: {e}")

if feature_selection == L["simulator_tab"]: 
    st.header(L["simulator_header"])
    st.markdown(L["simulator_desc"])
    
    # 1. TTS 유틸리티 (상태 표시기 및 JS 함수)를 페이지 상단에 삽입
    st.markdown(f'<div id="tts_status" style="padding: 5px; text-align: center; border-radius: 5px; background-color: #f0f0f0; margin-bottom: 10px;">{L["tts_status_ready"]}</div>', unsafe_allow_html=True)
    
    # TTS JS 유틸리티를 페이지 로드 시 단 한 번만 삽입 (TTS 함수가 글로벌로 정의되도록)
    if "tts_js_loaded" not in st.session_state:
         synthesize_and_play_audio(st.session_state.language) 
         st.session_state.tts_js_loaded = True

    # 1.5 이력 삭제 버튼 및 모달
    db = st.session_state.get('firestore_db')
    col_delete, _ = st.columns([1, 4])
    with col_delete:
        if st.button(L["delete_history_button"], key="trigger_delete_history"):
            st.session_state.show_delete_confirm = True

    if st.session_state.show_delete_confirm:
        with st.container(border=True):
            st.warning(L["delete_confirm_message"])
            col_yes, col_no = st.columns(2)
            if col_yes.button(L["delete_confirm_yes"], key="confirm_delete_yes", type="primary"):
                delete_all_history(db)
            if col_no.button(L["delete_confirm_no"], key="confirm_delete_no"):
                st.session_state.show_delete_confirm = False
                st.rerun()

    # ⭐ Firebase 상담 이력 로드 및 선택 섹션
    if db:
        with st.expander(L["history_expander_title"]): # ⭐ 다국어 적용
            
            # 2. 이력 검색 및 필터링 기능 (안정성 확보를 위해 단순 로드로 변경)
            histories = load_simulation_histories(db)
            
            if histories:
                history_options = {
                    f"[{h['timestamp'].strftime('%m-%d %H:%M')}] {h['customer_type']} - {h['initial_query'][:30]}...": h
                    for h in histories
                }
                
                selected_key = st.selectbox(
                    L["history_selectbox_label"], 
                    options=list(history_options.keys())
                )
                
                if st.button(L["history_load_button"]): 
                    selected_history = history_options[selected_key]
                    
                    # 상태 복원
                    st.session_state.customer_query_text_area = selected_history['initial_query']
                    st.session_state.initial_advice_provided = True
                    st.session_state.simulator_messages = selected_history['messages']
                    st.session_state.is_chat_ended = selected_history.get('is_chat_ended', False)
                    
                    # 메모리 초기화 및 메시지 재구성 (LangChain 호환성을 위해)
                    st.session_state.simulator_memory.clear()
                    
                    # LLM 메모리에 대화 이력 재주입 (실제 LLM이 응대할 수 있도록)
                    for i, msg in enumerate(selected_history['messages']):
                         if msg['role'] == 'customer':
                             st.session_state.simulator_memory.chat_memory.add_user_message(msg['content'])
                         elif msg['role'] in ['supervisor', 'customer_rebuttal', 'customer_end', 'system_end']:
                             st.session_state.simulator_memory.chat_memory.add_ai_message(msg['content'])
                         elif msg['role'] == 'agent_response':
                             st.session_state.simulator_memory.chat_memory.add_user_message(msg['content'])
                    
                    st.rerun()
            else:
                 st.info(L.get("no_history_found", "이력이 없습니다."))


    # ⭐ LLM 초기화가 되어있지 않아도 (API Key가 없어도) UI가 작동해야 함
    if st.session_state.is_llm_ready or not API_KEY:
        if st.session_state.is_chat_ended:
            st.success(L["prompt_customer_end"] + " " + L["prompt_survey"])
            
            if st.button(L["new_simulation_button"], key="new_simulation"): 
                 st.session_state.is_chat_ended = False
                 st.session_state.initial_advice_provided = False
                 st.session_state.simulator_messages = []
                 st.session_state.simulator_memory.clear()
                 st.rerun()
            st.stop()
        
        # 1. 고객 문의 입력 필드
        if 'customer_query_text_area' not in st.session_state:
            st.session_state.customer_query_text_area = ""

        # ⭐ 초기값 설정: Klook eSIM 이슈 및 필수 정보 요청 유도 (다국어 적용)
        initial_query_placeholder = L["initial_query_sample"]
        
        customer_query = st.text_area(
            L["customer_query_label"],
            key="customer_query_text_area",
            height=150,
            placeholder=initial_query_placeholder, # ⭐ 다국어 적용
            disabled=st.session_state.initial_advice_provided
        )

        # 2. 고객 성향 선택
        # ⭐ 기본값을 '까다로운 고객'으로 설정하여 난이도 부여
        customer_type_options_list = L["customer_type_options"]
        default_index = 1 if len(customer_type_options_list) > 1 else 0 # '까다로운 고객' 또는 'Challenging Customer'
        
        customer_type_display = st.selectbox(
            L["customer_type_label"],
            customer_type_options_list,
            index=default_index,
            disabled=st.session_state.initial_advice_provided
        )
        
        # 선택된 언어 키
        current_lang_key = st.session_state.language 

        # 4. '응대 조언 요청' 버튼: 초기 시뮬레이션 시작 및 메모리 초기화
        if st.button(L["button_simulate"], key="start_simulation", disabled=st.session_state.initial_advice_provided):
            if not customer_query:
                st.warning(L["simulation_warning_query"])
                st.stop()
            
            # 초기화
            st.session_state.simulator_memory.clear()
            st.session_state.simulator_messages = []
            st.session_state.is_chat_ended = False
            
            st.session_state.simulator_messages.append({"role": "customer", "content": customer_query})
            st.session_state.simulator_memory.chat_memory.add_user_message(customer_query)
            
            # ⭐ LLM 프롬프트에 컨텍스트 분리 및 협조적인 고객 역할을 부여
            initial_prompt = f"""
            You are an AI Customer Support Supervisor. Your task is to provide expert guidance to a customer support agent.
            The customer sentiment is: {customer_type_display}.
            The customer's initial inquiry is: "{customer_query}"
            
            Based on this, provide:
            1. Crucial advice on the tone and strategy for dealing with this specific sentiment. 
            2. A concise and compassionate recommended response draft.
            
            The recommended draft MUST be strictly in {LANG[current_lang_key]['lang_select']}.
            
            **CRITICAL RULE FOR DRAFT CONTENT:**
            - **Core Topic Filtering:** Analyze the customer's inquiry to determine its main subject. 
            - **Draft Content:** The draft MUST address the core topic directly. The draft MUST ONLY request *general* information needed for ALL inquiries (like booking ID, contact info). 
            - **Technical Info:** The draft MUST NOT include specific technical troubleshooting requests (Smartphone model, Location, Last Step of troubleshooting) **UNLESS** the core inquiry is explicitly about connection/activation failures (like "won't activate" or "no connection"). If the inquiry is about eSIM activation failure, use a standard troubleshooting request template.
            
            When the Agent subsequently asks for information, **Roleplay as the Customer** who is frustrated but **MUST BE HIGHLY COOPERATIVE** and provide the requested details piece by piece (not all at once). The customer MUST NOT argue or ask why the information is needed.
            """
            
            if not API_KEY:
                # API Key가 없을 경우 모의(Mock) 데이터 사용
                mock_data = get_mock_response_data(current_lang_key, customer_type_display)
                ai_advice_text = f"### {mock_data['advice_header']}\n\n{mock_data['advice']}\n\n### {mock_data['draft_header']}\n\n{mock_data['draft']}"
                
                # 메모리에 추가
                st.session_state.simulator_messages.append({"role": "supervisor", "content": ai_advice_text})
                st.session_state.simulator_memory.chat_memory.add_ai_message(ai_advice_text)

                st.session_state.initial_advice_provided = True
                save_simulation_history(db, customer_query, customer_type_display, st.session_state.simulator_messages)
                
                st.rerun() 
            
            if API_KEY:
                # API Key가 있을 경우 LLM 호출
                with st.spinner(L["response_generating"]): # ⭐ 다국어 적용
                    try:
                        if st.session_state.simulator_chain is None:
                            st.error(L['llm_error_init'] + " (시뮬레이터 체인 초기화 실패)")
                            st.stop()

                        response_text = st.session_state.simulator_chain.predict(input=initial_prompt)
                        ai_advice_text = response_text
                        
                        st.session_state.simulator_messages.append({"role": "supervisor", "content": ai_advice_text})
                        st.session_state.initial_advice_provided = True
                        
                        save_simulation_history(db, customer_query, customer_type_display, st.session_state.simulator_messages)
                        
                        st.rerun() 
                    except Exception as e:
                        st.error(f"AI 조언 생성 중 오류 발생: {e}")
        
        # 5. 시뮬레이션 채팅 기록 표시
        st.markdown("---")
        
        # 채팅 기록 렌더링
        for message in st.session_state.simulator_messages:
            if message["role"] == "customer":
                with st.chat_message("user", avatar="🙋"):
                    st.markdown(message["content"])
            elif message["role"] == "supervisor":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])
                    render_tts_button(message["content"], st.session_state.language) 
            elif message["role"] == "agent_response":
                 with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(message["content"])
            elif message["role"] == "customer_rebuttal":
                 with st.chat_message("assistant", avatar="😠"):
                    st.markdown(message["content"])
            elif message["role"] == "customer_end":
                 with st.chat_message("assistant", avatar="😊"):
                    st.markdown(message["content"])
            elif message["role"] == "system_end":
                 with st.chat_message("assistant", avatar="✨"):
                    st.markdown(message["content"])

        # 6. 대화형 시뮬레이션 진행 (추가 채팅)
        if st.session_state.initial_advice_provided and not st.session_state.is_chat_ended:
            
            last_role = st.session_state.simulator_messages[-1]['role'] if st.session_state.simulator_messages else None
            
            # 1. 에이전트(사용자)가 응답할 차례 (초기 문의 후, 재반박 후, 매너 질문 후)
            if last_role in ["customer_rebuttal", "customer_end", "supervisor", "customer"]:
                
                st.markdown(f"### {L['agent_response_header']}") 
                
                # st.text_area를 사용하여 명시적인 입력 필드를 제공
                agent_response = st.text_area(
                    L["agent_response_placeholder"], 
                    value="",
                    key="agent_response_area_text",
                    height=150
                )
                
                if st.button(L["send_response_button"], key="send_agent_response"): 
                    if agent_response.strip():
                        st.session_state.simulator_messages.append(
                            {"role": "agent_response", "content": agent_response}
                        )
                        st.session_state.simulator_memory.chat_memory.add_user_message(agent_response)
                        # DB 저장 및 리런
                        save_simulation_history(db, st.session_state.customer_query_text_area, customer_type_display, st.session_state.simulator_messages)
                        st.rerun()
                    else:
                        st.warning("응답 내용이 비어 있습니다.")
            
            # 2. 고객의 다음 반응 요청 (LLM 호출) 또는 종료 버튼 표시
            # 에이전트의 응답 후, 고객 반응 요청 버튼 또는 종료 버튼 표시
            if last_role == "agent_response":
                
                col_end, col_next = st.columns([1, 2])
                
                # A) 응대 종료 버튼 (매너 종료)
                if col_end.button(L["button_end_chat"], key="end_chat"): 
                    closing_messages = get_closing_messages(current_lang_key)
                    
                    # 매너 질문과 최종 종료 인사는 AI의 응답으로 간주하여 메모리에 추가
                    st.session_state.simulator_messages.append({"role": "supervisor", "content": closing_messages["additional_query"]}) # 매너 질문
                    st.session_state.simulator_memory.chat_memory.add_ai_message(closing_messages["additional_query"])

                    st.session_state.simulator_messages.append({"role": "system_end", "content": closing_messages["chat_closing"]}) # 최종 종료 인사
                    st.session_state.simulator_memory.chat_memory.add_ai_message(closing_messages["chat_closing"])
                    
                    st.session_state.is_chat_ended = True
                    
                    # ⭐ Firebase 이력 업데이트: 최종 종료 상태 저장
                    save_simulation_history(db, st.session_state.customer_query_text_area, customer_type_display, st.session_state.simulator_messages)
                    
                    st.rerun()

                # B) 고객의 다음 반응 요청 (LLM 호출)
                if col_next.button(L["request_rebuttal_button"], key="request_rebuttal"): # ⭐ LLM 호출 텍스트 제거
                    if not API_KEY:
                        st.warning("API Key가 없기 때문에 LLM을 통한 대화형 시뮬레이션은 불가능합니다.")
                        st.stop()
                    
                    if st.session_state.simulator_chain is None:
                        st.error(L['llm_error_init'] + " (시뮬레이터 체인 초기화 실패)")
                        st.stop()
                        
                    # ⭐ 핵심 수정된 프롬프트 (강력하게 협조적인 고객을 유도)
                    next_reaction_prompt = f"""
                    Analyze the entire chat history. Roleplay as the customer ({customer_type_display}). 
                    Based on the agent's last message, generate ONE of the following responses in the customer's voice:
                    1. Provide **ONE** of the crucial, previously requested details (Model, Location, or Last Step) in a cooperative tone.
                    2. A short, positive closing remark (e.g., "{L['customer_positive_response']}").
                    
                    Crucially, the customer MUST be highly cooperative. If the agent asks for information, the customer MUST provide the detail requested (Model, Location, or Last Step) without arguing or asking why. The purpose of this simulation is for the agent (human user) to practice systematically collecting information and troubleshooting.
                    
                    The response MUST be strictly in {LANG[current_lang_key]['lang_select']}.
                    """
                    
                    with st.spinner(L["response_generating"]): # ⭐ 다국어 적용
                        customer_reaction = st.session_state.simulator_chain.predict(input=next_reaction_prompt)
                        
                        # 긍정적 종료 키워드 확인 (대소문자 무시)
                        positive_keywords = ["감사", "thank you", "ありがとう", L['customer_positive_response'].lower().split('/')[-1].strip()]
                        is_positive_close = any(keyword in customer_reaction.lower() for keyword in positive_keywords)
                        
                        if is_positive_close:
                            role = "customer_end" # 긍정적 종료
                            st.session_state.simulator_messages.append({"role": role, "content": customer_reaction})
                            st.session_state.simulator_memory.chat_memory.add_ai_message(customer_reaction)

                            # 긍정 종료 후 에이전트에게 매너 질문 요청
                            st.session_state.simulator_messages.append({"role": "supervisor", "content": L["customer_closing_confirm"]})
                            st.session_state.simulator_memory.chat_memory.add_ai_message(L["customer_closing_confirm"])
                        else:
                            role = "customer_rebuttal" # 재반박, 추가 질문, 또는 정보 제공
                            st.session_state.simulator_messages.append({"role": role, "content": customer_reaction})
                            st.session_state.simulator_memory.chat_memory.add_ai_message(customer_reaction)
                             
                        # DB 저장 및 리런
                        save_simulation_history(db, st.session_state.customer_query_text_area, customer_type_display, st.session_state.simulator_messages)
                        st.rerun()

    else:
        # LLM 초기화 자체에 문제가 있을 경우의 오류 메시지 (다국어)
        st.error(L["llm_error_init"])

elif feature_selection == L["rag_tab"]:
    st.header(L["rag_header"])
    st.markdown(L["rag_desc"])
    if st.session_state.get('is_rag_ready', False) and st.session_state.get('conversation_chain'):
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(L["rag_input_placeholder"]):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(L["response_generating"]): # ⭐ 다국어 적용
                    try:
                        response = st.session_state.conversation_chain.invoke({"question":prompt})
                        answer = response.get('answer','응답을 생성할 수 없습니다.' if st.session_state.language == 'ko' else 'Could not generate response.')
                        st.markdown(answer)
                        st.session_state.messages.append({"role":"assistant","content":answer})
                    except Exception as e:
                        st.error(f"챗봇 오류: {e}")
                        st.session_state.messages.append({"role":"assistant","content":"오류 발생" if st.session_state.language == 'ko' else "An error occurred"})
    else:
        st.warning(L["warning_rag_not_ready"])

elif feature_selection == L["content_tab"]:
    st.header(L["content_header"])
    st.markdown(L["content_desc"])

    if st.session_state.is_llm_ready:
        topic = st.text_input(L["topic_label"])
        
        level_map = dict(zip(L["level_options"], ["Beginner", "Intermediate", "Advanced"]))
        content_map = dict(zip(L["content_options"], ["summary", "quiz", "example"]))
        
        level_display = st.selectbox(L["level_label"], L["level_options"])
        content_type_display = st.selectbox(L["content_type_label"], L["content_options"])

        level = level_map[level_display]
        content_type = content_map[content_type_display]

        if st.button(L["button_generate"]):
            if topic:
                target_lang = {"ko": "Korean", "en": "English", "ja": "Japanese"}[st.session_state.language]
                
                if content_type == 'quiz':
                    # 10문항으로 수정된 프롬프트
                    full_prompt = f"""You are a professional AI coach at the {level} level.
Please generate exactly 10 multiple-choice questions about the topic in {target_lang}.
Your entire response MUST be a valid JSON object wrapped in ```json tags.
The JSON must have a single key named 'quiz_questions', which is an array of objects.
Each question object must contain: 'question' (string), 'options' (array of objects with 'option' (A,B,C,D) and 'text' (string)), 'correct_answer' (A,B,C, or D), and 'explanation' (string).

Topic: {topic}"""
                else:
                    display_type_text = L["content_options"][L["content_options"].index(content_type_display)]
                    full_prompt = f"""You are a professional AI coach at the {level} level.
Please generate clear and educational content in the requested {display_type_text} format based on the topic.
The response MUST be strictly in {target_lang}.

Topic: {topic}
Requested Format: {display_type_text}"""
                
                
                with st.spinner(f"Generating {content_type_display} for {topic}..."):
                    
                    quiz_data_raw = None
                    try:
                        response = st.session_state.llm.invoke(full_prompt)
                        quiz_data_raw = response.content
                        st.session_state.quiz_data_raw = quiz_data_raw # 디버깅을 위해 raw data 저장
                        
                        if content_type == 'quiz':
                            quiz_data = clean_and_load_json(quiz_data_raw)
                            
                            if quiz_data and 'quiz_questions' in quiz_data:
                                st.session_state.quiz_data = quiz_data
                                st.session_state.current_question = 0
                                st.session_state.quiz_submitted = False
                                st.session_state.quiz_results = [None] * len(quiz_data.get('quiz_questions',[]))
                                
                                st.success(f"**{topic}** - **{content_type_display}** Result:")
                            else:
                                st.error(L["quiz_error_llm"])
                                st.markdown(f"**{L['quiz_original_response']}**:")
                                st.code(quiz_data_raw, language="json")

                        else: # 일반 콘텐츠 (요약, 예제)
                            st.success(f"**{topic}** - **{content_type_display}** Result:")
                            st.markdown(response.content)

                    except Exception as e:
                        st.error(f"Content Generation Error: {e}")
                        if quiz_data_raw:
                            st.markdown(f"**{L['quiz_original_response']}**: {quiz_data_raw}")

            else:
                st.warning(L["warning_topic"])
    else:
        st.error(L["llm_error_init"])
        
    # 퀴즈 풀이 렌더링을 메인 루프에서 조건부로 단 한 번 호출
    is_quiz_ready = content_type == 'quiz' and 'quiz_data' in st.session_state and st.session_state.quiz_data
    if is_quiz_ready and st.session_state.get('current_question', 0) < len(st.session_state.quiz_data.get('quiz_questions', [])):
        render_interactive_quiz(st.session_state.quiz_data, st.session_state.language)

elif feature_selection == L["lstm_tab"]:
    st.header(L["lstm_header"])
    st.markdown(L["lstm_desc"])
    
    if st.button(L["lstm_rerun_button"], key="rerun_lstm", on_click=force_rerun_lstm):
        pass
    
    try:
        # st.session_state.lstm_rerun_trigger가 변경될 때마다 캐시가 무효화되고 함수가 실행됨
        model, data = load_or_train_lstm()
        look_back = 5
        # 예측
        X_input = data[-look_back:]
        X_input = np.reshape(X_input, (1, look_back, 1))
        predicted_score = model.predict(X_input, verbose=0)[0][0]

        st.markdown("---")
        st.subheader(L["lstm_result_header"]) # ⭐ 다국어 적용
        col_score, col_chart = st.columns([1, 2])
        
        with col_score:
            st.metric(L["lstm_score_metric"], f"{predicted_score:.1f}{'점' if st.session_state.language == 'ko' else ''}") # ⭐ 다국어 적용
            st.info(L["lstm_score_info"].format(predicted_score=predicted_score)) # ⭐ 다국어 적용

        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(data, label='Past Scores', marker='o')
            ax.plot(len(data), predicted_score, label='Predicted Next Score', marker='*', color='red', markersize=10)
            ax.set_title(L["lstm_header"]) # ⭐ 다국어 적용
            ax.set_xlabel(f"Time ({L.get('score', 'Score')} attempts)")
            ax.set_ylabel(f"{L.get('score', 'Score')} (0-100)")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        # Streamlit 환경에서 tensorflow/matplotlib/LSTM 관련 문제가 발생할 경우의 fallback
        st.error("LSTM 모델 실행 중 오류가 발생했습니다. 환경 종속성 문제일 수 있습니다. (오류 메시지: %s)" % e)
        st.info("이 기능은 LLM 및 RAG 기능과는 별개로, 학습 성과 시뮬레이션을 위해 제공됩니다.")
