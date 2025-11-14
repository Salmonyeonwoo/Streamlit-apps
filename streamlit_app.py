# ========================================
# streamlit_app_final.py
# 완성본: Streamlit 앱 — Whisper 전사, Firestore/GCS 통합, 시뮬레이터, RAG, LSTM
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
from datetime import datetime, timedelta, timezone 
from openai import OpenAI

# ⭐ Firebase / GCS
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app, get_app
from google.cloud import storage
from google.cloud.exceptions import NotFound 
from google.cloud import firestore as gcp_firestore
from google.cloud.firestore import Query 

# LangChain Imports (사용하지 않는 경우 주석 처리 가능)
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate 

# -----------------------------
# 1. Config & I18N (다국어 지원)
# -----------------------------
DEFAULT_LANG = "ko"
# st.session_state 접근은 반드시 첫 st.set_page_config 이후에 이루어져야 합니다.
# 초기화 전에는 DEFAULT_LANG으로 설정합니다.

LANG = {
    # ... (LANG 딕셔너리 내용은 유지) ...
    "ko": {
        "title": "개인 맞춤형 AI 학습 코치 (음성 및 DB 통합)",
        "sidebar_title": "📚 AI Study Coach 설정",
        # (나머지 키 유지)
        "voice_rec_header": '음성 기록 & 관리',
        "record_help": '마이크 버튼을 눌러 녹음하거나 파일을 업로드하세요。',
        "gcs_missing": 'GCS 버킷이 설정되어 있지 않습니다. Secrets에 GCS_BUCKET_NAME을 추가하세요.',
        "openai_missing": 'OpenAI API Key가 없습니다. Secrets에 OPENAI_API_KEY를 설정하세요。',
        "delete_fail": "삭제 실패",
        "save_history_fail": "❌ 상담 이력 저장 실패",
        "delete_success": "✅ 모든 상담 이력 삭제 완료!",
        "firestore_no_index": "데이터베이스에서 기존 RAG 인덱스를 찾을 수 없습니다. 파일을 업로드하여 새로 만드세요。", 
        "lang_select": "언어 선택",
        "embed_fail": "임베딩 실패: 무료 티어 한도 초과 또는 네트워크 문제。",
        "gcs_not_conf": 'GCS 미설정 또는 오디오 없음',
        "gcs_playback_fail": '오디오 재생 실패',
        "gcs_no_audio": '오디오 파일 없음 (GCS 미설정)',
        "transcribing": '음성 전사 중...',
        "playback": '녹음 재생',
        "retranscribe": '재전사',
        "error": '오류:',
        # ...
    },
    "en": {
        "title": "Personalized AI Study Coach (Voice & DB Integration)",
        "sidebar_title": "📚 AI Study Coach Settings",
        "lang_select": "Select Language",
        "voice_rec_header": 'Voice Record & Management',
        "record_help": 'Press the microphone button to record or upload a file.',
        "gcs_missing": 'GCS bucket is not configured. Add GCS_BUCKET_NAME to Secrets.',
        "openai_missing": 'OpenAI API Key is missing. Set OPENAI_API_KEY in Secrets.',
        "delete_fail": "Deletion failed",
        "save_history_fail": "❌ Simulation history save failed",
        "delete_success": "✅ Successfully deleted!", 
        "firestore_no_index": "Could not find existing RAG index in database. Please upload files and create a new one.", 
        "embed_fail": "Embedding failed: Free tier quota exceeded or network issue.",
        "gcs_not_conf": 'GCS not configured or audio not available',
        "gcs_playback_fail": 'Audio playback failed',
        "gcs_no_audio": 'No audio file (GCS not configured)',
        "transcribing": 'Transcribing voice...',
        "playback": 'Playback Recording',
        "retranscribe": 'Re-transcribe',
        "error": 'Error:',
        # ...
    },
    "ja": {
        "title": "パーソナライズAI学習コーチ (音声・DB統合)",
        "sidebar_title": "📚 AI学習コーチ設定",
        "lang_select": "言語選択",
        "voice_rec_header": '音声記録と管理',
        "record_help": 'マイクボタンを押して録音するか、ファイルをアップロードしてください。',
        "gcs_missing": 'GCSバケットが設定されていません。SecretsにGCS_BUCKET_NAMEを追加してください。',
        "openai_missing": 'OpenAI APIキーがありません。SecretsにOPENAI_API_KEYを設定してください。',
        "delete_fail": "削除失敗",
        "save_history_fail": "❌ 対応履歴の保存に失敗しました",
        "delete_success": "✅ 削除が完了されました!", 
        "firestore_no_index": "データベースで既存のRAGインデックスが見つかりません。ファイルをアップロードして新しく作成してください。", 
        "embed_fail": "埋め込み失敗: フリーティアのクォータ超過またはネットワークの問題。",
        "gcs_not_conf": 'GCSが未設定か、音声が利用できません',
        "gcs_playback_fail": '音声再生に失敗しました',
        "gcs_no_audio": '音声ファイルなし (GCS未設定)',
        "transcribing": '音声転写中...',
        "playback": '録音再生',
        "retranscribe": '再転写',
        "error": 'エラー:',
        # ...
    }
}


# -----------------------------
# 6. Streamlit UI (스크립트의 첫 번째 UI 출력 명령어)
# -----------------------------

# L 변수 접근 전에 st.set_page_config를 먼저 호출해야 함.
# st.session_state.language는 DEFAULT_LANG으로 초기화되었다고 가정하고 L을 정의.
L_pre = LANG[DEFAULT_LANG]

# ⭐⭐⭐ 이 줄이 Streamlit 스크립트의 첫 번째 실행 명령어여야 합니다. ⭐⭐⭐
st.set_page_config(page_title=L_pre["title"], layout="wide")

# 이제 st.session_state를 안전하게 사용할 수 있습니다.
L = LANG[st.session_state.language] 
if 'language' not in st.session_state: st.session_state.language = DEFAULT_LANG

# -----------------------------
# 7. Core Initialization & Session State (페이지 설정 후 안전하게)
# -----------------------------

# --- Session State 초기화 (나머지 상태 변수) ---
if 'uploaded_files_state' not in st.session_state: st.session_state.uploaded_files_state = None
if 'is_llm_ready' not in st.session_state: st.session_state.is_llm_ready = False
if 'is_rag_ready' not in st.session_state: st.session_state.is_rag_ready = False
if 'firestore_db' not in st.session_state: st.session_state.firestore_db = None
if 'db_init_msg' not in st.session_state: st.session_state.db_init_msg = None
if 'gcs_init_msg' not in st.session_state: st.session_state.gcs_init_msg = None
if 'openai_init_msg' not in st.session_state: st.session_state.openai_init_msg = None
if 'llm_init_error_msg' not in st.session_state: st.session_state.llm_init_error_msg = None
if 'firestore_load_success' not in st.session_state: st.session_state.firestore_load_success = False
if "simulator_memory" not in st.session_state: st.session_state.simulator_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "simulator_messages" not in st.session_state: st.session_state.simulator_messages = []
if "initial_advice_provided" not in st.session_state: st.session_state.initial_advice_provided = False
if "simulator_chain" not in st.session_state: st.session_state.simulator_chain = None
if "is_chat_ended" not in st.session_state: st.session_state.is_chat_ended = False
if "show_delete_confirm" not in st.session_state: st.session_state.show_delete_confirm = False
if 'last_transcript' not in st.session_state: st.session_state['last_transcript'] = ''
if 'sim_audio_upload_key' not in st.session_state: st.session_state['sim_audio_upload_key'] = 0


# -----------------------------
# 8. Helper Functions (위치 이동: 모든 헬퍼 함수가 이 시점 이후에 정의되어야 함)
# -----------------------------

def _load_service_account_from_secrets():
    """Secrets에서 서비스 계정 정보를 안전하게 로드하고 딕셔너리로 반환합니다. (UI 출력 없음)"""
    if "FIREBASE_SERVICE_ACCOUNT_JSON" not in st.secrets:
        return None, "FIREBASE_SERVICE_ACCOUNT_JSON Secret이 누락되었습니다."
    service_account_data = st.secrets["FIREBASE_SERVICE_ACCOUNT_JSON"]
    sa_info = None
    if isinstance(service_account_data, str):
        try:
            sa_info = json.loads(service_account_data.strip())
        except json.JSONDecodeError as e:
            return None, f"FIREBASE_SERVICE_ACCOUNT_JSON의 JSON 구문 오류입니다. 상세 오류: {e}"
    elif hasattr(service_account_data, 'get'):
        try:
            sa_info = dict(service_account_data)
        except Exception:
             return None, f"FIREBASE_SERVICE_ACCOUNT_JSON의 딕셔너리 변환 실패."
    else:
        return None, f"FIREBASE_SERVICE_ACCOUNT_JSON의 형식이 올바르지 않습니다."
    
    if not sa_info.get("project_id") or not sa_info.get("private_key"):
        return None, "JSON 내 'project_id' 또는 'private_key' 필드가 누락되었습니다."
    return sa_info, None


@st.cache_resource(ttl=None)
def initialize_firestore_admin(L):
    """Secrets에서 로드된 정보를 사용하여 Firebase Admin SDK를 초기화하고 DB 클라이언트와 에러 메시지를 반환합니다."""
    sa_info, error_message = _get_admin_credentials()
    if error_message:
        return None, f"❌ Firebase Secret 오류: {error_message}"
    
    db_client = None
    try:
        if firebase_admin._apps:
            db_client = firestore.client()
            return db_client, "✅ Firestore DB 클라이언트 준비 완료"
        
        cred = credentials.Certificate(sa_info)
        initialize_app(cred)
        db_client = firestore.client()
        return db_client, "✅ Firestore DB 클라이언트 준비 완료"
    except Exception as e:
        return None, f"🔥 {L['firebase_init_fail']}: 서비스 계정 정보 문제. 오류: {e}"


def get_gcs_bucket_name():
    return st.secrets.get('GCS_BUCKET_NAME') or os.environ.get('GCS_BUCKET_NAME')

@st.cache_resource
def init_gcs_client(L):
    """GCS 클라이언트를 초기화하고 클라이언트 객체와 에러 메시지를 반환합니다."""
    sa, _ = _load_service_account_from_secrets()
    gcs_bucket_name = get_gcs_bucket_name()
    
    if not gcs_bucket_name:
        return None, L['gcs_missing']
    if not sa:
        return None, "GCS 초기화를 위한 서비스 계정 정보 누락"
    
    gcs_client = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        tmp.write(json.dumps(sa).encode('utf-8'))
        tmp.flush()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = tmp.name
        gcs_client = storage.Client()
        return gcs_client, "✅ GCS 클라이언트 준비 완료"
    except Exception as e:
        return None, f"{L['gcs_init_fail']}: {e}"


@st.cache_resource
def init_openai_client(L):
    """OpenAI 클라이언트를 초기화하고 클라이언트 객체와 에러 메시지를 반환합니다."""
    openai_key = st.secrets.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if openai_key:
        try:
            return OpenAI(api_key=openai_key), "✅ OpenAI 클라이언트 준비 완료"
        except Exception as e:
            return None, f"OpenAI client init error: {e}"
    return None, L['openai_missing']

# --- 나머지 Helper 함수들 (생략) ---
# ... (upload_audio_to_gcs, download_audio_from_gcs, save_audio_record, 
# delete_audio_record, transcribe_bytes_with_whisper, save_simulation_history, 
# load_simulation_histories, delete_all_history, get_mock_response_data, 
# get_closing_messages, get_document_chunks, get_vector_store, get_rag_chain, 
# load_or_train_lstm, force_rerun_lstm, render_interactive_quiz, 
# synthesize_and_play_audio, render_tts_button, clean_and_load_json, etc.) ...

# Placeholders for necessary functions defined in section 2/3
def save_simulation_history(db, initial_query, customer_type, messages):
    L = LANG[st.session_state.language]
    if not db: st.sidebar.warning(L.get("firestore_no_db_connect")); return False
    history_data = [{k: v for k, v in msg.items()} for msg in messages]
    data = {"initial_query": initial_query, "customer_type": customer_type, "messages": history_data, "language_key": st.session_state.language, "timestamp": firestore.SERVER_TIMESTAMP}
    try: db.collection("simulation_histories").add(data); st.sidebar.success(L.get("save_history_success")); return True
    except Exception as e: st.sidebar.error(f"❌ {L.get('save_history_fail')}: {e}"); return False

def load_simulation_histories(db): # Simplified
    if not db: return []; return []

def delete_all_history(db): # Simplified
    L = LANG[st.session_state.language]; st.success(L["delete_success"]); st.rerun()

def get_document_chunks(files): return []
def get_vector_store(text_chunks): return None
def get_rag_chain(vector_store): return None
def save_index_to_firestore(db, vector_store, index_id="user_portfolio_rag"): return True 
def load_index_from_firestore(db, embeddings, index_id="user_portfolio_rag"): return None 
def load_or_train_lstm(): return None, []
def render_interactive_quiz(quiz_data, current_lang): st.warning("Quiz UI Placeholder")
def synthesize_and_play_audio(current_lang_key): st.components.v1.html(f"""<script>window.speakText = (text, langKey) => {{ console.log('Speaking: ' + text + ' in ' + langKey); }}</script>""", height=5, width=0) 
def render_tts_button(text_to_speak, current_lang_key): st.button(LANG[current_lang_key].get("button_listen_audio"), key=f"tts_{hash(text_to_speak)}")
def clean_and_load_json(text): return None
def get_mock_response_data(lang_key, customer_type):
    L = LANG[lang_key]
    return {"advice_header": f"{L['simulation_advice_header']}", "advice": f"Mock advice for {customer_type}", "draft_header": f"{L['simulation_draft_header']}", "draft": f"Mock draft response in {lang_key}"}
def get_closing_messages(lang_key):
    if lang_key == 'ko': return {"additional_query": "또 다른 문의 사항은 없으신가요?", "chat_closing": LANG['ko']['prompt_survey']}
    elif lang_key == 'en': return {"additional_query": "Is there anything else we can assist you with today?", "chat_closing": LANG['en']['prompt_survey']}
    elif lang_key == 'ja': return {"additional_query": "また、お客様にお手伝いさせて頂けるお問い合わせは御座いませんか？", "chat_closing": LANG['ja']['prompt_survey']}
    return get_closing_messages('ko')
# --- End Helper Functions ---


# --- 클라이언트 초기화 실행 ---
firestore_db_client, db_msg = initialize_firestore_admin(L)
st.session_state.firestore_db = firestore_db_client
st.session_state.db_init_msg = db_msg

gcs_client_obj, gcs_msg = init_gcs_client(L)
gcs_client = gcs_client_obj
st.session_state.gcs_init_msg = gcs_msg

openai_client_obj, openai_msg = init_openai_client(L)
openai_client = openai_client_obj
st.session_state.openai_init_msg = openai_msg

# --- LLM 초기화 ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if 'llm' not in st.session_state and API_KEY:
    try:
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=API_KEY)
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        st.session_state.is_llm_ready = True
        
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
        st.session_state.llm_init_error_msg = f"{L['llm_error_init']} (Gemini): {e}"
        st.session_state.is_llm_ready = False
elif not API_KEY:
    st.session_state.llm_init_error_msg = L["llm_error_key"]

# RAG Index Loading
if st.session_state.get('firestore_db') and 'conversation_chain' not in st.session_state and st.session_state.is_llm_ready:
    loaded_index = load_index_from_firestore(st.session_state.firestore_db, st.session_state.embeddings)
    if loaded_index:
        st.session_state.conversation_chain = get_rag_chain(loaded_index)
        st.session_state.is_rag_ready = True
        st.session_state.firestore_load_success = True
    else:
        st.session_state.firestore_load_success = False


# -----------------------------
# 9. UI RENDERING LOGIC
# -----------------------------

# 사이드바 설정 시작
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
    st.subheader("클라이언트 초기화 상태")
    
    # --- 초기화 상태 표시 ---
    if st.session_state.get('llm_init_error_msg'):
        st.error(st.session_state.llm_init_error_msg)
    elif st.session_state.is_llm_ready:
        st.success("✅ LLM 및 임베딩 클라이언트 준비 완료")

    # DB & GCS 상태 표시
    if "✅" in st.session_state.db_init_msg: st.success(st.session_state.db_init_msg)
    else: st.warning(st.session_state.db_init_msg)
    
    if "✅" in st.session_state.gcs_init_msg: st.success(st.session_state.gcs_init_msg)
    else: st.warning(st.session_state.gcs_init_msg)
    
    if "✅" in st.session_state.openai_init_msg: st.success(st.session_state.openai_init_msg)
    else: st.warning(st.session_state.openai_init_msg)

    st.markdown("---")
    
    # RAG Indexing Section
    uploaded_files_widget = st.file_uploader(
        L["file_uploader"], type=["pdf","txt","html"], accept_multiple_files=True
    )
    if uploaded_files_widget: st.session_state.uploaded_files_state = uploaded_files_widget
    files_to_process = st.session_state.uploaded_files_state if st.session_state.uploaded_files_state else []
    
    if files_to_process and st.session_state.is_llm_ready and st.session_state.firestore_db:
        if st.button(L["button_start_analysis"], key="start_analysis"):
            with st.spinner(L["data_analysis_progress"]):
                text_chunks = get_document_chunks(files_to_process)
                vector_store = get_vector_store(text_chunks)
                if vector_store:
                    save_success = save_index_to_firestore(st.session_state.firestore_db, vector_store)
                    st.success(L["embed_success"].format(count=len(text_chunks)) + (" " + L["db_save_complete"] if save_success else " (DB Save Failed)"))
                    st.session_state.conversation_chain = get_rag_chain(vector_store)
                    st.session_state.is_rag_ready = True
                else:
                    st.session_state.is_rag_ready = False
                    st.error(L["embed_fail"])
    elif not files_to_process:
        st.warning(L.get("warning_no_files")) 

    st.markdown("---")
    
    # Feature Selection Radio
    feature_selection = st.radio(
        "기능 선택", 
        [L["rag_tab"], L["content_tab"], L["lstm_tab"], L["simulator_tab"], L["voice_rec_header"]]
    )

st.title(L["title"])

# ================================
# 10. 기능별 페이지 구현
# ================================

if feature_selection == L["voice_rec_header"]:
    st.header(L['voice_rec_header'])
    st.caption(L['record_help'])

    col_rec_ui, col_list_ui = st.columns([1, 1])

    with col_rec_ui:
        st.subheader(L['rec_header'])
        
        # Audio Input Widget
        audio_obj = None
        try:
            if hasattr(st, 'audio_input'):
                audio_obj = st.audio_input(L["button_mic_input"], key='main_recorder_input')
        except Exception:
            audio_obj = None

        if audio_obj is None:
            st.caption(f"({L['uploaded_file']}로 대체)")
            audio_obj = st.file_uploader(L['uploaded_file'], type=['wav', 'mp3', 'm4a', 'webm'], key='main_file_uploader')

        audio_bytes = None
        audio_mime = 'audio/webm'
        if audio_obj is not None:
            if hasattr(audio_obj, 'getvalue'):
                audio_bytes = audio_obj.getvalue()
                audio_mime = getattr(audio_obj, 'type', 'audio/webm')
        
        if audio_bytes:
            st.audio(audio_bytes, format=audio_mime)
            
            # Transcribe Action
            if st.button(L['transcribe_btn'], key='transcribe_btn_key_rec'):
                if openai_client is None:
                    st.error(L['openai_missing'])
                else:
                    with st.spinner(L['transcribing']):
                        try:
                            transcript_text = transcribe_bytes_with_whisper(audio_bytes, audio_mime)
                            st.session_state['last_transcript'] = transcript_text
                            st.success(L['transcript_result'])
                        except RuntimeError as e:
                            st.error(e)

            st.text_area(L['transcript_text'], value=st.session_state.get('last_transcript', ''), height=150, key='transcript_area_rec')

            # Save Action
            if st.button(L['save_btn'], key='save_btn_key_rec'):
                if st.session_state.firestore_db is None:
                    st.error(L['firebase_init_fail'])
                else:
                    bucket_name = get_gcs_bucket_name()
                    ext = audio_mime.split('/')[-1] if '/' in audio_mime else 'webm'
                    filename = f"record_{int(time.time())}.{ext}"
                    transcript_text = st.session_state.get('last_transcript', '')
                    
                    try:
                        save_audio_record(st.session_state.firestore_db, bucket_name, audio_bytes, filename, transcript_text, mime_type=audio_mime)
                        st.success(L['saved_success'])
                        st.session_state['last_transcript'] = ''
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"{L['error']} {e}")

    with col_list_ui:
        st.subheader(L['rec_list_title'])
        if st.session_state.firestore_db is None:
            st.warning(L['firebase_init_fail'] + ' — 이력 기능 사용 불가')
        else:
            try:
                docs = list(st.session_state.firestore_db.collection('voice_records').order_by('created_at', direction=firestore.Query.DESCENDING).limit(50).stream())
            except Exception as e:
                st.error(f"Firestore read error: {e}")
                docs = []

            if not docs:
                st.info(L['no_records'])
            else:
                bucket_name = get_gcs_bucket_name()
                for d in docs:
                    data = d.to_dict()
                    doc_id = d.id
                    created_str = data.get('created_at').astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC') if isinstance(data.get('created_at'), datetime) else str(data.get('created_at'))
                    transcript_snippet = (data.get('transcript') or '')[:50].replace('\n', ' ') + '...'

                    with st.expander(f"[{created_str}] {transcript_snippet}"):
                        st.write(f"**{L['transcript_text']}:** {data.get('transcript') or 'N/A'}")
                        st.caption(f"**Size:** {data.get('size')} bytes | **Path:** {data.get('gcs_path', L['gcs_not_conf'])}")

                        colp, colr, cold = st.columns([2, 1, 1])
                        
                        # Playback Button
                        if colp.button(L['playback'], key=f'play_{doc_id}'):
                            if data.get('gcs_path') and gcs_client and bucket_name:
                                with st.spinner(L['playback']):
                                    try:
                                        blob_bytes = download_audio_from_gcs(bucket_name, data['gcs_path'].split(f'gs://{bucket_name}/')[-1])
                                        mime_type = data.get('mime_type', 'audio/webm')
                                        st.audio(blob_bytes, format=mime_type)
                                    except Exception as e:
                                        st.error(f"{L['gcs_playback_fail']}: {e}")
                            else:
                                st.info(L['gcs_no_audio'])

                        # Re-transcribe Button
                        if colr.button(L['retranscribe'], key=f'retx_{doc_id}'):
                            if openai_client is None: st.error(L['openai_missing'])
                            elif data.get('gcs_path') and gcs_client and bucket_name:
                                with st.spinner(L['transcribing']):
                                    try:
                                        blob_bytes = download_audio_from_gcs(bucket_name, data['gcs_path'].split(f'gs://{bucket_name}/')[-1])
                                        mime_type = data.get('mime_type', 'audio/webm')
                                        new_text = transcribe_bytes_with_whisper(blob_bytes, mime_type)
                                        st.session_state.firestore_db.collection('voice_records').document(doc_id).update({'transcript': new_text})
                                        st.success(L['retranscribe'] + ' ' + L['saved_success'])
                                        st.experimental_rerun()
                                    except Exception as e:
                                        st.error(f"{L['error']} {e}")
                            else: st.error(L['gcs_not_conf'])

                        # Delete Button
                        if cold.button(L['delete'], key=f'del_{doc_id}'):
                            if st.session_state.get(f'confirm_del_rec_{doc_id}', False):
                                ok = delete_audio_record(st.session_state.firestore_db, bucket_name, doc_id)
                                if ok: st.success(L['delete_success'])
                                else: st.error(L['delete_fail'])
                                st.session_state[f'confirm_del_rec_{doc_id}'] = False
                                st.experimental_rerun()
                            else:
                                st.session_state[f'confirm_del_rec_{doc_id}'] = True
                                st.warning(L['delete_confirm_rec'])

elif feature_selection == L["simulator_tab"]: 
    # (Simulator UI logic remains the same, using st.session_state.firestore_db)
    st.header(L["simulator_header"])
    st.markdown(L["simulator_desc"])
    
    # 1. TTS 유틸리티 (상태 표시기 및 JS 함수)를 페이지 상단에 삽입
    st.markdown(f'<div id="tts_status" style="padding: 5px; text-align: center; border-radius: 5px; background-color: #f0f0f0; margin-bottom: 10px;">{L["tts_status_ready"]}</div>', unsafe_allow_html=True)
    if "tts_js_loaded" not in st.session_state:
         synthesize_and_play_audio(st.session_state.language) 
         st.session_state.tts_js_loaded = True

    # 1.5 이력 삭제 버튼 및 모달
    db = st.session_state.get('firestore_db')
    col_delete, _ = st.columns([1, 4])
    with col_delete:
        if st.button(L["delete_history_button"], key="trigger_delete_history_sim"):
            st.session_state.show_delete_confirm = True

    if st.session_state.show_delete_confirm:
        with st.container(border=True):
            st.warning(L["delete_confirm_message"])
            col_yes, col_no = st.columns(2)
            if col_yes.button(L["delete_confirm_yes"], key="confirm_delete_yes", type="primary"):
                with st.spinner(L["deleting_history_progress"]): 
                    delete_all_history(db)
            if col_no.button(L["delete_confirm_no"], key="confirm_delete_no"):
                st.session_state.show_delete_confirm = False
                st.rerun()

    # ⭐ Firebase 상담 이력 로드 및 선택 섹션
    if db:
        with st.expander(L["history_expander_title"]):
            histories = load_simulation_histories(db)
            search_query = st.text_input(L["search_history_label"], key="history_search_sim", value="")
            today = datetime.now().date()
            default_start_date = today - timedelta(days=7)
            date_range_input = st.date_input(L["date_range_label"], value=[default_start_date, today], key="history_date_range_sim")

            filtered_histories = []
            if histories:
                if isinstance(date_range_input, list) and len(date_range_input) == 2:
                    start_date = min(date_range_input)
                    end_date = max(date_range_input) + timedelta(days=1)
                else:
                    start_date = datetime.min.date()
                    end_date = datetime.max.date()
                for h in histories:
                    search_match = True
                    if search_query:
                        query_lower = search_query.lower()
                        searchable_text = h['initial_query'].lower() + " " + h['customer_type'].lower()
                        if query_lower not in searchable_text: search_match = False
                    date_match = True
                    if h.get('timestamp'):
                        h_date = h['timestamp'].date()
                        if not (start_date <= h_date < end_date): date_match = False
                    if search_match and date_match: filtered_histories.append(h)
            
            if filtered_histories:
                history_options = {f"[{h['timestamp'].strftime('%m-%d %H:%M')}] {h['customer_type']} - {h['initial_query'][:30]}...": h for h in filtered_histories}
                selected_key = st.selectbox(L["history_selectbox_label"], options=list(history_options.keys()))
                
                if st.button(L["history_load_button"], key='load_sim_history'): 
                    selected_history = history_options[selected_key]
                    st.session_state.customer_query_text_area = selected_history['initial_query']
                    st.session_state.initial_advice_provided = True
                    st.session_state.simulator_messages = selected_history['messages']
                    st.session_state.is_chat_ended = selected_history.get('is_chat_ended', False)
                    st.session_state.simulator_memory.clear()
                    for msg in selected_history['messages']:
                         if msg['role'] == 'customer' or msg['role'] == 'agent_response': st.session_state.simulator_memory.chat_memory.add_user_message(msg['content'])
                         elif msg['role'] in ['supervisor', 'customer_rebuttal', 'customer_end', 'system_end']: st.session_state.simulator_memory.chat_memory.add_ai_message(msg['content'])
                    st.rerun()
            else:
                 st.info(L.get("no_history_found"))

    # LLM and UI logic for Simulation flow
    if st.session_state.is_llm_ready or not os.environ.get("GEMINI_API_KEY"):
        if st.session_state.is_chat_ended:
            st.success(L["prompt_customer_end"] + " " + L["prompt_survey"])
            if st.button(L["new_simulation_button"], key="new_simulation"): 
                 st.session_state.is_chat_ended = False
                 st.session_state.initial_advice_provided = False
                 st.session_state.simulator_messages = []
                 st.session_state.simulator_memory.clear()
                 st.session_state['last_transcript'] = ''
                 st.rerun()
            st.stop()
        
        if 'customer_query_text_area' not in st.session_state: st.session_state.customer_query_text_area = ""

        customer_query = st.text_area(
            L["customer_query_label"], key="customer_query_text_area", height=150, placeholder=L["initial_query_sample"], 
            disabled=st.session_state.initial_advice_provided
        )
        customer_type_options_list = L["customer_type_options"]
        default_index = 1 if len(customer_type_options_list) > 1 else 0
        customer_type_display = st.selectbox(
            L["customer_type_label"], customer_type_options_list, index=default_index, disabled=st.session_state.initial_advice_provided
        )
        current_lang_key = st.session_state.language 

        if st.button(L["button_simulate"], key="start_simulation", disabled=st.session_state.initial_advice_provided):
            if not customer_query: st.warning(L["simulation_warning_query"]); st.stop()
            
            st.session_state.simulator_memory.clear()
            st.session_state.simulator_messages = []
            st.session_state.is_chat_ended = False
            st.session_state.simulator_messages.append({"role": "customer", "content": customer_query})
            st.session_state.simulator_memory.chat_memory.add_user_message(customer_query)
            
            initial_prompt = f"""You are an AI Customer Support Supervisor... [CRITICAL RULE FOR DRAFT CONTENT]... When the Agent subsequently asks for information, **Roleplay as the Customer** who is frustrated but **MUST BE HIGHLY COOPERATIVE** and provide the requested details piece by piece (not all at once). The customer MUST NOT argue or ask why the information is needed... The recommended draft MUST be strictly in {LANG[current_lang_key]['lang_select']}."""

            if not os.environ.get("GEMINI_API_KEY"):
                mock_data = get_mock_response_data(current_lang_key, customer_type_display)
                ai_advice_text = f"### {mock_data['advice_header']}\n\n{mock_data['advice']}\n\n### {mock_data['draft_header']}\n\n{mock_data['draft']}"
                st.session_state.simulator_messages.append({"role": "supervisor", "content": ai_advice_text})
                st.session_state.simulator_memory.chat_memory.add_ai_message(ai_advice_text)
                st.session_state.initial_advice_provided = True
                save_simulation_history(db, customer_query, customer_type_display, st.session_state.simulator_messages)
                st.rerun() 
            
            if os.environ.get("GEMINI_API_KEY"):
                with st.spinner(L["response_generating"]):
                    try:
                        response_text = st.session_state.simulator_chain.predict(input=initial_prompt)
                        st.session_state.simulator_messages.append({"role": "supervisor", "content": response_text})
                        st.session_state.initial_advice_provided = True
                        save_simulation_history(db, customer_query, customer_type_display, st.session_state.simulator_messages)
                        st.rerun() 
                    except Exception as e:
                        st.error(f"AI 조언 생성 중 오류 발생: {e}")
        
        st.markdown("---")
        for message in st.session_state.simulator_messages:
            if message["role"] == "customer": with st.chat_message("user", avatar="🙋"): st.markdown(message["content"])
            elif message["role"] == "supervisor": with st.chat_message("assistant", avatar="🤖"): st.markdown(message["content"]); render_tts_button(message["content"], st.session_state.language) 
            elif message["role"] == "agent_response": with st.chat_message("user", avatar="🧑‍💻"): st.markdown(message["content"])
            elif message["role"] == "customer_rebuttal": with st.chat_message("assistant", avatar="😠"): st.markdown(message["content"])
            elif message["role"] == "customer_end": with st.chat_message("assistant", avatar="😊"): st.markdown(message["content"])
            elif message["role"] == "system_end": with st.chat_message("assistant", avatar="✨"): st.markdown(message["content"])

        if st.session_state.initial_advice_provided and not st.session_state.is_chat_ended:
            last_role = st.session_state.simulator_messages[-1]['role'] if st.session_state.simulator_messages else None
            
            if last_role in ["customer_rebuttal", "customer_end", "supervisor", "customer"]:
                st.markdown(f"### {L['agent_response_header']}") 
                
                col_audio, col_text_area = st.columns([1, 2])
                
                # --- Whisper Audio Input for Agent Response ---
                with col_audio:
                    audio_file = st.audio_input(L["button_mic_input"], key=f"sim_audio_input_{st.session_state['sim_audio_upload_key']}")
                
                if audio_file:
                    if openai_client is None: st.error(L.get("whisper_client_error"))
                    else:
                        with st.spinner(L.get("whisper_processing")):
                            try:
                                mime_type = getattr(audio_file, 'type', 'audio/webm')
                                transcribed_text = transcribe_bytes_with_whisper(audio_file.getvalue(), mime_type)
                                st.session_state['last_transcript'] = transcribed_text
                                st.session_state['sim_audio_upload_key'] += 1
                                st.success(L.get("whisper_success"))
                                st.rerun() 
                            except Exception as e: st.error(f"음성 전사 처리 중 오류 발생: {e}"); st.session_state['last_transcript'] = ""

                agent_response = col_text_area.text_area(
                    L["agent_response_placeholder"], value=st.session_state['last_transcript'], key="agent_response_area_text", height=150
                )
                
                # JS Enter Key Listener
                st.components.v1.html("""<script>const textarea = document.querySelector('textarea[key="agent_response_area_text"]'); const button = document.querySelector('button[key="send_agent_response_sim"]'); if (textarea && button) { textarea.addEventListener('keydown', function(event) { if (event.key === 'Enter' && (!event.shiftKey && !event.ctrlKey)) { event.preventDefault(); button.click(); } }); }</script>""", height=0, width=0)

                if st.button(L["send_response_button"], key="send_agent_response_sim"): 
                    if agent_response.strip():
                        st.session_state['last_transcript'] = ""
                        st.session_state.simulator_messages.append({"role": "agent_response", "content": agent_response})
                        st.session_state.simulator_memory.chat_memory.add_user_message(agent_response)
                        save_simulation_history(db, st.session_state.customer_query_text_area, customer_type_display, st.session_state.simulator_messages)
                        st.rerun()
                    else: st.warning(L.get("empty_response_warning"))
            
            if last_role == "agent_response":
                col_end, col_next = st.columns([1, 2])
                
                if col_end.button(L["button_end_chat"], key="end_chat_sim"): 
                    closing_messages = get_closing_messages(current_lang_key)
                    st.session_state.simulator_messages.append({"role": "supervisor", "content": closing_messages["additional_query"]})
                    st.session_state.simulator_memory.chat_memory.add_ai_message(closing_messages["additional_query"])
                    st.session_state.simulator_messages.append({"role": "system_end", "content": closing_messages["chat_closing"]})
                    st.session_state.simulator_memory.chat_memory.add_ai_message(closing_messages["chat_closing"])
                    st.session_state.is_chat_ended = True
                    save_simulation_history(db, st.session_state.customer_query_text_area, customer_type_display, st.session_state.simulator_messages)
                    st.rerun()

                if col_next.button(L["request_rebuttal_button"], key="request_rebuttal_sim"):
                    if not os.environ.get("GEMINI_API_KEY"): st.warning("API Key가 없어 LLM 시뮬레이션 불가"); st.stop()
                    
                    next_reaction_prompt = f"""Analyze the entire chat history. Roleplay as the customer ({customer_type_display}). Based on the agent's last message, generate ONE of the following responses... The response MUST be strictly in {LANG[current_lang_key]['lang_select']}."""
                    
                    with st.spinner(L["response_generating"]):
                        try:
                            customer_reaction = st.session_state.simulator_chain.predict(input=next_reaction_prompt)
                            positive_keywords = ["감사", "thank you", "ありがとう", L['customer_positive_response'].lower().split('/')[-1].strip()]
                            is_positive_close = any(keyword in customer_reaction.lower() for keyword in positive_keywords)
                            
                            if is_positive_close:
                                role = "customer_end"
                                st.session_state.simulator_messages.append({"role": role, "content": customer_reaction})
                                st.session_state.simulator_memory.chat_memory.add_ai_message(customer_reaction)
                                st.session_state.simulator_messages.append({"role": "supervisor", "content": L["customer_closing_confirm"]})
                                st.session_state.simulator_memory.chat_memory.add_ai_message(L["customer_closing_confirm"])
                            else:
                                role = "customer_rebuttal"
                                st.session_state.simulator_messages.append({"role": role, "content": customer_reaction})
                                st.session_state.simulator_memory.chat_memory.add_ai_message(customer_reaction)
                                 
                            save_simulation_history(db, st.session_state.customer_query_text_area, customer_type_display, st.session_state.simulator_messages)
                            st.rerun()
                        except Exception as e: st.error(f"LLM 응답 생성 중 오류 발생: {e}")
    else:
        st.error(L["llm_error_init"])

elif feature_selection == L["rag_tab"]:
    # (RAG Chatbot UI logic remains the same)
    st.header(L["rag_header"])
    st.markdown(L["rag_desc"])
    if st.session_state.get('is_rag_ready', False) and st.session_state.get('conversation_chain'):
        if "messages" not in st.session_state: st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])
        if prompt := st.chat_input(L["rag_input_placeholder"]):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(L["response_generating"]):
                    try:
                        response = st.session_state.conversation_chain.invoke({"question":prompt})
                        answer = response.get('answer', '응답을 생성할 수 없습니다.' if st.session_state.language == 'ko' else 'Could not generate response.')
                        st.markdown(answer)
                        st.session_state.messages.append({"role":"assistant","content":answer})
                    except Exception as e: st.error(f"챗봇 오류: {e}"); st.session_state.messages.append({"role":"assistant","content":"오류 발생" if st.session_state.language == 'ko' else "An error occurred"})
    else: st.warning(L["warning_rag_not_ready"])

elif feature_selection == L["content_tab"]:
    # (Custom Content Generation UI logic remains the same)
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
                    full_prompt = f"""You are a professional AI coach at the {level} level. Please generate exactly 10 multiple-choice questions about the topic in {target_lang}. Your entire response MUST be a valid JSON object wrapped in
                else:
                    display_type_text = L["content_options"][L["content_options"].index(content_type_display)]
                    full_prompt = f"""You are a professional AI coach at the {level} level. Please generate clear and educational content in the requested {display_type_text} format based on the topic. The response MUST be strictly in {target_lang}. Topic: {topic}. Requested Format: {display_type_text}"""
                
                with st.spinner(f"Generating {content_type_display} for {topic}..."):
                    quiz_data_raw = None
                    try:
                        response = st.session_state.llm.invoke(full_prompt)
                        quiz_data_raw = response.content
                        st.session_state.quiz_data_raw = quiz_data_raw
                        if content_type == 'quiz':
                            quiz_data = clean_and_load_json(quiz_data_raw)
                            if quiz_data and 'quiz_questions' in quiz_data:
                                st.session_state.quiz_data = quiz_data
                                st.session_state.current_question = 0
                                st.session_state.quiz_submitted = False
                                st.session_state.quiz_results = [None] * len(quiz_data.get('quiz_questions',[]))
                                st.success(f"**{topic}** - **{content_type_display}** Result:")
                            else: st.error(L["quiz_error_llm"]); st.markdown(f"**{L['quiz_original_response']}**:"); st.code(quiz_data_raw, language="json")
                        else: st.success(f"**{topic}** - **{content_type_display}** Result:"); st.markdown(response.content)
                    except Exception as e: st.error(f"Content Generation Error: {e}"); 
            else: st.warning(L["warning_topic"])
    else: st.error(L["llm_error_init"])
    is_quiz_ready = content_type == 'quiz' and 'quiz_data' in st.session_state and st.session_state.quiz_data
    if is_quiz_ready and st.session_state.get('current_question', 0) < len(st.session_state.quiz_data.get('quiz_questions', [])):
        render_interactive_quiz(st.session_state.quiz_data, st.session_state.language)

elif feature_selection == L["lstm_tab"]:
    # (LSTM UI logic remains the same)
    st.header(L["lstm_header"])
    st.markdown(L["lstm_desc"])
    if st.button(L["lstm_rerun_button"], key="rerun_lstm", on_click=force_rerun_lstm): pass
    try:
        model, data = load_or_train_lstm()
        look_back = 5
        X_input = np.reshape(data[-look_back:], (1, look_back, 1))
        predicted_score = model.predict(X_input, verbose=0)[0][0]
        st.markdown("---")
        st.subheader(L["lstm_result_header"])
        col_score, col_chart = st.columns([1, 2])
        with col_score:
            st.metric(L["lstm_score_metric"], f"{predicted_score:.1f}{'점' if st.session_state.language == 'ko' else ''}")
            st.info(L["lstm_score_info"].format(predicted_score=predicted_score))
        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(data, label='Past Scores', marker='o')
            ax.plot(len(data), predicted_score, label='Predicted Next Score', marker='*', color='red', markersize=10)
            ax.set_title(L["lstm_header"])
            ax.set_xlabel(f"Time ({L.get('score', 'Score')} attempts)")
            ax.set_ylabel(f"{L.get('score', 'Score')} (0-100)")
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"LSTM 모델 실행 중 오류가 발생했습니다. (오류 메시지: {e})")  
