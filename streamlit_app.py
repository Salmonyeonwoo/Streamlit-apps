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

# LangChain Imports
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate 

# -----------------------------
# Config & I18N (다국어 지원)
# -----------------------------
DEFAULT_LANG = "ko"
if 'language' not in st.session_state:
    st.session_state.language = DEFAULT_LANG

LANG = {
    "ko": {
        "title": "개인 맞춤형 AI 학습 코치 (음성 및 DB 통합)",
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
        "quiz_error_llm": "퀴즈 생성 실패: LLM이 올바른 JSON 형식을 반환하지 않았습니다.",
        "quiz_original_response": "LLM 원본 응답",
        "firestore_loading": "데이터베이스에서 RAG 인덱스 로드 중...",
        "firestore_no_index": "데이터베이스에서 기존 RAG 인덱스를 찾을 수 없습니다. 파일을 업로드하여 새로 만드세요。", 
        "db_save_complete": "(DB 저장 완료)", 
        "data_analysis_progress": "자료 분석 및 학습 DB 구축 중...", 
        "response_generating": "답변 생성 중...", 
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
        "simulation_no_key_warning": "⚠️ API Key가 없는 경우, 응답 생성은 실행되지 않습니다.",
        "simulation_advice_header": "AI의 응대 가이드라인",
        "simulation_draft_header": "추천 응대 초안",
        "button_listen_audio": "음성으로 듣기",
        "tts_status_ready": "음성으로 듣기 준비됨",
        "tts_status_generating": "오디오 생성 중...",
        "tts_status_success": "✅ 오디오 재생 완료!",
        "tts_status_error": "❌ TTS 오류 발생",
        "history_expander_title": "📝 이전 상담 이력 로드 (최근 10개)", 
        "initial_query_sample": "프랑스 파리에 도착했는데, 클룩에서 구매한 eSIM이 활성화가 안 됩니다. 연결이 안 돼서 너무 곤란합니다. 어떻게 해야 하나요?", 
        "button_mic_input": "🎙 음성 입력",
        "prompt_customer_end": "고객님의 추가 문의 사항이 없어, 이 상담 채팅을 종료하겠습니다。",
        "prompt_survey": "고객 문의 센터에 연락 주셔서 감사드립니다. 추가 문의 사항이 있으시면 언제든지 연락 주십시오。",
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
        "delete_confirm_message": "정말로 모든 상담 이력을 삭제하시겠습니까? 되돌릴 수 없습니다。", 
        "delete_confirm_yes": "예, 삭제합니다", 
        "delete_confirm_no": "아니오, 유지합니다", 
        "delete_success": "✅ 모든 상담 이력 삭제 완료!",
        "deleting_history_progress": "이력 삭제 중...", 
        "search_history_label": "이력 키워드 검색", 
        "date_range_label": "날짜 범위 필터", 
        "no_history_found": "검색 조건에 맞는 이력이 없습니다。",
        
        # ⭐ 음성 기록 통합 관련 키 (Voice/GCS)
        "voice_rec_header": '음성 기록 & 관리',
        "record_help": '마이크 버튼을 눌러 녹음하거나 파일을 업로드하세요.',
        "uploaded_file": '오디오 파일 업로드',
        "rec_list_title": '저장된 음성 기록 (Whisper/GCS)',
        "transcribe_btn": '전사(Whisper)',
        "save_btn": '음성 기록 저장',
        "transcribing": '음성 전사 중...',
        "transcript_result": '전사 결과:',
        "transcript_text": '전사 텍스트',
        "openai_missing": 'OpenAI API Key가 없습니다. Secrets에 OPENAI_API_KEY를 설정하세요.',
        "whisper_client_error": "❌ 오류: Whisper API Client가 초기화되지 않았습니다. Secrets에 OPENAI_API_KEY를 설정했는지 확인하세요.",
        "whisper_auth_error": "❌ Whisper API 인증 실패: API Key를 확인하세요.",
        "whisper_format_error": "❌ 오류: 지원하지 않는 오디오 형식입니다.",
        "whisper_success": "✅ 음성 전사 완료! 텍스트 창을 확인하세요。",
        "playback": '녹음 재생',
        "retranscribe": '재전사',
        "delete": '삭제',
        "no_records": '저장된 음성 기록이 없습니다.',
        "gcs_missing": 'GCS 버킷이 설정되어 있지 않습니다. Secrets에 GCS_BUCKET_NAME을 추가하세요.',
        "saved_success": '저장 완료!',
        "delete_confirm_rec": '정말로 이 음성 기록을 삭제하시겠습니까? GCS 파일도 삭제됩니다.',
        "gcs_init_fail": 'GCS 초기화 실패. 권한 및 버킷 이름을 확인하세요.',
        "firebase_init_fail": 'Firebase Admin 초기화 실패.',
        "upload_fail": 'GCS 오디오 파일 업로드 실패',
        "gcs_not_conf": 'GCS 미설정 또는 오디오 없음',
        "gcs_playback_fail": '오디오 재생 실패',
        "gcs_no_audio": '오디오 파일 없음 (GCS 미설정)',
        "error": '오류:',
        "firestore_no_db_connect": "❌ DB 연결 실패: 상담 이력 저장 불가",
        "save_history_success": "✅ 상담 이력이 저장되었습니다.",
        "save_history_fail": "❌ 상담 이력 저장 실패",
        "delete_fail": "삭제 실패",
    },
    "en": {
        "title": "Personalized AI Study Coach (Voice & DB Integration)",
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
        "simulation_no_key_warning": "⚠️ API Key is missing. Response generation cannot proceed.",
        "simulation_advice_header": "AI Response Guidelines",
        "simulation_draft_header": "Recommended Response Draft",
        "button_listen_audio": "Listen to Audio",
        "tts_status_ready": "Ready to listen",
        "tts_status_generating": "Generating audio...",
        "tts_status_success": "✅ Audio playback complete!",
        "tts_status_error": "❌ TTS API error occurred",
        "history_expander_title": "📝 Load Previous Simulation History (Last 10)", 
        "initial_query_sample": "I arrived in Paris, France, but the eSIM I bought from Klook won't activate. I'm really struggling to get connected. What should I do?", 
        "button_mic_input": "🎙 Voice Input",
        "prompt_customer_end": "As there are no further inquiries, we will now end this chat session.",
        "prompt_survey": "Thank you for contacting our Customer Support Center. Please feel free to contact us anytime if you have any additional questions.",
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
        "delete_history_button": "❌ Delete All History", 
        "delete_confirm_message": "Are you sure you want to delete ALL simulation history? This action cannot be undone.", 
        "delete_confirm_yes": "Yes, Delete", 
        "delete_confirm_no": "No, Keep", 
        "delete_success": "✅ Successfully deleted!", 
        "deleting_history_progress": "Deleting history...", 
        "search_history_label": "Search History by Keyword", 
        "date_range_label": "Date Range Filter", 
        "no_history_found": "No history found matching the criteria.",

        # ⭐ 음성 기록 통합 관련 키 (Voice/GCS)
        "voice_rec_header": 'Voice Record & Management',
        "record_help": 'Press the microphone button to record or upload a file.',
        "uploaded_file": 'Upload Audio File',
        "rec_list_title": 'Saved Voice Records (Whisper/GCS)',
        "transcribe_btn": 'Transcribe (Whisper)',
        "save_btn": 'Save Voice Record',
        "transcribing": 'Transcribing voice...',
        "transcript_result": 'Transcription Result:',
        "transcript_text": 'Transcribed Text',
        "openai_missing": 'OpenAI API Key is missing. Set OPENAI_API_KEY in Secrets.',
        "whisper_client_error": "❌ Error: Whisper API Client not initialized. Check OPENAI_API_KEY in Secrets.",
        "whisper_auth_error": "❌ Whisper API Authentication failed: Check your API Key.",
        "whisper_format_error": "❌ Error: Unsupported audio format.",
        "whisper_success": "✅ Voice transcription complete! Check the text box.",
        "playback": 'Playback Recording',
        "retranscribe": 'Re-transcribe',
        "delete": 'Delete',
        "no_records": 'No voice records saved yet.',
        "gcs_missing": 'GCS bucket is not configured. Add GCS_BUCKET_NAME to Secrets.',
        "saved_success": 'Save successful!',
        "delete_confirm_rec": 'Are you sure you want to delete this voice record? The GCS file will also be deleted.',
        "gcs_init_fail": 'GCS initialization failed. Check permissions and bucket name.',
        "firebase_init_fail": 'Firebase Admin initialization failed.',
        "upload_fail": 'GCS audio file upload failed',
        "gcs_not_conf": 'GCS not configured or audio not available',
        "gcs_playback_fail": 'Audio playback failed',
        "gcs_no_audio": 'No audio file (GCS not configured)',
        "error": 'Error:',
        "firestore_no_db_connect": "❌ DB 연결 실패: 상담 이력 저장 불가",
        "save_history_success": "✅ 상담 이력이 저장되었습니다.",
        "save_history_fail": "❌ 상담 이력 저장 실패",
        "delete_fail": "삭제 실패",
    },
    "ja": {
        "title": "パーソナライズAI学習コーチ (音声・DB統合)",
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
        "db_save_complete": "(DB保存完了)", 
        "data_analysis_progress": "資料分析および学習DB構築中...", 
        "response_generating": "応答生成中...", 
        "lstm_result_header": "達成度予測結果",
        "lstm_score_metric": "現在の予測達成度",
        "lstm_score_info": "次のクイズの推定スコアは約 **{predicted_score:.1f}点**です。学習の成果を維持または向上させてください！",
        "lstm_rerun_button": "新しい仮想データで予測",

        # ⭐ シミュレーター関連テキスト
        "simulator_header": "AI顧客対応シミュレーター",
        "simulator_desc": "難しい顧客の問い合わせに対して、AIによる対応案とガイドラインを提供します。",
        "customer_query_label": "顧客の問い合わせ内容（リンク任意）",
        "customer_type_label": "顧客の傾向",
        "customer_type_options": ["一般的な問い合わせ", "手ごわい顧客", "非常に不満な顧客"],
        "button_simulate": "対応アドバイスを要求",
        "simulation_warning_query": "顧客の問い合わせ内容を入力してください。",
        "simulation_no_key_warning": "⚠️ APIキーが不足しています。応答の生成は続行できません。",
        "simulation_advice_header": "AI対応ガイドライン",
        "simulation_draft_header": "推奨される対応草案",
        "button_listen_audio": "音声で聞く",
        "tts_status_ready": "音声再生の準備ができました",
        "tts_status_generating": "音声生成中...",
        "tts_status_success": "✅ 音声再生完了!",
        "tts_status_error": "❌ TTS APIエラーが発生しました",
        "history_expander_title": "📝 以前の対応履歴をロード (最新 10件)", 
        "initial_query_sample": "フランスのパリに到着しましたが、Klookで購入したeSIMがアクティベートできません。接続できなくて困っています。どうすればいいですか？", 
        "button_mic_input": "🎙 音声入力",
        "prompt_customer_end": "お客様からの追加のお問い合わせがないため、本チャットサポートを終了させていただきます。",
        "prompt_survey": "お問い合わせいただき、誠にありがとうございました。追加のご質問がございましたらいつでもご連絡ください。",
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
        "delete_success": "✅ 削除が完了されました!", 
        "deleting_history_progress": "履歴削除中...", 
        "search_history_label": "履歴キーワード検索", 
        "date_range_label": "日付範囲フィルター", 
        "no_history_found": "検索条件に一致する履歴はありません。",

        # ⭐ 음성 기록 통합 관련 키 (Voice/GCS)
        "voice_rec_header": '音声記録と管理',
        "record_help": 'マイクボタンを押して録音するか、ファイルをアップロードしてください。',
        "uploaded_file": '音声ファイルをアップロード',
        "rec_list_title": '保存された音声記録 (Whisper/GCS)',
        "transcribe_btn": '転写(Whisper)',
        "save_btn": '音声記録を保存',
        "transcribing": '音声転写中...',
        "transcript_result": '転写結果:',
        "transcript_text": '転写テキスト',
        "openai_missing": 'OpenAI APIキーがありません。SecretsにOPENAI_API_KEYを設定してください。',
        "whisper_client_error": "❌ エラー: Whisper API Clientが初期化されていません。SecretsのOPENAI_API_KEYを確認してください。",
        "whisper_auth_error": "❌ Whisper API認証失敗: APIキーを確認してください。",
        "whisper_format_error": "❌ エラー: サポートされていない音声フォーマットです。",
        "whisper_success": "✅ 音声転写完了！テキストボックスをご確認ください。",
        "playback": '録音再生',
        "retranscribe": '再転写',
        "delete": '削除',
        "no_records": '保存された音声記録はありません。',
        "gcs_missing": 'GCSバケットが設定されていません。SecretsにGCS_BUCKET_NAMEを追加してください。',
        "saved_success": '保存が完了しました！',
        "delete_confirm_rec": '本当にこの音声記録を削除してもよろしいですか？GCSファイルも削除されます。',
        "gcs_init_fail": 'GCSの初期化に失敗しました。権限とバケット名を確認してください。',
        "firebase_init_fail": 'Firebase Adminの初期化に失敗しました。',
        "upload_fail": 'GCS音声ファイルのアップロードに失敗しました',
        "gcs_not_conf": 'GCSが未設定か、音声が利用できません',
        "gcs_playback_fail": '音声再生に失敗しました',
        "gcs_no_audio": '音声ファイルなし (GCS未設定)',
        "error": 'エラー:',
        "firestore_no_db_connect": "❌ DB 연결 실패: 상담 이력 저장 불가",
        "save_history_success": "✅ 상담 이력이 저장되었습니다.",
        "save_history_fail": "❌ 상담 이력 저장 실패",
        "delete_fail": "削除失敗",
    }
}


# -----------------------------
# 1. Firebase Admin, GCS, OpenAI Initialization
# -----------------------------

def _get_admin_credentials():
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
        get_app()
    except ValueError:
        pass
    else:
        try:
            db_client = firestore.client()
            return db_client, "✅ Firestore DB 클라이언트 준비 완료"
        except Exception as e:
            return None, f"🔥 Firebase 클라이언트 로드 실패: {e}"

    try:
        cred = credentials.Certificate(sa_info)
        initialize_app(cred)
        db_client = firestore.client()
        return db_client, "✅ Firestore DB 클라이언트 준비 완료"
    except Exception as e:
        return None, f"🔥 {L['firebase_init_fail']}: 서비스 계정 정보 문제. 오류: {e}"


@st.cache_resource
def init_gcs_client(L):
    """GCS 클라이언트를 초기화하고 클라이언트 객체와 에러 메시지를 반환합니다."""
    sa = _load_service_account_from_secrets()
    gcs_bucket_name = st.secrets.get('GCS_BUCKET_NAME') or os.environ.get('GCS_BUCKET_NAME')
    
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

def get_gcs_bucket_name():
    return st.secrets.get('GCS_BUCKET_NAME') or os.environ.get('GCS_BUCKET_NAME')

# -----------------------------
# 2. GCS, Firestore, Whisper Helpers (통합된 함수)
# -----------------------------

def upload_audio_to_gcs(bucket_name: str, blob_name: str, audio_bytes: bytes, content_type: str = 'audio/webm'):
    L = LANG[st.session_state.language]
    gcs_client = init_gcs_client(L)[0] # 클라이언트 객체만 가져옴
    if not gcs_client:
        raise RuntimeError(L['gcs_not_conf'])
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(audio_bytes, content_type=content_type)
    return f'gs://{bucket_name}/{blob_name}' 

def download_audio_from_gcs(bucket_name: str, blob_name: str) -> bytes:
    L = LANG[st.session_state.language]
    gcs_client = init_gcs_client(L)[0] # 클라이언트 객체만 가져옴
    if not gcs_client:
        raise RuntimeError(L['gcs_not_conf'])
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    except NotFound:
        raise FileNotFoundError(f"GCS Blob not found: {blob_name}")
    except Exception as e:
        raise RuntimeError(f"{L['gcs_playback_fail']}: {e}")

def save_audio_record(db, bucket_name, audio_bytes: bytes, filename: str, transcript_text: str, meta: dict = None, mime_type: str = 'audio/webm'):
    L = LANG[st.session_state.language]
    if not db:
        raise RuntimeError('Firestore not initialized')

    ts = datetime.now(timezone.utc)
    doc_ref = db.collection('voice_records').document()
    blob_name = f"voice_records/{doc_ref.id}/{filename}"

    gcs_path = None
    gcs_client = init_gcs_client(L)[0]
    if bucket_name and gcs_client:
        try:
            gcs_path = upload_audio_to_gcs(bucket_name, blob_name, audio_bytes, mime_type)
        except Exception as e:
            st.warning(f"{L['upload_fail']}: {e}")
            gcs_path = None
    else:
        st.warning(L['gcs_missing'])

    data = {
        'created_at': ts,
        'filename': filename,
        'size': len(audio_bytes),
        'gcs_path': gcs_path,
        'transcript': transcript_text,
        'mime_type': mime_type, 
        'language': st.session_state.language,
        'meta': meta or {}
    }

    doc_ref.set(data)
    return doc_ref.id

def delete_audio_record(db, bucket_name, doc_id: str):
    L = LANG[st.session_state.language]
    doc_ref = db.collection('voice_records').document(doc_id)
    doc = doc_ref.get()
    if not doc.exists:
        return False
    data = doc.to_dict()
    
    gcs_client = init_gcs_client(L)[0]
    # delete GCS blob
    try:
        if data.get('gcs_path') and gcs_client and bucket_name:
            blob_name = data['gcs_path'].split(f'gs://{bucket_name}/')[-1]
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()
    except Exception as e:
        st.warning(f"GCS delete warning: {e}")
    
    # delete firestore doc
    doc_ref.delete()
    return True

def transcribe_bytes_with_whisper(audio_bytes: bytes, mime_type: str = 'audio/webm'):
    L = LANG[st.session_state.language]
    openai_client = init_openai_client(L)[0] # 클라이언트 객체만 가져옴
    if openai_client is None:
        raise RuntimeError(L['openai_missing'])
    
    # Determine file extension
    ext = mime_type.split('/')[-1].lower() if '/' in mime_type else 'webm'
    
    # write to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    
    try:
        with open(tmp.name, 'rb') as af:
            res = openai_client.audio.transcriptions.create(
                model='whisper-1', 
                file=af,
                response_format='text'
            )
        return res.strip() or ''
    except Exception as e:
        raise RuntimeError(f"{L['error']} Whisper: {e}")
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass


# -----------------------------
# 3. Firestore/RAG/LLM Helpers (기존 코드 유지)
# -----------------------------

# (나머지 Helper 함수들은 st.error/success를 직접 호출하지 않으므로 변경 없음)
# _get_admin_credentials 함수는 위에서 수정된 내용이 반영되어야 함.

def save_simulation_history(db, initial_query, customer_type, messages):
    L = LANG[st.session_state.language]
    if not db: 
        st.sidebar.warning(L.get("firestore_no_db_connect", "❌ DB 연결 실패: 상담 이력 저장 불가"))
        return False
    history_data = [{k: v for k, v in msg.items()} for msg in messages]
    data = {
        "initial_query": initial_query,
        "customer_type": customer_type,
        "messages": history_data,
        "language_key": st.session_state.language, 
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    try:
        db.collection("simulation_histories").add(data)
        st.sidebar.success(L.get("save_history_success", "✅ 상담 이력이 저장되었습니다."))
        return True
    except Exception as e:
        st.sidebar.error(f"❌ {L.get('save_history_fail', '상담 이력 저장 실패')}: {e}")
        return False

# (나머지 LangChain 및 Utility 함수들은 변경 없음)
# ...

# -----------------------------
# 5. Core Initialization & Session State
# -----------------------------

# --- Session State 초기화 (st.set_page_config 이전에 실행 가능) ---
if 'language' not in st.session_state: st.session_state.language = DEFAULT_LANG
L = LANG[st.session_state.language] 

# 모든 session state 초기화
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


# --- 클라이언트 초기화 ---
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
    # (RAG Index loading logic remains the same)
    loaded_index = load_index_from_firestore(st.session_state.firestore_db, st.session_state.embeddings)
    if loaded_index:
        st.session_state.conversation_chain = get_rag_chain(loaded_index)
        st.session_state.is_rag_ready = True
        st.session_state.firestore_load_success = True
    else:
        st.session_state.firestore_load_success = False


# -----------------------------
# 6. Streamlit UI (스크립트의 첫 번째 UI 출력 명령어)
# -----------------------------

# ⭐ 이 줄이 스크립트에서 첫 번째 Streamlit 명령어여야 합니다.
st.set_page_config(page_title=L["title"], layout="wide")

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
    
    # --- 초기화 상태 표시 (st.set_page_config 이후에 안전하게 출력) ---
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
# 7. 기능별 페이지 구현
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
