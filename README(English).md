🚀 AI Study Coach Portfolio App

This project delivers a personalized learning experience through an AI coach application. It combines RAG (Retrieval-Augmented Generation) technology with an LSTM prediction model to support customized user study.

Key Features

RAG Knowledge Chatbot: Answers user questions based on uploaded documents (PDF, TXT, HTML).

Custom Content Generation: Generates summaries, quizzes, and practical example ideas tailored to specific topics and difficulty levels using the Gemini LLM.

LSTM Achievement Prediction: Provides motivation by predicting future learning achievement based on hypothetical historical study data.

Technical Stack

Frontend/App Framework: Streamlit

Backend/ML/LLM: Google Gemini API, LangChain

Vector Database: FAISS (Index stored in Firestore)

Persistent Storage: Google Cloud Firestore (Permanent data management via Admin SDK)

Deployment: Streamlit Cloud

🔑 Deployment Setup (Secrets)

This app requires the following two Secrets keys for deployment on Streamlit Cloud. (Sensitive information is not included in this repository.)

GEMINI_API_KEY: The personal key for making Gemini API calls.

FIREBASE_SERVICE_ACCOUNT_JSON: The complete Service Account JSON string for accessing Firestore via the Admin SDK.
