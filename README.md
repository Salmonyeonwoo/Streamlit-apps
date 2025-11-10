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

Script for Japansese RAG version

<img width="1872" height="730" alt="image" src="https://github.com/user-attachments/assets/d4aaf6bc-168c-4ed9-8bb9-38dde0759a6c" />

Script for English RAG version

<img width="1890" height="830" alt="image" src="https://github.com/user-attachments/assets/eb94594a-ff80-4609-b2af-ad2a3f9c0a7d" />

Utilizing LSTM

<img width="1836" height="751" alt="image" src="https://github.com/user-attachments/assets/7f8c6ffd-a0f0-40da-994d-6f06859e2520" />

Movies for RAG and LSTM via my streamlit app

https://www.youtube.com/watch?v=MOQ2qD7Ws-E








