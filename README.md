# 📄 SpeakDoc: Talk to your PDFs, Get Instant Answers

**SpeakDoc** is a dynamic Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, and **Weaviate**, featuring an integrated **voice assistant** for natural language interaction.

It allows users to upload PDF documents, ask questions about their content (via text or voice), and receive **spoken and textual answers**.

---
![SpeakDoc Application Screenshot](path/to/your/screenshot.png)

## ✨ Features

- **PDF Document Upload:** Easily upload your PDF documents to create a custom knowledge base.
- **Automated Document Processing:**
  - Loads PDF content.
  - Splits documents into manageable chunks.
  - Generates embeddings for efficient retrieval.
- **Dynamic Knowledge Base:** Utilizes Weaviate as a vector database to store and retrieve document embeddings.
- **Retrieval-Augmented Generation (RAG):** Uses Google's Gemini-1.5-Flash LLM with LangChain to answer queries based on uploaded documents.
- **Voice Input (Speech-to-Text):**
  - Click a microphone icon to speak your questions.
  - Uses OpenAI's Whisper model for accurate speech transcription.
  - Automatically populates the query input box.
- **Voice Output (Text-to-Speech):** Answers are spoken aloud using gTTS (Google Text-to-Speech).
- **Interactive Streamlit UI:** Clean and user-friendly interface for seamless interaction.

---

## ⚙️ Tech Stack

- **Frontend:** Streamlit  
- **Backend Orchestration:** LangChain  
- **Vector Database:** Weaviate  
- **LLM:** Gemini-1.5-Flash (via `langchain-google-genai`)  
- **Embeddings:** Sentence Transformers (`all-mpnet-base-v2`)  
- **PDF Loader:** `PyPDFLoader`  
- **Text Splitting:** `RecursiveCharacterTextSplitter`  
- **Speech-to-Text:** OpenAI Whisper  
- **Text-to-Speech:** gTTS  
- **Audio Playback:** Pygame  
- **Audio Recording:** PyAudio  
- **Environment Management:** `python-dotenv`

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
