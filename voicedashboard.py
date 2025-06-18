import streamlit as st

# --- Your Project's Dashboard.py Code (as a string) ---
# This multiline string contains the *entire* content of your dashboard.py file.
# It will be displayed using st.code().
# Ensure all backslashes in paths are escaped (e.g., C:\\Users\\...)
# And double-check any internal quotes within the original string are also escaped if necessary,
# although Streamlit's markdown often handles this if the outer string uses triple quotes.
DASHBOARD_CODE = """
# --- Important Library Imports ---
import streamlit as st
import os
from dotenv import load_dotenv
import tempfile # For handling temporary uploaded files
import hashlib # For creating a hash of the uploaded file for tracking

# LangChain components
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Weaviate client and configuration
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.collections.classes.config import Property, DataType, Vectorizers

# Voice Assistance Libraries
from gtts import gTTS
import io
import pygame
import whisper # This is the library with known compatibility issues on some Python versions
import pyaudio
import wave

# --- Environment Variable Loading ---
load_dotenv(dotenv_path=r"C:\\Users\\PMLS\\Desktop\\Python\\FYP\\notebooks\\rag.env") # Assumes rag.env is in the same directory

# Retrieve API keys
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Dynamic RAG Document Query App", layout="wide")
st.title("📄 Dynamic RAG Document Query App")

st.markdown(\"\"\"
This application allows you to upload a PDF document, which will then be processed
to create a searchable knowledge base using a Retrieval-Augmented Generation (RAG) model.
You can then ask questions about the content of your uploaded document, either by typing or by voice.
\"\"\")

# --- Tech Stack Description ---
st.header("⚙️ Tech Stack Overview")
st.markdown(\"\"\"
This application leverages the following technologies:
-   **Streamlit:** For building the interactive web user interface.
-   **LangChain:** A framework for developing applications powered by language models.
    -   `PyPDFLoader`: To load PDF documents.
    -   `RecursiveCharacterTextSplitter`: To break down documents into manageable chunks.
    -   `SentenceTransformerEmbeddings`: To convert text chunks into numerical vectors (embeddings).
    -   `WeaviateVectorStore`: To store and retrieve document embeddings efficiently.
    -   `ChatGoogleGenerativeAI`: To interface with Google's Gemini-1.5-Flash LLM for generating answers.
    -   `RetrievalQA`: To orchestrate the RAG process (retrieve relevant chunks and generate answers).
-   **Weaviate:** A vector database that stores the embeddings and enables fast similarity searches.
-   **python-dotenv:** For securely loading environment variables (API keys).
-   **gTTS:** Google Text-to-Speech for generating audio responses.
-   **Pygame:** For playing audio responses.
-   **Whisper:** OpenAI's robust speech-to-text model for transcribing user queries.
-   **PyAudio:** Python bindings for PortAudio, used for audio input.
\"\"\")

# --- Session State Management ---
# Initialize session state variables to avoid re-running expensive operations
if 'documents_loaded_flag' not in st.session_state:
    st.session_state.documents_loaded_flag = False
if 'weaviate_setup_done_flag' not in st.session_state:
    st.session_state.weaviate_setup_done_flag = False
if 'rag_initialized_flag' not in st.session_state:
    st.session_state.rag_initialized_flag = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'weaviate_client' not in st.session_state:
    st.session_state.weaviate_client = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "DynamicKnowledgeBase" # Default collection name
if 'current_uploaded_file_hash' not in st.session_state:
    st.session_state.current_uploaded_file_hash = None
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'query_text_input' not in st.session_state:
    st.session_state.query_text_input = "" # To hold the text input value for transcription


# --- Initialize Pygame Mixer (Run once) ---
if not pygame.mixer.get_init():
    pygame.mixer.init()

# --- Helper Functions for Voice Assistance ---

@st.cache_resource
def load_whisper_model():
    \"\"\"Loads the Whisper model (cached to avoid reloading).\"\"\"
    st.info("Loading Whisper ASR model... This may take a moment.")
    model = whisper.load_model("small")
    st.success("Whisper model loaded.")
    return model

def text_to_speech(text, lang='en'):
    \"\"\"Converts text to speech and plays it.\"\"\"
    try:
        tts = gTTS(text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        
        # Wait until audio finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10) # Small delay to prevent busy-waiting
        fp.close() # Close the BytesIO object
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")

def record_audio(record_seconds=10):
    \"\"\"Records audio from the microphone and returns the file path.\"\"\"
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    st.toast("🎙️ Recording... Speak now!", icon="🎤")
    frames = []
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        filename = temp_audio.name
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    st.toast("Recording finished!", icon="✅")
    return filename

def transcribe_audio(audio_filepath, model, lang='en'):
    \"\"\"Transcribes an audio file using the Whisper model.\"\"\"
    try:
        result = model.transcribe(audio_filepath, language=lang, fp16=False)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None
    finally:
        if os.path.exists(audio_filepath):
            os.remove(audio_filepath) # Clean up the temporary audio file

# --- Function to setup Weaviate and load data ---
def setup_weaviate_and_load_data(docs, embeddings, collection_name, embedding_model):
    \"\"\"Sets up Weaviate client, creates collection, and loads document embeddings.\"\"\"
    if st.session_state.weaviate_client is None or not st.session_state.weaviate_client.is_ready():
        st.info("Step 4: Setting up Weaviate client...")
        try:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
            )
            if client.is_ready():
                st.session_state.weaviate_client = client
                st.success("Weaviate client connected successfully.")
            else:
                st.error("Weaviate client is not ready. Check your connection details.")
                return False
        except Exception as e:
            st.error(f"Error connecting to Weaviate: {e}. Please check your WEAVIATE_URL and WEAVIATE_API_KEY.")
            return False

    st.info(f"Step 5: Creating or getting Weaviate collection '{collection_name}'...")
    try:
        if st.session_state.weaviate_client.collections.exists(collection_name):
            st.warning(f"Collection '{collection_name}' already exists. Deleting to update with new document.")
            st.session_state.weaviate_client.collections.delete(collection_name)
            st.success(f"Collection '{collection_name}' deleted.")

        collection = st.session_state.weaviate_client.collections.create(
            name=collection_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
            ],
            vectorizer_config={"vectorizer": Vectorizers.NONE}
        )
        st.success(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        st.error(f"Error creating/getting collection '{collection_name}': {e}")
        return False

    st.info(f"Step 6: Loading {len(docs)} chunks into Weaviate collection '{collection_name}'...")
    try:
        collection = st.session_state.weaviate_client.collections.get(collection_name)
        with collection.batch.fixed_size(batch_size=100) as batch:
            for i, (doc_item, embedding_vector) in enumerate(zip(docs, embeddings)):
                batch.add_object(
                    properties={"text": doc_item.page_content},
                    vector=embedding_vector
                )
        st.success(f"Successfully loaded all {len(docs)} chunks into Weaviate.")
        return True
    except Exception as e:
        st.error(f"Error loading data into Weaviate: {e}")
        return False

# --- Function to initialize RAG components ---
def initialize_rag_components(client, collection_name, embedding_model):
    \"\"\"Initializes LangChain vector store, retriever, LLM, and QA chain.\"\"\"
    st.info("Step 8: Setting up LangChain components (VectorStore, Retriever, LLM, QA Chain)...")
    try:
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=collection_name,
            text_key="text",
            embedding=embedding_model
        )
        st.session_state.retriever = vectorstore.as_retriever()
        st.success("LangChain WeaviateVectorStore and Retriever initialized.")

        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
        st.success("Google Generative AI LLM initialized.")

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            retriever=st.session_state.retriever,
            chain_type="stuff"
        )
        st.success("RetrievalQA chain initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Error initializing RAG components: {e}")
        return False

# --- Document Upload and Processing ---
st.header("⬆️ Upload Your Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    file_content_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if file_content_hash != st.session_state.current_uploaded_file_hash:
        st.info("New document detected! Resetting application state to process new document.")
        st.session_state.documents_loaded_flag = False
        st.session_state.weaviate_setup_done_flag = False
        st.session_state.rag_initialized_flag = False
        st.session_state.qa_chain = None
        st.session_state.query_text_input = "" # Clear query input
        st.session_state.current_uploaded_file_hash = file_content_hash
        st.rerun()
    elif not st.session_state.documents_loaded_flag:
        pass

if uploaded_file is not None and not st.session_state.documents_loaded_flag:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Step 1: Loading documents..."):
        try:
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            st.session_state.documents = documents
            st.success(f"Document '{uploaded_file.name}' loaded successfully! Pages: {len(documents)}")
        except Exception as e:
            st.error(f"Error loading document: {e}")
            st.session_state.documents_loaded_flag = False
            os.unlink(tmp_file_path)
            st.stop()

    with st.spinner("Step 2: Splitting documents into chunks..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(st.session_state.documents)
            st.session_state.docs = docs
            st.success(f"Documents split into {len(docs)} chunks.")
        except Exception as e:
            st.error(f"Error splitting documents: {e}")
            st.session_state.documents_loaded_flag = False
            os.unlink(tmp_file_path)
            st.stop()

    with st.spinner("Step 3: Generating embeddings..."):
        try:
            if st.session_state.embedding_model is None:
                st.session_state.embedding_model = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
            
            text_content_for_embedding = [doc.page_content for doc in st.session_state.docs]
            embeddings = st.session_state.embedding_model.embed_documents(text_content_for_embedding)
            st.session_state.embeddings = embeddings
            st.success(f"Generated embeddings for {len(embeddings)} chunks. Embedding dimension: {len(embeddings[0])}")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            st.session_state.documents_loaded_flag = False
            os.unlink(tmp_file_path)
            st.stop()

    os.unlink(tmp_file_path)
    st.session_state.documents_loaded_flag = True
    st.rerun()

# --- Weaviate and RAG Setup Section ---
if st.session_state.documents_loaded_flag and not st.session_state.weaviate_setup_done_flag:
    with st.spinner("Setting up Weaviate database and loading data..."):
        if setup_weaviate_and_load_data(st.session_state.docs, st.session_state.embeddings, st.session_state.collection_name, st.session_state.embedding_model):
            st.session_state.weaviate_setup_done_flag = True
            st.rerun()

if st.session_state.weaviate_setup_done_flag and not st.session_state.rag_initialized_flag:
    with st.spinner("Initializing RAG model components..."):
        if initialize_rag_components(st.session_state.weaviate_client, st.session_state.collection_name, st.session_state.embedding_model):
            st.session_state.rag_initialized_flag = True
            if st.session_state.whisper_model is None:
                st.session_state.whisper_model = load_whisper_model()
            st.rerun()


# --- Query Section ---
if st.session_state.rag_initialized_flag:
    st.header("❓ Ask a Question About Your Document")
    
    col_query, col_mic = st.columns([0.9, 0.1])

    with col_query:
        # The key ensures Streamlit knows when the input value is programmatically changed
        query = st.text_input(
            "Enter your query here:",
            value=st.session_state.query_text_input,
            key="text_query_input_box", # Important for controlled input
            placeholder="e.g., What is Huzaifa's experience?",
            on_change=lambda: setattr(st.session_state, 'query_text_input', st.session_state.text_query_input_box)
        )
    
    with col_mic:
        st.write("") # Add a bit of space
        st.write("") # Add a bit of space to align the button
        if st.button("🎤", key="mic_button", help="Speak your question"):
            audio_filepath = None
            try:
                audio_filepath = record_audio(record_seconds=10) # Record for 5 seconds
                if audio_filepath:
                    st.toast("Transcribing speech...", icon="⏳")
                    transcribed_text = transcribe_audio(audio_filepath, st.session_state.whisper_model)
                    if transcribed_text:
                        st.session_state.query_text_input = transcribed_text # Update session state to reflect in text_input
                        st.toast("Transcription complete!", icon="✅")
                        st.rerun() # Rerun to update the text input box and trigger answer
                    else:
                        st.warning("Could not transcribe audio. Please try again.")
            except Exception as e:
                st.error(f"Error during voice input: {e}")
            finally:
                if audio_filepath and os.path.exists(audio_filepath):
                    os.remove(audio_filepath)

    # Automatically get answer if query_text_input is set (either by typing or voice)
    # and if it has changed from the last processed query
    if query and st.session_state.rag_initialized_flag:
        if st.session_state.get('last_processed_query') != query:
            st.session_state.last_processed_query = query # Update last processed query
            with st.spinner("Fetching answer..."):
                try:
                    response = st.session_state.qa_chain.run(query)
                    st.success("Answer:")
                    st.write(response)
                    text_to_speech(response, lang='en') # Speak the answer
                except Exception as e:
                    st.error(f"Error getting answer: {e}. Please check your Google API key or query.")
        
else:
    if uploaded_file is None:
        st.info("Please upload a PDF document to start the RAG process.")
    else:
        st.info("Processing document and setting up RAG components. Please wait...")

# --- Close Weaviate Client (best effort) ---
# if st.session_state.weaviate_client:
#     try:
#         st.session_state.weaviate_client.close()
#         st.session_state.weaviate_client = None
#         st.info("Weaviate client closed.")
#     except Exception as e:
#         st.warning(f"Could not close Weaviate client gracefully: {e}")
"""

# --- Streamlit App Setup ---
st.set_page_config(page_title="SpeakDoc Project Showcase", layout="wide")
st.title("📄 SpeakDoc Project Showcase")

# --- Project Title and Description ---
st.markdown("""
# 📄 SpeakDoc: Talk to your PDFs, get instant answers.

This project is a dynamic Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and Weaviate, featuring an integrated voice assistant for natural language interaction. It allows users to upload PDF documents, ask questions about their content (via text or voice), and receive spoken and textual answers.
""")

# --- Features Section ---
st.markdown("## ✨ Features")
st.markdown("""
* **PDF Document Upload:** Easily upload your PDF documents to create a custom knowledge base.
* **Automated Document Processing:**
    * Loads PDF content.
    * Splits documents into manageable chunks.
    * Generates embeddings for efficient retrieval.
* **Dynamic Knowledge Base:** Utilizes Weaviate as a vector database to store and retrieve document embeddings.
* **Retrieval-Augmented Generation (RAG):** Leverages Google's Gemini-1.5-Flash LLM with LangChain to provide accurate answers based on the uploaded document's content.
* **Voice Input (Speech-to-Text):**
    * Click a microphone icon to speak your questions.
    * Utilizes OpenAI's Whisper model for highly accurate speech transcription.
    * Transcribed text automatically populates the query input box.
* **Voice Output (Text-to-Speech):** Answers generated by the LLM are automatically converted to speech and played aloud using Google Text-to-Speech (gTTS).
* **Intuitive Streamlit UI:** A clean and interactive web interface for seamless user experience.
""")

# --- Tech Stack Section ---
st.markdown("## ⚙️ Tech Stack")
st.markdown("""
* **Frontend:** Streamlit
* **Backend/Orchestration:** LangChain
* **Vector Database:** Weaviate
* **Large Language Model (LLM):** Google Gemini-1.5-Flash (via `langchain-google-genai`)
* **Embeddings:** Sentence Transformers (`all-mpnet-base-v2`)
* **PDF Loading:** `PyPDFLoader`
* **Text Splitting:** `RecursiveCharacterTextSplitter`
* **Speech-to-Text (STT):** OpenAI Whisper
* **Text-to-Speech (TTS):** gTTS
* **Audio Playback:** Pygame
* **Audio Recording:** PyAudio
* **Environment Variables:** `python-dotenv`
""")


# --- Usage Section ---
st.markdown("## 💡 Usage")
st.markdown("""
Once the application is running:

1.  **Upload PDF:** Use the file uploader to select and upload your PDF document.
2.  **Wait for Processing:** The application will process the document (chunking, embedding, Weaviate setup, RAG initialization).
3.  **Ask a Question (Text or Voice):**
    * **Text Input:** Type your question directly into the query box.
    * **Voice Input:** Click the microphone icon next to the input box, speak your question, and the transcription will appear in the query box.
4.  **Get Answer:** The system will retrieve relevant information and generate an answer, which will be displayed and spoken aloud.
""")

# --- Deployment Note (Red Ribbon) ---
st.markdown("""
<div style="background-color: #ffe0e0; border-left: 6px solid #ff0000; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem; margin-bottom: 2rem;">
    <h3 style="color: #ff0000; margin-top: 0; margin-bottom: 0.5rem;">🚨 Important Note on Deployment 🚨</h3>
    <p style="color: #cc0000; margin-bottom: 0.5rem;">
        Due to compatibility issues with `openai-whisper` and resource limitations on cloud platforms, this version of the app is not publicly deployed.
        We are showcasing its functionality via the demo video and providing the code for local execution.
    </p>
</div>
""", unsafe_allow_html=True)


# --- Project Demo Video ---
st.markdown("## ▶️ Project Demo Video")
# Replace <your-github-username> and <your-repository-name> with your actual GitHub details
# Ensure the video file 'speakdoc-demo.mp4' is in your repository.
# For raw video, GitHub URL pattern is usually:
# https://raw.githubusercontent.com/<your-github-username>/<your-repository-name>/main/speakdoc-demo.mp4
# Assuming 'speakdoc' is the repository name based on logs, and 'mhuzaifa5' is a placeholder username
# PLEASE REPLACE THIS URL WITH YOUR ACTUAL RAW GITHUB VIDEO URL!
video_url = "https://raw.githubusercontent.com/mhuzaifa5/speakdoc/main/SpeakDoc-Demo.mp4" # Placeholder URL

st.video(video_url, format="video/mp4", start_time=0)
st.markdown(f"*(If the video doesn't load directly, you can also view it [here]({video_url}))*")


# --- Project Code Section ---
st.markdown("## 💻 Project Code (`dashboard.py`)")
st.markdown("""
Below is the complete Python code for the main RAG application (`dashboard.py`). You can copy and run this code on your local machine after setting up the environment as described in the 'Setup and Installation' section.
""")
st.code(DASHBOARD_CODE, language="python")

# --- Project Workflow Section ---
st.markdown("## 📈 Project Workflow")
st.markdown("""
Below is a basic workflow chart illustrating the core processes of the SpeakDoc application:

```mermaid
graph TD
    A[Start] --> B(User Uploads PDF Document)
    B --> C{Document Processing}
    C --> C1(Load PDF)
    C1 --> C2(Split into Chunks)
    C2 --> C3(Generate Embeddings)
    C3 --> D(Initialize Weaviate & Load Data)
    D --> E(Initialize RAG Components)
    E --> F{User Interaction}

    F --> F1[Type Query]
    F1 --> F2(Click 'Get Answer' Button)
    F2 --> G(RAG Process: Retrieve & Generate Answer)

    F --> F3[Click Microphone Icon]
    F3 --> F4(Record Audio)
    F4 --> F5(Transcribe Audio using Whisper)
    F5 --> F6(Transcribed Text Populates Query Box)
    F6 --> G

    G --> H(Display Answer)
    H --> I(Speak Answer using gTTS)
    I --> J[End]
```
**Explanation of the Workflow:**

1.  **Start:** The application begins.
2.  **User Uploads PDF Document:** The user provides a PDF file.
3.  **Document Processing:** The uploaded PDF undergoes several steps:
    * **Load PDF:** The document content is loaded.
    * **Split into Chunks:** The document is divided into smaller, manageable text chunks.
    * **Generate Embeddings:** Each text chunk is converted into a numerical vector (embedding).
4.  **Initialize Weaviate & Load Data:** A connection is established with the Weaviate vector database, and the document embeddings are loaded into a new or existing collection.
5.  **Initialize RAG Components:** The core components for Retrieval-Augmented Generation are set up, including the vector store, retriever, Large Language Model (LLM - Gemini-1.5-Flash), and the RetrievalQA chain.
6.  **User Interaction:** The user can now query the document using one of two methods:
    * **Type Query:** The user types their question into the input box and clicks the "Get Answer" button.
    * **Click Microphone Icon:** The user clicks the microphone icon next to the input box:
        * **Record Audio:** Their voice query is recorded.
        * **Transcribe Audio using Whisper:** The recorded audio is converted into text using the Whisper ASR model.
        * **Transcribed Text Populates Query Box:** The transcribed text automatically appears in the input query box, making the interaction seamless.
7.  **RAG Process: Retrieve & Generate Answer:** Whether typed or spoken, the query triggers the RAG process. Relevant document chunks are retrieved from Weaviate, and the LLM uses these chunks to formulate an answer.
8.  **Display Answer:** The generated answer is displayed on the Streamlit interface.
9.  **Speak Answer using gTTS:** The textual answer is also converted into speech and played aloud to the user.
10. **End:** The interaction concludes, awaiting further queries.
""")

# --- Contributing and License ---
st.markdown("## 🤝 Contributing")
st.markdown("""
Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open a pull request or an issue on the GitHub repository.
""")

st.markdown("## 📄 License")
st.markdown("""
This project is licensed under the MIT License - see the `LICENSE` file for details.
""")
