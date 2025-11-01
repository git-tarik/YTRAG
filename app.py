import os
import re
import time
import streamlit as st
from urllib.parse import urlparse, parse_qs

# --- LangChain and other necessary imports ---
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# --- Import Google Generative AI LLM wrapper ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONSTANTS ---
SESSION_TIMEOUT_SECONDS = 900 # 15 minutes
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
}

# --- Helper Function to Extract YouTube Video ID (No changes) ---
def get_video_id(url: str) -> str | None:
    """
    Extracts the YouTube video ID from a YouTube URL.
    Handles standard, short, and embed URLs.
    """
    if not url:
        return None
    regex = r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

# --- Core RAG Functions ---

@st.cache_data(show_spinner="Fetching transcript...")
def get_transcript(video_id: str, language: str) -> str | None:
    """
    Fetches and formats the transcript for a given video ID and language.
    """
    try:
        transcript_list_obj = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        transcript = " ".join([d['text'] for d in transcript_list_obj])
        return transcript
    except TranscriptsDisabled:
        st.error(f"Transcripts are disabled for this video (ID: {video_id}). Please try another video.")
        return None
    except Exception as e:
        # Attempt to find available transcripts if the selected one fails
        try:
            available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            available_langs = [t.language for t in available_transcripts]
            st.error(f"Could not retrieve transcript for language '{language}'. This video has transcripts available for: {', '.join(available_langs)}. Please select one of these.")
        except Exception:
            st.error(f"Could not retrieve transcript for language '{language}'. This language might not be available, or another error occurred: {e}")
        return None

def create_rag_chain(_transcript: str):
    """Creates a RAG chain with OpenAI embeddings + Google Gemini LLM."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([_transcript])

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    template = """
    You are a helpful assistant that answers questions based ONLY on the provided video transcript.
    Your tone should be conversational and helpful.
    If the context is insufficient to answer the question, politely state that the information is not in the transcript.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
    prompt = PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Updated to a generally available and strong model
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    rag_chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def reset_session():
    """Resets the Streamlit session state to start with a new video."""
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm ready for a new video. Provide a URL to get started."}
    ]
    st.session_state.rag_chain = None
    st.session_state.video_id = None
    # This line specifically clears the text in the URL input box
    if 'youtube_url_input' in st.session_state:
        st.session_state.youtube_url_input = ""
    # Clear any cached transcripts from previous runs to be safe
    get_transcript.clear()

# --- Streamlit UI ---

st.set_page_config(page_title="Chat with YouTube", page_icon="📺", layout="centered")

if 'last_interaction_time' in st.session_state and \
   (time.time() - st.session_state.last_interaction_time > SESSION_TIMEOUT_SECONDS):
    st.warning(f"Session timed out due to inactivity. Please start over.")
    reset_session()

if "messages" not in st.session_state:
    reset_session()

with st.sidebar:
    st.header("🔗 Video Setup")
    youtube_url = st.text_input("YouTube URL", key="youtube_url_input")

    selected_lang_name = st.selectbox(
        "Select Transcript Language",
        options=list(LANGUAGES.keys()),
        key="language_select"
    )

    if st.button("Start Chatting", type="primary"):
        if youtube_url:
            video_id = get_video_id(youtube_url)
            if video_id:
                st.session_state.rag_chain = None
                st.session_state.video_id = video_id
                language_code = LANGUAGES[selected_lang_name]
                transcript = get_transcript(video_id, language_code)
                if transcript:
                    st.session_state.rag_chain = create_rag_chain(transcript)
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"I'm ready! Ask me anything about the video."}
                    ]
                    st.success("Assistant is ready!")
                else:
                    st.session_state.video_id = None
            else:
                st.error("Invalid YouTube URL. Please enter a valid one.")
        else:
            st.warning("Please provide a YouTube URL.")

    st.divider()

    if st.session_state.get('video_id'):
        st.success(f"Video Loaded")
        # Display the video using the url from the input field
        if st.session_state.get('youtube_url_input'):
            st.video(st.session_state.youtube_url_input)
        
        # --- THIS IS THE FIX ---
        # Use on_click to call the reset function before the script reruns
        st.button("Chat with Another Video", on_click=reset_session)

st.title("📺 Chat with any YouTube Video")
st.write("Enter a YouTube URL in the sidebar, choose the transcript language, and start asking questions!")
st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the video..."):
    if st.session_state.rag_chain is None:
        st.error("Please set up a video in the sidebar first.")
    else:
        st.session_state.last_interaction_time = time.time()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})