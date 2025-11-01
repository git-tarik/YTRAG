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

# --- Core RAG Functions (No changes) ---

@st.cache_data(show_spinner="Fetching transcript...")
def get_transcript(video_id: str, language: str) -> str | None:
    """
    Fetches and formats the transcript for a given video ID and language.
    This implementation now strictly follows your provided code structure.
    """
    try:
        fetched_transcript_object = YouTubeTranscriptApi().fetch(video_id, languages=[language])
        transcript_list = fetched_transcript_object.to_raw_data()
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
        
    except TranscriptsDisabled:
        st.error(f"Transcripts are disabled for this video (ID: {video_id}). Please try another video.")
        return None
    except Exception as e:
        st.error(f"Could not retrieve transcript for language '{language}'. Error: {e}")
        try:
            available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            available_langs = [t.language_code for t in available_transcripts]
            st.warning(f"This video has transcripts available for: {', '.join(available_langs)}")
        except Exception:
            pass
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
        model="gemini-2.5-flash",
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
        # UI FIX: Changed AI emoji
        {"role": "assistant", "content": "Hi! I'm ready for a new video. Provide a URL to get started."}
    ]
    st.session_state.rag_chain = None
    st.session_state.video_id = None
    if 'youtube_url_input' in st.session_state:
        st.session_state.youtube_url_input = ""
    get_transcript.clear()

# --- Streamlit UI ---

# UI FIX: Make the UI responsive by using the whole width
st.set_page_config(page_title="YT-CHAT-AI", page_icon="📺", layout="wide")

if 'last_interaction_time' in st.session_state and \
   (time.time() - st.session_state.last_interaction_time > SESSION_TIMEOUT_SECONDS):
    st.warning(f"Session timed out due to inactivity. Please start over.")
    reset_session()

if "messages" not in st.session_state:
    reset_session()
    
if 'welcome_message_shown' not in st.session_state:
    st.toast("Welcome to Youtube Video Agent! 👋", icon="🎉")
    time.sleep(0.5)
    st.toast("Clear your doubts without watching the whole video.", icon="🚀")
    st.session_state.welcome_message_shown = True

# UI FIX: Add a beautiful, colorful header without emojis
st.markdown("""
<div style="
    background: linear-gradient(90deg, #4F80C9, #7A60B3);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
">
<h1 style="color:white; font-size: 2.5rem; font-weight: bold; margin: 0;">YT-CHAT-AI</h1>
<p style="margin: 5px 0 0 0; font-size: 1.1rem;">Your Intelligent YouTube Video Assistant</p>
</div>
""", unsafe_allow_html=True)


# UI FIX: The "Get Started" window (expander) now stays open all the time.
with st.expander("🔗 Get Started: Enter Video Details Here", expanded=True):
    youtube_url = st.text_input("YouTube URL", key="youtube_url_input", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")

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
                    st.success("Assistant is ready! You can now ask questions below.")
                    # Removed st.rerun() so the expander stays open
                else:
                    st.session_state.video_id = None
            else:
                st.error("Invalid YouTube URL. Please enter a valid one.")
        else:
            st.warning("Please provide a YouTube URL.")

    if st.session_state.get('video_id') and st.session_state.get('rag_chain'):
        st.divider()
        st.success(f"Video Loaded Successfully!")
        if st.session_state.get('youtube_url_input'):
             st.video(st.session_state.youtube_url_input)
        
        st.button("Chat with Another Video", on_click=reset_session)

st.divider()

# --- Chat History Display ---
for message in st.session_state.messages:
    # UI FIX: Changed AI emoji
    with st.chat_message(message["role"], avatar="✨" if message["role"] == "assistant" else "👤"):
        with st.container(border=True):
            st.markdown(message["content"])

# --- Chat Input and Response Handling ---
if prompt := st.chat_input("Ask a question about the video..."):
    if st.session_state.rag_chain is None:
        st.error("Please set up a video in the 'Get Started' section above first.")
    else:
        st.session_state.last_interaction_time = time.time()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            with st.container(border=True):
                st.markdown(prompt)

        # UI FIX: Changed AI emoji
        with st.chat_message("assistant", avatar="✨"):
            with st.container(border=True):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_chain.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})