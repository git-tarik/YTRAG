import os
import re
from urllib.parse import urlparse, parse_qs

# --- Environment and API Key Setup ---
from dotenv import load_dotenv
load_dotenv()

# --- LangChain and other necessary imports ---
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Helper Function to Extract YouTube Video ID ---
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

def get_transcript(video_id: str, language: str) -> str | None:
    """
    Fetches and formats the transcript for a given video ID and language.
    """
    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=[language])
        transcript_list = fetched_transcript.to_raw_data()
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for this video (ID: {video_id}).")
        return None
    except Exception as e:
        print(f"Could not retrieve transcript for language '{language}'. Error: {e}")
        return None


def create_rag_chain(transcript: str):
    """Creates a RAG chain with OpenAI embeddings + Google Gemini LLM."""
    # 1. Split the transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([transcript])

    # 2. Create embeddings (OpenAI)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 3. Store in FAISS
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 4. Prompt Template
    template = """
    You are a helpful assistant.
    Answer the question based ONLY on the provided transcript context.
    If the context is insufficient to answer the question, just say:
    "I don't know based on the provided transcript."

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
    prompt = PromptTemplate.from_template(template)

    # 5. Define Google Gemini LLM (Text Generation)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=500,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 6. Function to format retrieved docs
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 7. Build RAG Chain (LCEL)
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


# --- Main Application Logic ---
def main():
    """Main function to run the interactive RAG app."""
    print("--- YouTube Video RAG Assistant (Hybrid: OpenAI + Gemini) ---")

    youtube_url = input("Please enter the YouTube video URL: ")
    video_id = get_video_id(youtube_url)

    if not video_id:
        print("Invalid YouTube URL. Exiting.")
        return

    print(f"✅ Extracted Video ID: {video_id}")

    language = input("Enter transcript language code (e.g., 'en' or 'hi'): ").strip().lower()

    print("\nFetching transcript...")
    transcript_text = get_transcript(video_id, language)

    if not transcript_text:
        print("❌ Transcript unavailable. Exiting.")
        return

    print("✅ Transcript fetched. Building RAG chain...")
    rag_chain = create_rag_chain(transcript_text)
    print("RAG chain ready! Ask questions below (type 'exit' to quit).")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == "exit":
            print("👋 Goodbye!")
            break

        print("Thinking...")
        answer = rag_chain.invoke(user_query)

        print("\n--- Answer ---")
        print(answer)
        print("--------------")


if __name__ == "__main__":
    main()
