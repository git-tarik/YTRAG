import os
import re
from urllib.parse import urlparse, parse_qs

# --- Environment and API Key Setup ---
# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# --- LangChain and other necessary imports ---
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# --- Helper Function to Extract YouTube Video ID ---
def get_video_id(url: str) -> str | None:
    """
    Extracts the YouTube video ID from a YouTube URL.
    Handles standard, short, and embed URLs.
    """
    if not url:
        return None
    
    # Regex to find the video ID in various URL formats
    regex = r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    
    match = re.search(regex, url)
    
    if match:
        return match.group(1)
    
    return None

# --- Core RAG Functions ---

def get_transcript(video_id: str, language: str) -> str | None:
    """
    Fetches and formats the transcript for a given video ID and language.
    Uses the non-deprecated .fetch() method.
    """
    try:
        # Fetch returns a TranscriptList object
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=[language])
        
        # Convert the TranscriptList to a list of dictionaries
        transcript_list = fetched_transcript.to_raw_data()
        
        # Join the text from each dictionary
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for this video (ID: {video_id}).")
        return None
    except Exception as e:
        print(f"Could not retrieve transcript for language '{language}'. This language might not be available. Error: {e}")
        return None

def create_rag_chain(transcript: str):
    """Creates a RAG chain from the provided transcript text."""
    # 1. Split the transcript into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([transcript])
    
    # 2. Create embeddings and a vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # 3. Define the prompt template
    template = """
      You are a helpful assistant.
      Answer the question based ONLY on the provided transcript context.
      If the context is insufficient to answer the question, just say "I don't know based on the provided transcript."

      CONTEXT:
      {context}

      QUESTION:
      {question}
    """
    prompt = PromptTemplate.from_template(template)

    # 4. Define the LLM, including a token limit for cost control
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=500)

    # 5. Define a function to format the retrieved documents
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 6. Build the RAG chain using LangChain Expression Language (LCEL)
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
    """Main function to run the interactive RAG application."""
    print("--- YouTube Video RAG Assistant ---")
    
    # Get YouTube URL from user
    youtube_url = input("Please enter the YouTube video URL: ")
    video_id = get_video_id(youtube_url)
    
    if not video_id:
        print("Invalid YouTube URL. Could not extract video ID. Exiting.")
        return

    print(f"Successfully extracted Video ID: {video_id}")

    # Get desired language from user
    language = input("Enter the language code for the transcript (e.g., 'en' for English, 'hi' for Hindi): ").strip().lower()

    # Fetch and process the transcript
    print("\nFetching and processing transcript...")
    transcript_text = get_transcript(video_id, language)
    
    if not transcript_text:
        print("Could not proceed without a transcript. Exiting.")
        return
        
    print("Transcript processed successfully. Building RAG chain...")
    
    # Create the RAG chain
    rag_chain = create_rag_chain(transcript_text)
    print("RAG chain is ready. You can now ask questions about the video.")
    print("Type 'exit' to quit.")

    # Interactive query loop
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        
        print("Thinking...")
        
        # Invoke the chain and get the answer
        answer = rag_chain.invoke(user_query)
        
        print("\n--- Answer ---")
        print(answer)
        print("--------------")

if __name__ == "__main__":
    main()