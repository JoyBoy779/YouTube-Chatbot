# Load environment variables (API keys)

from dotenv import load_dotenv
load_dotenv()

# Imports
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_core.runnables import (RunnableParallel,RunnablePassthrough,RunnableLambda,)
from langchain_core.output_parsers import StrOutputParser

# STEP 1: INDEXING
# (Document Loading + Text Splitting + Embeddings + Vector Store)

# Step 1a - Indexing (Document Loading / Ingestion)

video_id = input("Enter YouTube video ID: ").strip()   # Only the video ID, not full URL

try:
    # Fetch transcript (English)
    fetched_transcript = YouTubeTranscriptApi().fetch(
        video_id,
        languages=["en"]
    )

    # Convert transcript chunks into plain text
    transcript_list = fetched_transcript.to_raw_data()
    transcript = " ".join(chunk["text"] for chunk in transcript_list)

except TranscriptsDisabled:
    raise RuntimeError("No captions available for this video.")

except Exception as e:
    raise RuntimeError(f"Invalid video ID or transcript fetch failed: {e}")


# Step 1b - Indexing (Text Splitting / Chunking)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])


# Step 1c - Indexing (Embedding Generation)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"   # Free, local model
)
# Step 1d - Indexing (Storing embeddings in Vector Store)
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

# STEP 2: RETRIEVAL
# (Semantic search over vector store)
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 4})

user_query = input("Ask a question about the video: ")
retrieved_docs = retriever.invoke(user_query)

# STEP 3: AUGMENTATION
# (Combine retrieved context with the user question)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY using the provided transcript context.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

# Convert retrieved documents into a single context string
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({
    "context": context_text,
    "question": user_query
})

# STEP 4: GENERATION
# (LLM generates the final answer)
answer = llm.invoke(final_prompt)

print("\n================ ANSWER ================\n")
print(answer.content)
print("\n=======================================\n")


# =======================================================
# Building a proper LangChain RAG chain
# =======================================================

def format_docs(docs):
    """Formats retrieved documents into a single context string"""
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# Example chain invocation
summary = main_chain.invoke("Can you summarize the video?")
print("\n============= SUMMARY =============\n")
print(summary)
print("\n==================================\n")
