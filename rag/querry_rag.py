import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# === Load environment variables from .env file ===
load_dotenv()

# === Get Hugging Face API token from environment ===
HF_TOKEN = os.getenv("HF_TOKEN")

# === Set up paths and embedding model ===
INDEX_PATH = "rag_index"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === Load FAISS vector index ===
print("🔍 Loading FAISS index...")
db = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# === Define the custom prompt template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful travel assistant. Based on the following context from travel reviews and destination data, answer the user's question in a specific and informative way.

Context:
{context}

User question:
{question}

Answer:
"""
)

# === Set up Hugging Face inference endpoint ===
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # 🔁 Replace with your model
    temperature=0.7,
    max_new_tokens=512,
    huggingfacehub_api_token=HF_TOKEN
)

# === Set up the RetrievalQA chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# === Start interactive Q&A loop ===
print("🧠 Travel Recommender Ready! Ask your travel-related question.")
while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting. Safe travels!")
        break
    result = qa_chain.invoke({"query": query})
    print("\nAI Recommendation:\n", result["result"])
    print("\n🔎 Top sources used:")
    for doc in result["source_documents"][:3]:
        print("-", doc.page_content[:150].replace("\n", " ") + "...")
