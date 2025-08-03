from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Paths to datasets ===
files = [
    "datasets_raw/TripAdvisor Hotel Reviews/hotel.reviews.csv",
    "datasets_raw/Tripadvisor reviews 2023/tripadvisor_reviews_2023.csv",
    "datasets_raw/European Tour Destinations Dataset/tourist.destinations.csv"
]

docs = []

# === Load CSV files ===
for file in files:
    file_path = Path(file)
    print(f"📂 Loading: {file_path}")
    try:
        loader = CSVLoader(
            file_path=str(file_path),
            encoding="utf-8",
            csv_args={"delimiter": ","}
        )
        documents = loader.load()
        docs.extend(documents)
        print(f"✅ Loaded {len(documents)} documents from {file_path}")
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")

if not docs:
    print("❌ No valid documents found. Aborting.")
    exit()

# === Split documents ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)
print(f"📎 Created {len(splits)} document chunks.")

# === Generate embeddings and build FAISS index ===
print("🔎 Generating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("📦 Creating FAISS index...")
db = FAISS.from_documents(splits, embeddings)

# === Save FAISS index ===
db.save_local("rag_index")
print("✅ Index successfully saved to 'rag_index'")
