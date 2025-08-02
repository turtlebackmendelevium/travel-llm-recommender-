**🧠 Travel LLM Recommender
A smart, modular recommendation system that explores and compares three major techniques for travel recommendations using Large Language Models (LLMs):

✅ Prompt Engineering

✅ Fine-Tuning a T5 Transformer

⚙️ Retrieval-Augmented Generation (RAG) (in progress)

🚧 This is a work-in-progress project intentionally structured to reflect real-world iterative development. Recruiters: See below for what’s complete and how each part demonstrates relevant technical skill.

**🔍 Project Goal
Build a smart travel assistant that recommends destinations using natural language inputs and integrates:

LLMs via Hugging Face

Custom dataset curation (TripAdvisor + European destination data)

Vector search with FAISS

Text embeddings via Sentence Transformers

Evaluation and optimization across LLM techniques

**✅ Completed Components
1. 🔧 Prompt Engineering
Developed a structured prompt template for travel recommendations

Integrated LLM API (DeepSeek) for iteration

Demonstrated impact of prompt structure on model outputs

2. 🧪 Fine-Tuning with Hugging Face Transformers
Used a cleaned and formatted JSONL dataset of travel queries

Fine-tuned a T5 model using Hugging Face’s Trainer API

Ran training locally using PyTorch with checkpoints saved and reused

Evaluated model output to prevent overfitting on input sequences

Final model saved at: finetune/output/final-model

3. 📁 Data Preprocessing & EDA
Merged multiple datasets (TripAdvisor, 2023 reviews, European destinations)

Cleaned CSVs, handled encoding issues, removed redundancy

Used pandas, numpy, and langchain’s document loaders for formatting

Explored and visualized travel patterns to better guide model behavior

**🚧 In Progress
4. 🔄 RAG (Retrieval-Augmented Generation)
Set up LangChain pipeline using:

RecursiveCharacterTextSplitter

HuggingFaceEmbeddings (all-MiniLM-L6-v2)

FAISS for similarity search

Current status: Indexing is taking unusually long due to large corpus size

Next Steps:

Monitor and optimize document splitting

Finalize vectorstore saving/loading

Connect RAG to inference script for hybrid QA generation

**🛠️ Tech Stack
Tool	Purpose
🐍 Python	Core programming
🧠 Hugging Face Transformers	LLMs, tokenizers, fine-tuning
🔗 LangChain	RAG + Embedding pipeline
🧮 FAISS	Vector store for semantic search
🔤 Sentence Transformers	all-MiniLM-L6-v2 for embedding
⚙️ PyTorch	Model training backend
📊 Pandas / NumPy	Data prep and transformation
📁 JSONL / CSV	Data formats

travel-llm-recommender/
│
├── finetune/
│   ├── prepare_data.py
│   ├── train.py
│   ├── dataset.jsonl
│   └── output/
│
├── rag/
│   ├── build_index.py      ← [Indexing logic]
│   ├── querry_rag.py       ← [Querying interface]
│
├── datasets_raw/           ← [CSV source files]
│
└── README.md               ← [You're here]

**✍️ What I Learned
Fine-tuning a transformer model with custom data

Data formatting best practices for NLP

Comparative analysis of LLM techniques (prompting vs fine-tuning vs RAG)

Troubleshooting LangChain and Hugging Face integration

Managing real-world dataset inconsistency (encoding, structure)

**📌 Why This Project Matters
This project isn't just about the end result—it's a demonstration of:

🔍 Deep understanding of LLM internals

🧱 Layered architecture (modular components)

🧠 Ability to evaluate and iterate between techniques

🧩 Clear version control and extensibility

Even though RAG is still being finalized, the depth of experimentation, full fine-tuning pipeline, and understanding of embedding-driven retrieval clearly show readiness for applied AI/ML roles.

**🔜 Next Steps
Complete FAISS indexing and connect it to RAG query script

Add Streamlit UI for demo

Implement unit tests for each module

Optional: Dockerize and deploy inference API


💻 VS Code	Dev environment

