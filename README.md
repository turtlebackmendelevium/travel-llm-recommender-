# 🧠 Travel LLM Recommender

A smart, modular recommendation system that explores and compares **three major techniques** for travel recommendations using **Large Language Models (LLMs)**:

- ✅ **Prompt Engineering**
- ✅ **Fine-Tuning a T5 Transformer**
- ⚙️ **Retrieval-Augmented Generation (RAG)** *(in progress)*

> 🚧 This is a **work-in-progress project** intentionally structured to reflect real-world iterative development.  
> 💼 This README clearly outlines what’s complete, what’s underway, and how each part reflects applied AI/ML and data engineering skills.

---

## 🔍 Project Goal

Build a smart travel assistant that recommends destinations using natural language queries and integrates:

- LLMs via Hugging Face Transformers  
- Custom dataset curation from TripAdvisor and European tourism sources  
- Vector search using FAISS  
- Text embedding using Sentence Transformers  
- Comparison of different LLM-based strategies for recommendation

---

## ✅ Completed Components

### 1. 🔧 Prompt Engineering

- Built structured, optimized prompts to get high-quality travel suggestions from LLM APIs (OpenAI, DeepSeek)
- Demonstrated impact of temperature, instruction clarity, and token limits
- Designed prompt templates for reproducibility and easy testing

### 2. 🧪 Fine-Tuning a T5 Model

- Cleaned and formatted 500-entry dataset into Hugging Face–compatible JSONL format
- Fine-tuned a **T5 model** using the Hugging Face `Trainer` API
- Used **PyTorch** backend with GPU acceleration
- Implemented training checkpoints and saved the final model under:
- Avoided overfitting by controlling input-output similarity in training data

### 3. 📁 Data Preprocessing & EDA

- Combined and cleaned datasets from:
- TripAdvisor hotel reviews
- 2023 traveler reviews
- European tourist destinations
- Handled encoding issues (`utf-8`)
- Used `pandas` and `langchain_community.document_loaders.CSVLoader`
- Explored destination clusters and patterns for grounding LLM output

---

## 🚧 In Progress

### 4. 🔄 Retrieval-Augmented Generation (RAG)

- Setup includes:
- `RecursiveCharacterTextSplitter` for document chunking
- `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2`
- `FAISS` for semantic vector indexing
- Status:
- Document splitting initiated on multi-thousand row CSV corpus
- FAISS index build is currently underway (expected due to large corpus size)
- Next Steps:
- Save index and test LangChain-based query inference
- Validate RAG vs fine-tuned and prompt-based results

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Core programming |
| 🧠 Hugging Face Transformers | LLMs, tokenizers, fine-tuning |
| 🔗 LangChain | RAG and embedding pipelines |
| 🧮 FAISS | Vector similarity search |
| 🧬 Sentence Transformers | `all-MiniLM-L6-v2` for embedding |
| ⚙️ PyTorch | Fine-tuning backend |
| 📊 pandas / NumPy | Data preprocessing |
| 📁 JSONL / CSV | Dataset formats |
| 💻 VS Code | Dev environment |
| 🧪 Jupyter | Local EDA experiments |

---

### 📁 Folder Structure

```bash
travel-llm-recommender/
├── finetune/
│   ├── prepare_data.py
│   ├── train.py
│   ├── dataset.jsonl
│   └── output/
│       └── final-model/
├── rag/
│   ├── build_index.py          # Indexing logic (FAISS)
│   ├── querry_rag.py           # Inference using vector store
├── datasets_raw/               # Raw CSV datasets
└── README.md                   # Project documentation
```
---

## 📦 Download Required Datasets

To keep the repository lightweight, large data folders are hosted externally on Google Drive.  
Please manually download and extract the following folders:

- [`datasets_raw/`](https://drive.google.com/drive/folders/1VKZX22fWZjsI6xKxy8oFEc2UxH3_V5Nz?usp=sharing) – Raw CSVs and travel data


### How to Set Up:

1. Click each link above and choose **“Download”** from the Google Drive interface.
2. Google will deliver a `.zip`; unzip it on your machine.

---

## ✍️ What I Learned

- Fine-tuning transformer models with domain-specific data
- Vector search pipelines and hybrid LLM architectures
- Real-world dataset issues (formatting, encoding, redundancy)
- Comparison of prompt engineering vs. learned representations
- Integrating tools across Hugging Face, LangChain, FAISS, and PyTorch

---

## 📌 Why This Project Matters

This project demonstrates:

- 🔍 Depth in working with LLMs at various levels (prompting, fine-tuning, RAG)
- 🧱 Modular design that mirrors real-world system building
- 📚 Clear separation between exploration, modeling, and retrieval
- 🧩 Realistic challenges in large-scale unstructured data handling

> 🧠 Even though RAG is still in progress, this project shows full-cycle model development, thoughtful architecture, and the ability to build and debug deep AI systems from scratch.

---

---

## 🔜 Next Steps

- Complete FAISS indexing and integrate RAG inference  
- Benchmark all 3 approaches (prompting, fine-tune, RAG)  
- Build Streamlit UI for public demo  
- Dockerize the pipeline for reproducible deployment

---


🤝 Connect & Collaborate
This project reflects my commitment to building real-world AI systems that integrate LLMs, vector search, and modern machine learning workflows.
If you're working on anything at the intersection of AI, data, or product—and want to trade ideas or collaborate—I'm always open to meaningful conversations and innovative work.

💼 [LinkedIn](https://www.linkedin.com/in/mohammad-abbasi-393254263/)

🧠 [GitHub](https://github.com/turtlebackmendelevium)

📨 [Email](ayaan.abbasi01@outlook.com)



---

