# 🤱 RAG Buku KIA — Maternal & Child Health Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about
**pregnancy, childbirth, and breastfeeding**, grounded in Indonesia's official
**Buku KIA (Maternal and Child Health Handbook)** published by the Ministry of Health
(Kemenkes RI). Built entirely with open-source models — no paid API required.

> ⚕️ **Disclaimer:** This chatbot provides general information only and is **not a
> substitute** for consultation with a midwife or doctor. For emergencies, seek
> immediate care at the nearest health facility.

## ✨ Features

- Q&A on pregnancy (by trimester), childbirth, and breastfeeding
- Every answer cites its **source page(s)** from the Buku KIA
- Automatic **danger-sign detection** — when a question or retrieved context involves
  an emergency, the chatbot proactively advises seeking immediate medical care
- Hybrid retrieval (semantic search + keyword boost) for more accurate results

## 🏗️ Architecture

```
Buku KIA (PDF)
   │
   ├─ Per-page extraction (pdfplumber, with left/right column splitting)
   ├─ Automatic section-title & table detection
   ├─ Chunking + sub-chunking per symptom/item
   │
   ▼
Embedding (multilingual-e5-base) → FAISS index
   │
   ▼
Retrieval (hybrid: semantic + keyword boost)
   │
   ▼
Generation (local LLM, grounded to context + danger-sign guardrail)
   │
   ▼
Answer + source pages
```

## 🛠️ Tech Stack

| Component | Tools |
|---|---|
| PDF extraction | `pdfplumber` |
| Embedding | `sentence-transformers` (`intfloat/multilingual-e5-base`) |
| Vector store | `FAISS` |
| Generation (notebook/Colab) | `transformers` + `bitsandbytes` (Qwen2.5-3B-Instruct, 4-bit) |
| Generation (deployment) | `llama-cpp-python` (Qwen2.5-3B-Instruct GGUF, CPU-friendly) |
| UI | `Streamlit` |

## 📁 Repo Structure

```
├── app.py                  # Streamlit app (deployment)
├── requirements.txt        # Deployment dependencies
├── data/
│   ├── chunks.json         # Chunked content from Buku KIA
│   └── faiss_index.bin     # Vector index (FAISS)
└── notebook/
    └── KIA_RAG.ipynb       # Colab notebook: extraction → chunking → embedding → testing
```

## 🚀 Rebuilding the Pipeline (Optional)

To rebuild the index from a different Buku KIA PDF:

1. Open `notebook/KIA_RAG.ipynb` in Google Colab
2. Upload the Buku KIA PDF you want to process
3. Run the cells in order — from PDF extraction through manual testing
4. Copy the resulting `chunks.json` and `faiss_index.bin` into the `data/` folder for
   `app.py` to use

## 🌐 Live Demo

Deployed on **Streamlit Community Cloud**: *(add link once live)*

## 📖 Data Source

Buku KIA (Maternal and Child Health Handbook), published by the Ministry of Health of
the Republic of Indonesia (Kemenkes RI).
