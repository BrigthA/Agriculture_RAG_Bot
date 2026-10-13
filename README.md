# Tamil Nadu Agriculture AI Assistant 🌱

A local-first, privacy-conscious Retrieval-Augmented Generation (RAG) assistant designed for Tamil Nadu agricultural schemes, crop practices, subsidies, and department data. It ingests official PDFs, images, and approved web links, performs hybrid semantic-keyword retrieval (FAISS + BM25), maintains strict source citations, and includes an interactive operations and token-usage dashboard.

---

## Key Features

- **Multi-Format Ingestion:** Process official PDFs and layout-aware images (PNG, JPG, WEBP) using OpenAI Vision OCR and custom table/budget extraction.
- **Controlled Web Crawling:** Safely discover and index documents and scheme pages from approved government domains (`tn.gov.in`, `tnau.ac.in`, `agritech.tnau.ac.in`, etc.).
- **Hybrid Search (FAISS + BM25):** Combine dense multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) with sparse keyword matching using Reciprocal Rank Fusion (RRF).
- **Grounded Answers:** Strict grounding policy that cites official sources (`[Source 1]`) and abstains when evidence is missing.
- **Voice Support:** Local speech-to-text transcription (`faster-whisper`), voice recording, and text-to-speech audio playback (`pyttsx3`).
- **Operations & Analytics Dashboard:** Track API token consumption, ingestion performance, error logs, and user feedback via embedded SQLite storage and Plotly charts.

---

## Prerequisites

- **Python:** Version 3.10 or higher.
- **OpenAI API Key:** Required for LLM chat generation and Vision image OCR. (Local Hugging Face embeddings run locally on CPU/GPU without requiring an API token).

---

## Installation & Setup

1. **Clone or download this repository** containing `AgricultureAI_Bot.py`.

2. **Install the required Python packages:**
   ```bash
   pip install streamlit pandas requests beautifulsoup4 pypdf langchain-core langchain-openai langchain-huggingface langchain-text-splitters langchain-community faiss-cpu rank-bm25 plotly faster-whisper pyttsx3 Pillow python-dotenv
   ```
   *(Note: Depending on your system audio configuration for text-to-speech, you may need system packages like `espeak` on Linux or `SAPI5` on Windows).*

3. **Configure your environment variables:**
   Create a `.env` file in the project root directory or configure Streamlit secrets (`.streamlit/secrets.toml`):
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_CHAT_MODEL=gpt-4o-mini
   OPENAI_VISION_MODEL=gpt-4o
   HF_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   AGRI_ALLOWED_DOMAINS=tn.gov.in,tnau.ac.in,agritech.tnau.ac.in,pmkisan.gov.in,icar.gov.in,agmarknet.gov.in,agricoop.nic.in
   ```

---

## Running the Application

Launch the Streamlit app from your terminal:

```bash
streamlit run AgricultureAI_Bot.py
```

Open your browser at `http://localhost:8501`.

---

## How to Use the App

1. **Knowledge Ingestion (Admin):**
   - Navigate to the **Data Ingestion** tab.
   - Upload official agricultural PDFs, brochures, or scanned document images.
   - Optionally enter approved URLs to crawl for scheme guidelines.
   - Click **Process and index** to build the search vector database.

2. **Farmer Assistant & Chat:**
   - Navigate to the **AI Assistant** tab.
   - Ask questions about crops, irrigation, government schemes, or budget allocations.
   - View inline source citations, listen to voice audio, copy answers, or provide feedback (`👍` / `👎`).

3. **Usage & Operations Dashboard:**
   - Navigate to the **Dashboard** tab to inspect token usage, daily consumption charts, system event logs, and operational metrics.