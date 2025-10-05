## Multimodal RAG Chatbot (Streamlit + LangChain + Ollama)

### Overview
This project is a multimodal Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDFs and ask questions grounded in their content. It extracts text, tables, and images from PDFs, generates semantic summaries for images, and uses a local LLM (LLaVA via Ollama) to synthesize answers using both text and visual context.

### Features
- **PDF ingestion via Unstructured**: Extracts text, HTML tables, and base64 images.
- **Multimodal retrieval**: Stores text/table chunks and image summaries in ChromaDB; retrieves relevant parents using LangChain `MultiVectorRetriever`.
- **Local LLM (LLaVA)**: Responds using text context and inline base64 images.
- **Persistence & reproducibility**: Chroma collection persisted at `.chroma/multi_modal_rag`.
- **Streamlit UI**: Upload multiple PDFs, process, and chat.

### Architecture (high level)
- `main.py`: Streamlit app (upload, processing, model selection), PDF parsing, image summary generation.
- `vectorDB.py`: Vector store and retriever setup; persists Chroma; stores parent `Document` objects in an in-memory docstore with metadata.
- `answer_synthesis.py`: Builds the RAG chain: retrieves docs, formats multimodal message (text + images), calls LLaVA via Ollama.

Data flow:
1) PDF -> Unstructured chunks (text, tables, images as base64) -> optional image files dumped to `extracted_images/<pdf_name>/`.
2) Each image -> base64 -> summary via Ollama LLaVA REST -> stored as retrievable vector.
3) Text/table chunks + image summaries -> Chroma vectors; parent `Document`s (original text/table/base64 image) -> docstore.
4) Query -> retriever -> parent `Document`s -> formatted multimodal prompt -> LLaVA -> answer.

### Prerequisites
- Python 3.13 (as specified in `pyproject.toml`).
- System dependencies for Unstructured (PDF hi-res parsing may require these on Linux):
  - `tesseract-ocr`, `poppler-utils` (provides `pdftoppm`/`pdftocairo`), `libmagic`.
  - Example (Debian/Ubuntu):
```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils libmagic1
```
- Ollama running locally with the LLaVA model available.
  - Install Ollama: see [Ollama](https://ollama.com)
  - Pull model (first run will also pull automatically):
```bash
ollama pull llava
ollama serve
```

### Setup
This repo uses `pyproject.toml`. You can use `uv` (recommended for speed) or `pip`.

- Using `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace
uv sync
```

- Using `pip` (creates a venv and installs deps):
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Running the app
1) Ensure Ollama is running and the `llava` model is available:
```bash
ollama serve
# in another terminal
ollama run llava "hello"
```
2) Start Streamlit:
```bash
cd /workspace
streamlit run main.py
```
3) In the app sidebar:
- Select model: `llava:latest`.
- Click “Initialize Model”.
- Upload one or more PDFs and click “Process”.
- Ask questions in the main input box and click “Submit”.

### Configuration
- **Model selection**: The UI sets `OLLAMA_MODEL_NAME` (default fallback is `llava`). `answer_synthesis.py` reads this env var.
- **Chroma persistence**: Vector DB stored in `.chroma/multi_modal_rag`.
- **Working directories**:
  - Uploaded PDFs: `uploaded_files/`
  - Extracted images: `extracted_images/<pdf_basename>/`

### Retrieval details
- Embeddings: `HuggingFaceEmbeddings("all-MiniLM-L6-v2")`.
- Vector store: Chroma, persisted.
- Retriever: `MultiVectorRetriever` with parent `Document`s stored in an in-memory docstore.
- Chunking (Unstructured `partition_pdf`): tuned to `max_characters=1200`, `new_after_n_chars=900`, `combine_text_under_n_chars=200` for better recall.

### Troubleshooting
- “No relevant context” or weak answers:
  - Verify processing logs in the UI captions (counts of texts/tables/images and image summaries).
  - Check that `.chroma/multi_modal_rag` is populated after processing.
  - Ensure Ollama is running; watch for image summary errors in the terminal.
- Image summaries length mismatch warning:
  - The app will align pairs to the minimum length and log a warning. Check Ollama connectivity.
- Unstructured errors on Linux:
  - Install `tesseract-ocr`, `poppler-utils`, `libmagic`. Ensure `pdftoppm` is on PATH.

### Project structure
```
/workspace
  ├── main.py                # Streamlit UI, ingestion, processing
  ├── vectorDB.py            # Chroma + retriever setup, persistence
  ├── answer_synthesis.py    # RAG chain and multimodal formatting
  ├── pyproject.toml         # Python project + dependencies
  ├── uv.lock                # uv lockfile (if using uv)
  └── README.md              # This file
```

### Roadmap (ideas)
- Add citation highlighting and source previews in the UI.
- Add GPU-accelerated image captioning and batch processing.
- Support additional local models and remote providers.

### License
This project is provided as-is for educational/demo purposes.

