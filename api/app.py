from __future__ import annotations

from typing import List, Optional, Dict, Any
import os
import base64

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.pdf_service import process_pdf
from services.llava_service import summarize_image, synthesize_answer, get_llava_model
from services.vector_store import VectorStore


app = FastAPI(title="Multimodal RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


vector_store = VectorStore()
# Lazily initialize LLaVA to avoid startup failure if Ollama isn't available
llava = None


class IngestResponse(BaseModel):
    files: List[str]
    text_count: int
    table_count: int
    image_count: int


@app.post("/ingest_pdf", response_model=IngestResponse)
async def ingest_pdf(files: List[UploadFile] = File(...)):
    all_texts: List[str] = []
    all_tables: List[str] = []
    all_images_b64: List[str] = []
    ingested_files: List[str] = []

    for f in files:
        save_dir = os.path.abspath("uploaded_files")
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f.filename)
        with open(file_path, "wb") as out:
            out.write(await f.read())

        texts, tables, images_b64 = process_pdf(file_path)
        ingested_files.append(f.filename)

        # Summarize images using LLaVA (if available)
        image_summaries: List[str] = []
        # Try to create the model on first demand
        global llava
        if llava is None:
            try:
                llava = get_llava_model()
            except Exception:
                llava = None

        for img_b64 in images_b64:
            if llava is not None:
                try:
                    image_summaries.append(summarize_image(img_b64, model=llava))
                except Exception:
                    image_summaries.append("Image summary unavailable (LLaVA error)")
            else:
                image_summaries.append("Image summary unavailable (LLaVA not available)")

        # For simplicity, store raw text/table chunks as-is (they are already retrieval-friendly)
        # You can optionally summarize them with a text model if desired

        # Upsert documents into vector store with metadata linking back to raw items
        metadatas = []
        documents = []

        for t in texts:
            documents.append(t)
            metadatas.append({
                "source_file": f.filename,
                "modality": "text",
            })

        for tbl in tables:
            documents.append(tbl)
            metadatas.append({
                "source_file": f.filename,
                "modality": "table",
            })

        for idx, (img_b64, img_sum) in enumerate(zip(images_b64, image_summaries)):
            documents.append(img_sum)
            metadatas.append({
                "source_file": f.filename,
                "modality": "image",
                "image_b64": img_b64,
                "image_index": idx,
            })

        if documents:
            vector_store.upsert_documents(documents=documents, metadatas=metadatas)

        all_texts.extend(texts)
        all_tables.extend(tables)
        all_images_b64.extend(images_b64)

    return IngestResponse(
        files=ingested_files,
        text_count=len(all_texts),
        table_count=len(all_tables),
        image_count=len(all_images_b64),
    )


class SummarizeImageRequest(BaseModel):
    image_b64: str


class SummarizeImageResponse(BaseModel):
    summary: str


@app.post("/summarize_image", response_model=SummarizeImageResponse)
async def summarize_image_endpoint(payload: SummarizeImageRequest):
    global llava
    if llava is None:
        try:
            llava = get_llava_model()
        except Exception:
            llava = None
    if llava is None:
        return SummarizeImageResponse(summary="Image summary unavailable (LLaVA not available)")
    summary = summarize_image(payload.image_b64, model=llava)
    return SummarizeImageResponse(summary=summary)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 6


class QueryResponse(BaseModel):
    answer: str
    references: List[Dict[str, Any]]


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
    docs, metas, ids = vector_store.similarity_search(payload.question, k=payload.top_k)

    text_chunks: List[str] = []
    image_b64_list: List[str] = []
    references: List[Dict[str, Any]] = []

    for doc, meta, id_ in zip(docs, metas, ids):
        references.append({"id": id_, **(meta or {})})
        if meta and meta.get("modality") == "image":
            b64 = meta.get("image_b64")
            if b64:
                image_b64_list.append(b64)
        else:
            text_chunks.append(doc)

    global llava
    if llava is None:
        try:
            llava = get_llava_model()
        except Exception:
            llava = None

    if llava is not None:
        try:
            answer = synthesize_answer(
                question=payload.question,
                text_chunks=text_chunks,
                image_b64_list=image_b64_list,
                model=llava,
            )
        except Exception:
            answer = (
                ("\n\n".join(text_chunks[:3])[:1000])
                + "\n\n(Note: LLaVA error; answering with retrieved text excerpts.)"
            )
    else:
        # Fallback: return concatenated text snippets if multimodal model unavailable
        answer = (
            ("\n\n".join(text_chunks[:3])[:1000])
            + "\n\n(Note: LLaVA not available; showing retrieved text excerpts.)"
        )

    return QueryResponse(answer=answer, references=references)


@app.get("/health")
async def health():
    return {"status": "ok"}

