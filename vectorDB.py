import uuid
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
import os

# The vectorstore to use to index the child chunks

# This i used to do earlier but this does not persist.
#vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

persist_directory = ".chroma/multi_modal_rag"
os.makedirs(persist_directory,exist_ok = True)

vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory=".chroma/multi_modal_rag"
)

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
    # This tells the retriever to return the full Document object from the docstore
    docstore_document_id_key="doc_id"
)


# def add_to_vectorstore(texts, tables, images,image_summaries):
#     """Add texts, tables, and images to the vectorstore with their summaries."""
#     # Add text documents
#     if texts:
#         doc_ids = [str(uuid.uuid4()) for _ in texts]
#         texts_vectore_docs = [
#             Document(page_content=text, metadata={id_key: doc_ids[i] , "type": "text"}) for i, text in enumerate(texts)
#         ]
#         retriever.vectorstore.add_documents(texts_vectore_docs)
#         retriever.docstore.mset(list(zip(doc_ids, texts)))

#     # Add tables
#     if tables:
#         table_ids = [str(uuid.uuid4()) for _ in tables]
#         table_vectore_docs = [
#             Document(page_content=summary, metadata={id_key: table_ids[i], "type" : "table"}) for i, summary in enumerate(tables)
#         ]
#         retriever.vectorstore.add_documents(table_vectore_docs)
#         retriever.docstore.mset(list(zip(table_ids, tables)))

#     # Add image summaries
#     if images:
#         img_ids = [str(uuid.uuid4()) for _ in images]
#         summary_img = [
#             Document(page_content=summary, metadata={id_key: img_ids[i], "type": "image"}) for i, summary in enumerate(image_summaries)
#         ]
#         retriever.vectorstore.add_documents(summary_img)
#         retriever.docstore.mset(list(zip(img_ids, images)))

def add_to_vectorstore(texts, tables, images, image_summaries):
    """
    Add texts, tables, and images to the vectorstore with their summaries.

    Changes made:
    - Store parent documents in the docstore as langchain.schema.Document with metadata (type, doc_id) so the retriever returns
      full Documents that downstream code can use (metadata + page_content).
    - Align image IDs to the number of successfully generated summaries to avoid ID/content mismatch.
    - Persist the vectorstore after writes for durability.
    - Add minimal logging to observe how many items were added per modality.
    """
    added_texts = 0
    added_tables = 0
    added_images = 0

    # Add text documents
    if texts:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        texts_vector_docs = [
            Document(page_content=text, metadata={id_key: doc_ids[i], "type": "text"})
            for i, text in enumerate(texts)
        ]
        retriever.vectorstore.add_documents(texts_vector_docs)
        # Old behavior (storing raw strings) kept for reference:
        # retriever.docstore.mset(list(zip(doc_ids, texts)))
        # New behavior: store full parent Documents with metadata so downstream can access .metadata and .page_content
        parent_text_docs = [
            Document(page_content=texts[i], metadata={"doc_id": doc_ids[i], "type": "text"})
            for i in range(len(texts))
        ]
        retriever.docstore.mset(list(zip(doc_ids, parent_text_docs)))
        added_texts = len(texts)

    # Add tables
    if tables:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        table_vector_docs = [
            Document(page_content=summary, metadata={id_key: table_ids[i], "type": "table"})
            for i, summary in enumerate(tables)
        ]
        retriever.vectorstore.add_documents(table_vector_docs)
        # Old behavior (storing raw strings) kept for reference:
        # retriever.docstore.mset(list(zip(table_ids, tables)))
        parent_table_docs = [
            Document(page_content=tables[i], metadata={"doc_id": table_ids[i], "type": "table"})
            for i in range(len(tables))
        ]
        retriever.docstore.mset(list(zip(table_ids, parent_table_docs)))
        added_tables = len(tables)

    # Add image summaries
    if images:
        # Ensure we do not create mismatched IDs if some summaries failed to generate
        num_pairs = min(len(images), len(image_summaries))
        if len(images) != len(image_summaries):
            print(
                f"[vectorDB] Warning: images ({len(images)}) and image_summaries ({len(image_summaries)}) length mismatch. Using {num_pairs} aligned items."
            )
        if num_pairs > 0:
            img_ids = [str(uuid.uuid4()) for _ in range(num_pairs)]
            summary_img_docs = [
                Document(page_content=image_summaries[i], metadata={id_key: img_ids[i], "type": "image"})
                for i in range(num_pairs)
            ]
            retriever.vectorstore.add_documents(summary_img_docs)
            # Old behavior (storing raw b64 strings) kept for reference:
            # retriever.docstore.mset(list(zip(img_ids, images)))
            parent_image_docs = [
                Document(page_content=images[i], metadata={"doc_id": img_ids[i], "type": "image"})
                for i in range(num_pairs)
            ]
            retriever.docstore.mset(list(zip(img_ids, parent_image_docs)))
            added_images = num_pairs

    # Persist changes to disk so the collection survives restarts
    try:
        retriever.vectorstore.persist()
    except Exception as e:
        print(f"[vectorDB] Persist failed (non-fatal): {e}")

    print(
        f"[vectorDB] Added -> texts: {added_texts}, tables: {added_tables}, images: {added_images}"
    )