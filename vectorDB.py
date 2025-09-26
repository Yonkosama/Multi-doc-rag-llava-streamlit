import uuid
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

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


def add_to_vectorstore(texts, tables, images,image_summaries):
    """Add texts, tables, and images to the vectorstore with their summaries."""
    # Add text documents
    if texts:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        texts_vectore_docs = [
            Document(page_content=text, metadata={id_key: doc_ids[i] , "type": "text"}) for i, text in enumerate(texts)
        ]
        retriever.vectorstore.add_documents(texts_vectore_docs)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    if tables:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        table_vectore_docs = [
            Document(page_content=summary, metadata={id_key: table_ids[i], "type" : "table"}) for i, summary in enumerate(tables)
        ]
        retriever.vectorstore.add_documents(table_vectore_docs)
        retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    if images:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i], "type": "image"}) for i, summary in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, images)))