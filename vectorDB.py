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


def add_to_vectorstore(texts, tables, images, image_summaries):
    """Add texts, tables, and images to the vectorstore and parent docstore.

    - Child documents go to the vector store (what gets embedded/searched)
    - Full parent documents (with metadata) go to the docstore (what gets returned)
    """
    # Add text documents
    if texts:
        text_ids = [str(uuid.uuid4()) for _ in texts]
        text_child_docs = [
            Document(page_content=text, metadata={id_key: text_ids[i], "type": "text"})
            for i, text in enumerate(texts)
        ]
        retriever.vectorstore.add_documents(text_child_docs)
        text_parent_docs = [
            Document(page_content=text, metadata={"type": "text"}) for text in texts
        ]
        retriever.docstore.mset(list(zip(text_ids, text_parent_docs)))

    # Add tables
    if tables:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        table_child_docs = [
            Document(page_content=table_html, metadata={id_key: table_ids[i], "type": "table"})
            for i, table_html in enumerate(tables)
        ]
        retriever.vectorstore.add_documents(table_child_docs)
        table_parent_docs = [
            Document(page_content=table_html, metadata={"type": "table"}) for table_html in tables
        ]
        retriever.docstore.mset(list(zip(table_ids, table_parent_docs)))

    # Add image summaries and raw image payloads
    if images and image_summaries:
        count = min(len(images), len(image_summaries))
        img_ids = [str(uuid.uuid4()) for _ in range(count)]
        image_child_docs = [
            Document(page_content=image_summaries[i], metadata={id_key: img_ids[i], "type": "image"})
            for i in range(count)
        ]
        retriever.vectorstore.add_documents(image_child_docs)
        image_parent_docs = [
            Document(page_content=images[i], metadata={"type": "image"})
            for i in range(count)
        ]
        retriever.docstore.mset(list(zip(img_ids, image_parent_docs)))