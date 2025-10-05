from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage
import os


# Assuming 'retriever' is correctly imported and configured from your vectorDB.py
from vectorDB import retriever 

# --- 1. Initialize the LLM ---
# This connects to your locally running Ollama server
# Old behavior: hard-coded model name "llava".
# llm = ChatOllama(model="llava", temperature=0)
# New behavior: read model name from env (set by UI) with fallback to "llava" to keep defaults working.
_model_name = os.getenv("OLLAMA_MODEL_NAME", "llava")
llm = ChatOllama(model=_model_name, temperature=0)

# --- 2. Define the Multimodal Input Formatting Function ---
# def format_multimodal_input(inputs: dict) -> HumanMessage:
#     """
#     Takes the retrieved documents and the user's question and formats
#     them into a single HumanMessage for the multimodal LLM.
#     """

#     # The retriever returns Documents from the docstore with metadata and page_content.
#     # (We updated the docstore to store full Document objects.)
#     retrieved_docs = inputs["context"]
#     question = inputs["question"]
    
#     # Start building the text part of the prompt
#     text_context = ""
    
#     # Start the message content list
#     message_content = []
    
#     # Iterate through the retrieved documents
#     for doc in retrieved_docs:
#         # Check the metadata for the content type
#         doc_type = doc.metadata.get("type", "text") # Default to 'text' if no type
        
#         if doc_type == "image":
#             # If it's an image, add it to the message content as an image part
#             message_content.append(
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{doc.page_content}"}
#                 }
#             )
#         elif doc_type in ["text", "table"]:
#             # If it's text or a table, add its content to our text context string
#             text_context += doc.page_content + "\n\n"
            
#     # Add the final combined text prompt (question + text context) as the first part
#     final_prompt_text = f"Question: {question}\n\nUse the following text and images to answer the question.\n\nText Context:\n{text_context}"
#     message_content.insert(0, {"type": "text", "text": final_prompt_text})
        
#     return [HumanMessage(content=message_content)]

def format_multimodal_input(inputs: dict) -> HumanMessage:
    """
    Takes the retrieved documents and the user's question and formats
    them into a single HumanMessage for the multimodal LLM.
    """
    # The retriever returns Documents from the docstore with metadata and page_content.
    # (We updated the docstore to store full Document objects.)
    retrieved_docs = inputs["context"]
    question = inputs["question"]
    
    # Start building the text part of the prompt
    text_context = ""
    
    # Start the message content list
    message_content = []
    
    # Iterate through the retrieved documents
    for doc in retrieved_docs:
        # Robustness: handle both Document and raw string (edge cases)
        if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
            doc_type = doc.metadata.get("type", "text")
            content = doc.page_content
        else:
            # Fallback if non-Document slips through (should not happen after our storage fix)
            doc_type = "text"
            content = str(doc)
        
        if doc_type == "image":
            # If it's an image, add it to the message content as an image part
            message_content.append(
                {
                    "type": "image_url",
                    # We stored parent image docs' page_content as base64. Keep data URL inline for Llava.
                    "image_url": {"url": f"data:image/png;base64,{content}"}
                }
            )
        elif doc_type in ["text", "table"]:
            # If it's text or a table, add its content to our text context string
            text_context += content + "\n\n"
            
    # Add the final combined text prompt (question + text context) as the first part
    final_prompt_text = f"Question: {question}\n\nUse only the following text and images to answer the question, if you can't form an answer using these, don't answer at all but do reply as that \"the provided context do not have the information you ask for\".\n\nText Context:\n{text_context}"
    message_content.insert(0, {"type": "text", "text": final_prompt_text})
        
    return [HumanMessage(content=message_content)]

# --- 3. Build the RAG Chain ---
# This chain is now simpler as it doesn't reference the unused prompt template.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(format_multimodal_input)  # Formats the input message for Llava
    | llm                      # Passes the HumanMessage to the Llava model
    | StrOutputParser()        # Parses the LLM's output into a string
)

