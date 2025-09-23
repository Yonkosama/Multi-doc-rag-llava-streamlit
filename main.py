from pyexpat import model
import streamlit as st
from dotenv import load_dotenv
import os
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama.chat_models import ChatOllama
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


def process_file_into_texts_tables_imagesb64(file_path):
    #using unstructured to extract text, tables and images in base64 format
    # Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
    chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True,            # extract tables
    strategy="hi_res",                     # mandatory to infer tables
    
    extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
    # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

    chunking_strategy="by_title",          # or 'basic'
    max_characters=10000,                  # defaults to 500
    combine_text_under_n_chars=2000,       # defaults to 0
    new_after_n_chars=6000,

    # extract_images_in_pdf=True,          # deprecated
    )
    #parsing individual chunks into text, tables, images in base64 format
    tables = []
    texts = []
    images_b64 = []

    for chunk in chunks:
        chunk_type = str(type(chunk))
        
        if "CompositeElement" in chunk_type:
            # Composite element may contain multiple sub-elements
            # Separate tables, texts, images inside the composite
            chunk_els = getattr(chunk.metadata, "orig_elements", [])
            
            # Collect tables inside the composite
            for el in chunk_els:
                el_type = str(type(el))
                if "Table" in el_type:
                    tables.append(el)
                elif "Image" in el_type:
                    images_b64.append(el.metadata.image_base64)
                else:
                    texts.append(el)
        
        elif "Table" in chunk_type:
            # Standalone table element
            tables.append(chunk)
                    
        elif "Image" in chunk_type:
            # Standalone image element outside composite
            images_b64.append(chunk.metadata.image_base64)
        
        else:
            # Treat everything else as plain text element
            texts.append(chunk)

    return texts, tables, images_b64

# Generate summaries of text elements
def generate_summaries(texts, tables, image_b64, summarize_texts=False, summarize_tables=False, summarize_images=False, model=None):
    """
    First we summarize the text and table elements using the LLM.
    Then we summarize the images using a separate function.
    The summaries are optimized for retrieval.
    """

    
    #Summarize text elements
    #texts: List of str
    #tables: List of str
    #summarize_texts: Bool to summarize texts
    
    
    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = PromptTemplate.from_template(prompt_text)
    empty_response = RunnableLambda(
        lambda x: AIMessage(content="Error processing document")
    )
    # Text summary chain
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
    elif texts:
        text_summaries = texts
    if text_summaries :
        st.write("Text summaries generated successfully.")

    # Apply to tables if tables are provided
    if tables and summarize_tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})
    elif tables:
        table_summaries = tables
    if table_summaries :
        st.write("table summaries generated successfully.")

    #Summarize image elements

    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    def image_summarize(image_b64, prompt):
        # Create messages
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Image (base64): {image_b64}"),
        ]
        try:
            response = model.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error processing image: {str(e)}"
    # Apply to images
    for img_b64 in image_b64:
        image_summaries.append(image_summarize(img_b64, prompt))
    
    if image_summaries :
        st.write("Image summaries generated successfully.")
    
    return text_summaries, table_summaries, image_summaries


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Multi-RAG-Bhaisaab",
        page_icon=":Prism:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # api_key = os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     raise ValueError("GOOGLE_API_KEY not found in environment variables")

    st.title("Chat with Multiple pdfs:")
    st.header("First upload your pdf files in the sidebar, then ask your query below.")
    

    with st.sidebar:

        st.session_state.llm_model_name = st.selectbox("Choose your LLM:",
                                                       ("Llava(Local model)",
                                                        # "gemini-1.5-flash-latest",
                                                        # "gemini-2.5-pro",
                                                        # "gemini-pro",
                                                        # "gemini-1.5-pro"
                                                        ), 
                                                        key="llm_model")
        initialize_model = st.button("Initialize Model")

        if initialize_model and not st.session_state.llm_model_name:
            st.error("Please select a model before initializing.")

        if initialize_model and st.session_state.llm_model_name:     
            def load_model(model_name):
                # Load the model
                llm=ChatGoogleGenerativeAI(model=model_name, temperature=0, max_output_tokens=1024)
                return llm
            if st.session_state.llm_model_name == "Llava(Local model)":
                st.session_state.model = ChatOllama(model="llava")
            else:
                st.session_state.model = load_model(st.session_state.llm_model_name)
            if st.session_state.model:
                st.success(f"Model {st.session_state.llm_model_name} initialized successfully!")

            store = {}  # memory is maintained outside the chain

            def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
                if session_id not in store:
                    store[session_id] = InMemoryChatMessageHistory()
                    return store[session_id]

                memory = ConversationBufferWindowMemory(
                    chat_memory=store[session_id],
                    k=3,
                    return_messages=True,
                )
                assert len(memory.memory_variables) == 1
                key = memory.memory_variables[0]
                messages = memory.load_memory_variables({})[key]
                store[session_id] = InMemoryChatMessageHistory(messages=messages)
                return store[session_id]

            st.session_state.Conversation_chain_with_memory = RunnableWithMessageHistory(st.session_state.model, get_session_history)
            
            


        st.subheader("Your Documents:")
        files = st.file_uploader("Upload your pdf files and start chatting!",
                                            type=["pdf"], 
                                            accept_multiple_files=True
                                            )
        button = st.button("Process")
        if button:
            if files:
                st.session_state["uploaded_files"] = files

                # Create directory if it doesn't exist
                save_dir = "uploaded_files"
                os.makedirs(save_dir, exist_ok=True)

                for file in files:
                    # Construct full file path
                    file_path = os.path.join(save_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    texts, tables, images_b64 = process_file_into_texts_tables_imagesb64(file_path)
                    text_summaries, table_summaries, image_summaries = generate_summaries(texts, tables, images_b64, 
                                                                                          summarize_texts=True, summarize_tables=True, summarize_images=True, 
                                                                                          model=st.session_state.model)
                
                    st.success("Files uploaded successfully!")
                    st.write(texts[0][:100],"/n", text_summaries[0])
                    st.write(tables[0][:100],"/n", table_summaries[0])
                    st.write(images_b64[0][:100],"/n", image_summaries[0])

            else:
                st.error("Please upload at least one pdf file.")
    
    st.session_state.text_input = st.text_input("Ask a query about your pdf files", placeholder="Type your query here...", 
                    key="input"
                    )
    submit = st.button("Submit")
    if submit:
        response = st.session_state.Conversation_chain_with_memory.invoke(
                f"{st.session_state.text_input}",
                config={"configurable": {"session_id": "1"}},
            )  # session_id determines thread
        st.write(response.content)
  
    if st.button("Reload"):
        st.rerun()


if __name__ == "__main__":
  main()

