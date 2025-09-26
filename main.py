from pyexpat import model
from tkinter import Place
import streamlit as st
from dotenv import load_dotenv
import os
from sympy import im
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
from base64 import b64encode
import base64
from vectorDB import add_to_vectorstore, retriever
from langchain.chains import ConversationalRetrievalChain


def process_file_into_texts_tables_imagesb64(file_path, file_name):
    """
    this function takes in a file path and file name of a pdf file,
    and returns three lists: texts, tables, images in base64 format.
    It uses unstructured to extract text, tables and images in base64 format.
    The images are also saved in a folder named extracted_images under the current directory.
    The images are saved with the file name as the original file name with an index appended to it.
    e.g. if the original file name is sample.pdf, the images will be saved as sample_img_0.png, sample_img_1.png, etc.
    """

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
                    tables.append(el.metadata.text_as_html)
                elif "Image" in el_type:
                    images_b64.append(el.metadata.image_base64)
                else:
                    texts.append(str(el))
        
        elif "Table" in chunk_type:
            # Standalone table element
            tables.append(chunk.metadata.text_as_html)
                    
        elif "Image" in chunk_type:
            # Standalone image element outside composite
            images_b64.append(chunk.metadata.image_base64)
        
        else:
            # Treat everything else as plain text element
            texts.append(str(chunk))
    
    image_output_dir = "extracted_images/" + os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(image_output_dir, exist_ok=True)

    # this is a function to decode base64 and save as image file under extracted_images folder
    def _save_b64_image(image_b64: str, idx: int) -> str:
        base_name = os.path.basename(file_path)
        name_no_ext = os.path.splitext(base_name)[0]
        out_path = os.path.join(image_output_dir, f"{name_no_ext}_img_{idx}.png")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
    
    for idx,image_b64 in enumerate(images_b64):
        _save_b64_image(image_b64, idx) 

    return texts, tables, images_b64

# Generate summaries of image elements
from io import BytesIO
import requests
from PIL import Image

def generate_image_summaries(file_path: str):
    """
    Generates summaries for all images in a directory using a local Llava model.

    Args:
        image_directory (str): The path to the directory containing images.

    Returns:
        dict: A dictionary mapping image filenames to their generated summaries.
              Returns an empty dictionary if the directory doesn't exist or an error occurs.
    """
    image_directory = "extracted_images/" + os.path.splitext(os.path.basename(file_path))[0]

    if not os.path.isdir(image_directory):
        print(f"Error: Directory not found at '{image_directory}'")
        return {}

    summaries = []
    ollama_api_url = "http://localhost:11434/api/generate"
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    # Iterate over all files in the given directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(image_directory, filename)
            print(f"Processing {filename}...")

            try:
                # Open image and convert to a base64 string
                with Image.open(image_path) as img:
                    # Ensure image is in RGB format
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Prepare the payload for the Ollama API
                payload = {
                    "model": "llava",
                    "prompt": "You are a helpful assistant that generates detailed summaries for images. These summaries should be descriptive and provide context about the content of the image.They should be optimized for fast retrieval of the image3 embeddings.",
                    "images": [image_b64],
                    "stream": False
                }

                # Make the request to the Ollama server
                response = requests.post(ollama_api_url, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Extract the summary from the response
                response_data = response.json()
                summary = response_data.get("response", "Could not generate summary.").strip()
                summaries.append(summary)
                print(f"  -> Summary: {summary}")

            except requests.exceptions.RequestException as e:
                print(f"Error communicating with Ollama API for {filename}: {e}")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

    return summaries 

def load_model(model_name):
    # Load the model
    llm=ChatOllama(model=model_name, temperature=0)
    return llm

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

def main():
    load_dotenv()
    st.set_page_config(
        page_title="MultiDoc RagChat",
        page_icon=":Multiple books:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # api_key = os.getenv("GOOGLE_API_KEY")
    # if not api_key:
    #     raise ValueError("GOOGLE_API_KEY not found in environment variables")

    st.title("Chat with Multiple pdfs:")
    st.header("First upload your pdf files in the sidebar, then ask your query below.")
    

    with st.sidebar:

        st.session_state.llm_model_name = st.selectbox("Right now we only support llava local model. More models will be coming soon!",
                                                       ("llava:latest",
                                                        # "gemini-1.5-flash-latest",
                                                        # "gemini-2.5-pro",
                                                        # "gemini-pro",
                                                        # "gemini-1.5-pro"
                                                        ), 
                                                        key="llm_model",
                                                        placeholder="Select a model")
        
        initialize_model = st.button("Initialize Model")

        if initialize_model and not st.session_state.llm_model_name:
            st.error("Please select a model before initializing.")

        if initialize_model and st.session_state.llm_model_name:     
            
            if st.session_state.llm_model_name == "llava:latest":
                st.session_state.model = load_model("llava:latest")
                st.success(f"Model {st.session_state.llm_model_name} initialized successfully!")
            else:
                st.success(f"Currently we do not support any other model because of the api costing factors. Please select llava:latest model.")
            
            


        st.subheader("Your Documents:")
        files = st.file_uploader("Upload your pdf files and start chatting!",
                                            type=["pdf"], 
                                            accept_multiple_files=True
                                            )
        button = st.button("Process")
        if button:
            if files:
                # st.session_state["uploaded_files"] = files

                # Create directory if it doesn't exist
                save_dir = "uploaded_files"
                os.makedirs(save_dir, exist_ok=True)

                for file in files:
                    # Construct full file path
                    file_path = os.path.join(save_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    texts, tables, images_b64 = process_file_into_texts_tables_imagesb64(file_path,file.name)
                    image_summaries = generate_image_summaries(file_path)
                    add_to_vectorstore(texts, tables, images_b64,image_summaries)
                    st.success(f"{file.name} processed successfully!")

                st.success("Files uploaded successfully!")


            else:
                st.error("Please upload at least one pdf file.")
    
    st.session_state.text_input = st.text_input("Ask a query about your pdf files", placeholder="Type your query here...", key="input")

    submit = st.button("Submit")

    if submit:

        from answer_synthesis import rag_chain

        response = rag_chain.invoke(st.session_state.text_input)

        st.write(response)
  
    if st.button("Reload"):
        st.rerun()


if __name__ == "__main__":
  main()

