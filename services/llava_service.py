from typing import List, Optional
import base64

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


def get_llava_model(model_name: str = "llava", temperature: float = 0.0) -> ChatOllama:
    """
    Returns a ChatOllama instance configured for LLaVA.
    Assumes the local Ollama runtime has the model pulled (e.g., `ollama pull llava`).
    """
    return ChatOllama(model=model_name, temperature=temperature)


def build_image_message_content(
    instruction: str,
    image_b64_list: List[str],
    mime_type: str = "image/png",
) -> List[dict]:
    """
    Build multimodal content for LangChain's HumanMessage that LLaVA understands via ChatOllama.
    Uses data URLs for images.
    """
    content_parts: List[dict] = []
    if instruction:
        content_parts.append({"type": "text", "text": instruction})
    for img_b64 in image_b64_list:
        # Ensure no whitespace/newlines in base64
        norm_b64 = "".join(img_b64.split())
        data_url = f"data:{mime_type};base64,{norm_b64}"
        content_parts.append({"type": "image_url", "image_url": data_url})
    return content_parts


def summarize_image(
    image_b64: str,
    model: Optional[ChatOllama] = None,
    instruction: str = (
        "You are an assistant tasked with summarizing images for retrieval. "
        "These summaries will be embedded and used to retrieve the raw image. "
        "Give a concise summary of the image that is optimized for retrieval."
    ),
) -> str:
    """
    Generate a concise, retrieval-optimized summary of a single image.
    """
    llm = model or get_llava_model()
    content = build_image_message_content(instruction=instruction, image_b64_list=[image_b64])

    messages = [HumanMessage(content=content)]
    response = llm.invoke(messages)
    return response.content


def synthesize_answer(
    question: str,
    text_chunks: List[str],
    image_b64_list: List[str],
    model: Optional[ChatOllama] = None,
    mime_type: str = "image/png",
) -> str:
    """
    Use a multimodal LLM (LLaVA) to synthesize an answer given a question,
    relevant text chunks, and associated raw images.
    """
    llm = model or get_llava_model()

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using the provided \n"
        "text context and images. Prefer precise, grounded answers. If not enough \n"
        "information is present, say so explicitly."
    )

    text_context = "\n\n".join(text_chunks[:6]) if text_chunks else ""
    text_instruction = (
        f"Context (text):\n{text_context}\n\n"
        f"Question: {question}\n"
        "Provide a concise, accurate answer."
    )

    content = build_image_message_content(
        instruction=text_instruction,
        image_b64_list=image_b64_list[:6],
        mime_type=mime_type,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]
    response = llm.invoke(messages)
    return response.content

