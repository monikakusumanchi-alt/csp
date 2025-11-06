import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType
import streamlit as st
import uuid

# --- Load environment variables ---
load_dotenv()

# --- Qdrant Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "https://5a4f5665-f0f9-488c-be5a-4d7dff7c07e1.eu-west-2-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7885TTkf9K01O0UL2NDU7NmCjj7E-RWlY3Axqdle-Bg")
QDRANT_COLLECTION_NAME = "pdf_embeddings_collection"

# --- Embedding Model ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # can be auto-updated if model changes


# --- Qdrant Client ---
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# --- Helper: Get API key dynamically from Streamlit session ---
def get_api_key():
    return st.session_state.get("openai_api_key")


# --- Generate embeddings using OpenAI ---
async def generate_embedding(text: str) -> list[float]:
    """Generate embeddings for given text using the current OpenAI API key."""
    client = AsyncOpenAI(api_key=get_api_key())
    response = await client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


# ✅ --- Ensure collection exists (idempotent) ---
def ensure_qdrant_collection(collection_name: str = QDRANT_COLLECTION_NAME, vector_size: int = EMBEDDING_DIMENSION):
    """Ensure Qdrant collection exists with correct configuration."""
    try:
        collections = qdrant_client.get_collections().collections
        existing_names = [c.name for c in collections]
        print(existing_names)
        if collection_name not in existing_names:
            print(f"🆕 Creating new collection '{collection_name}'...")
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Collection '{collection_name}' created successfully.")
        else:
            print(f"✅ Collection '{collection_name}' already exists.")

        # Ensure index on metadata.page
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.page",
                field_schema=PayloadSchemaType.INTEGER
            )
            print("📘 Index created on 'metadata.page'.")
        except Exception:
            print("ℹ️ Index 'metadata.page' already exists (skipped).")

    except Exception as e:
        print(f"❌ Error ensuring Qdrant collection: {e}")


import io
import os
import uuid
import asyncio
from PyPDF2 import PdfReader
from qdrant_client.models import PointStruct
from pdf2image import convert_from_bytes
import pytesseract
import streamlit as st

# --- Upload PDF to Qdrant ---
async def upload_pdf_to_qdrant(uploaded_file, qdrant_client=None):
    """Reads a PDF (including image-based ones), generates embeddings, and uploads to Qdrant."""
    
    # ✅ Use Qdrant client from session state if not passed
    if qdrant_client is None:
        qdrant_client = st.session_state.get("qdrant_client")

    if not qdrant_client:
        return 0, "❌ Qdrant client not initialized."

    ensure_qdrant_collection(qdrant_client, QDRANT_COLLECTION_NAME)

    # ✅ Wrap uploaded file in a BytesIO object
    pdf_bytes = uploaded_file.read()
    pdf_stream = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_stream)
    points = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()

        # 🧠 OCR fallback if page has no text (image-based page)
        if not page_text or not page_text.strip():
            print(f"⚠️ Page {i+1} has no text — using OCR fallback.")
            try:
                images = convert_from_bytes(pdf_bytes, first_page=i+1, last_page=i+1)
                if images:
                    ocr_text = pytesseract.image_to_string(images[0])
                    page_text = ocr_text.strip()
            except Exception as ocr_err:
                print(f"❌ OCR failed on page {i+1}: {ocr_err}")
                continue

        if not page_text:
            print(f"⚠️ Page {i+1} has no extractable text, skipped.")
            continue

        embedding = await generate_embedding(page_text)
        point_id = str(uuid.uuid4())

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": page_text,
                    "metadata": {
                        "page": i + 1,
                        "source_file": uploaded_file.name
                    },
                },
            )
        )
        print(f"✅ Embedded page {i+1} ({len(page_text)} chars).")

    if points:
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            wait=True,
            points=points,
        )
        msg = f"📤 Uploaded {len(points)} pages from '{uploaded_file.name}' to Qdrant."
        print(msg)
        return len(points), msg
    else:
        msg = f"⚠️ No text found in '{uploaded_file.name}'."
        print(msg)
        return 0, msg


# --- Process all PDFs in folder ---
async def process_pdfs_in_folder(folder_path: str):
    """Process all PDF files in the given folder and upload them to Qdrant."""
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No PDFs found in folder.")
        return

    ensure_qdrant_collection(qdrant_client, QDRANT_COLLECTION_NAME)

    for pdf_file in pdf_files:
        await upload_pdf_to_qdrant(os.path.join(folder_path, pdf_file))


# --- Search Qdrant ---
async def qdrant_search_tool_function(query: str, top_k: int = 3):
    """Search Qdrant for documents relevant to the query."""
    ensure_qdrant_collection(qdrant_client, QDRANT_COLLECTION_NAME)  # ✅ Ensure before search

    query_embedding = await generate_embedding(query)
    search_result = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for hit in search_result:
        meta = hit.payload.get("metadata", {})
        content = hit.payload.get("content", "")
        results.append(f"(Page {meta.get('page', '?')}) {content}")

    return "\n".join(results) if results else "No relevant results found."

import json

async def qdrant_search_tool_on_invoke(tool_context, args_json_string: str) -> str:
    """
    Safe wrapper around qdrant_search_tool_function to handle non-serializable ToolContext.
    """
    try:
        # Handle both dict and JSON-string args
        if isinstance(args_json_string, str):
            args = json.loads(args_json_string)
        elif isinstance(args_json_string, dict):
            args = args_json_string
        else:
            raise ValueError(f"Unsupported args type: {type(args_json_string)}")

        query = args.get("query")
        top_k = args.get("top_k", 5)

        if not query:
            return "Error: Missing 'query' parameter for Qdrant search."

        return await qdrant_search_tool_function(query, top_k)

    except Exception as e:
        return f"Error invoking Qdrant search tool: {e}"
