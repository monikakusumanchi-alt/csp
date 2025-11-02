import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType
import streamlit as st

# --- Qdrant Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "YOUR_QDRANT_URL_HERE")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "YOUR_QDRANT_API_KEY_HERE")
QDRANT_COLLECTION_NAME = "pdf_embeddings_collection"

# --- Embedding Model ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


# --- Qdrant Client (OK at global level - uses fixed credentials) ---
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def get_api_key():
    key = st.session_state.get("openai_api_key") if "openai_api_key" in st.session_state else None
    return key

async def generate_embedding(text: str) -> list[float]:
    client = AsyncOpenAI(api_key=get_api_key())
    response = await client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding





async def prepare_qdrant_collection():
    """Ensure Qdrant collection exists and that 'metadata.page' is indexed."""
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Collection '{QDRANT_COLLECTION_NAME}' exists.")
        try:
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="metadata.page",
                field_schema=PayloadSchemaType.INTEGER
            )
        except Exception:
            pass
    except Exception:
        print(f"Creating collection '{QDRANT_COLLECTION_NAME}'...")
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
        )
        qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION_NAME,
            field_name="metadata.page",
            field_schema=PayloadSchemaType.INTEGER
        )

async def upload_pdf_to_qdrant(pdf_file_path: str):
    reader = PdfReader(pdf_file_path)
    points = []
    point_id_counter = 0

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            embedding = await generate_embedding(page_text)
            points.append(
                PointStruct(
                    id=point_id_counter,
                    vector=embedding,
                    payload={
                        "content": page_text,
                        "metadata": {
                            "page": i + 1,
                            "source_file": os.path.basename(pdf_file_path)
                        }
                    },
                )
            )
            point_id_counter += 1
            print(f"  - Embedded page {i + 1} ({len(page_text)} chars).")
        else:
            print(f"  - Page {i + 1} has no text, skipped.")

    if points:
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            wait=True,
            points=points,
        )
        print(f"Uploaded {len(points)} text chunks from PDF to Qdrant.")
    else:
        print("No text found in PDF to upload to Qdrant.")

async def process_pdfs_in_folder(folder_path: str):
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDFs found in folder.")
        return

    await prepare_qdrant_collection()

    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        await upload_pdf_to_qdrant(full_path)

async def qdrant_search_tool_function(query: str, top_k: int = 3):
    query_embedding = await generate_embedding(query)
    search_result = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
    results_text = []
    for r in search_result:
        page = r.payload.get("metadata", {}).get("page", "?")
        text = r.payload.get("content", "")
        results_text.append(f"(Page {page}) {text}")

    return "\n".join(results_text) if results_text else "No relevant results found."

async def qdrant_search_tool_on_invoke(tool_context, args_json_string: str) -> str:
    try:
        args = json.loads(args_json_string)
        query = args.get("query")
        top_k = args.get("top_k", 5)
        if not query:
            return "Error: 'query' is missing for QdrantPDFSearch tool."
    except json.JSONDecodeError:
        return f"Error: Invalid JSON args for QdrantPDFSearch tool: {args_json_string}"
    except Exception as e:
        return f"Error parsing QdrantPDFSearch tool args: {e}"

    try:
        print(f"Qdrant Search Tool: Query '{query}' top_k={top_k}")
        query_embedding = await generate_embedding(query)
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_vectors=False,
            with_payload=True
        )
        if not search_result:
            return "No relevant information found via Qdrant search."
        retrieved_content = []
        for hit in search_result:
            content = hit.payload.get("content", "")
            page = hit.payload.get("metadata", {}).get("page", "unknown")
            score = hit.score
            retrieved_content.append(f"Content (Page {page}, Score: {score:.2f}):\n{content}\n---")
        return "\n\n".join(retrieved_content)
    except Exception as e:
        print(f"Error during Qdrant search: {str(e)}")
        return f"Error performing Qdrant search: {str(e)}"

# Uncomment if you want standalone PDF upload:
# if __name__ == "__main__":
#     asyncio.run(process_pdfs_in_folder("/workspace/csp/data"))
