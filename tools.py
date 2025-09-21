import asyncio
import os
import io
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from agents import Agent, Runner, trace
from agents.tool import FunctionTool, ToolContext


# Load environment variables
load_dotenv()

# --- Qdrant Configuration ---

# --- Qdrant Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "https://dec76c01-c3b9-4df2-9258-e637aaa81b52.us-west-2-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ca7mlWcfsg_V6196gwkQ6fIfB2py3fq31OwkmE99H6Q")
QDRANT_COLLECTION_NAME = "pdf_embeddings_collection"

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


async def generate_embedding(text: str) -> list[float]:
    """Generate embeddings from OpenAI API."""
    response = await openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


async def prepare_qdrant_collection():
    """Ensure collection exists and payload index is ready."""
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")
        try:
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="metadata.page",
                field_schema=PayloadSchemaType.INTEGER
            )
        except Exception:
            pass  # Likely already exists
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
    """Parses a PDF, generates embeddings for each page, and uploads to Qdrant."""
    print(f"Processing PDF: {pdf_file_path}")
    reader = PdfReader(pdf_file_path)
    points = []
    point_id_counter = 0

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text.strip(): # Only process pages with actual text
            embedding = await generate_embedding(page_text)
            points.append(
                PointStruct(
                    id=point_id_counter,
                    vector=embedding,
                    payload={
                        "content": page_text,
                        "metadata": {"page": i + 1, "source_file": os.path.basename(pdf_file_path)}
                    },
                )
            )
            point_id_counter += 1
            print(f"  - Embedded page {i + 1} with {len(page_text)} chars.")
        else:
            print(f"  - Page {i + 1} has no extractable text, skipping.")

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
    """Process all PDFs in a given folder."""
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDFs found in folder.")
        return

    await prepare_qdrant_collection()

    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        await upload_pdf_to_qdrant(full_path)


async def main():
    folder_path = "/workspace/csp/data"  # 📂 Change this to your folder
    await process_pdfs_in_folder(folder_path)

# --- UPDATED qdrant_search_tool_on_invoke function ---
async def qdrant_search_tool_function(query: str, top_k: int = 3):
    query_embedding = await generate_embedding(query)

    search_result = qdrant_client.search(
        collection_name="pdf_embeddings_collection",
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=False  # ✅ don't use append_vectors
    )

    # Format the output nicely for the agent
    results_text = []
    for r in search_result:
        page = r.payload.get("metadata", {}).get("page", "?")
        text = r.payload.get("text", "")
        results_text.append(f"(Page {page}) {text}")

    return "\n".join(results_text) if results_text else "No relevant results found."

async def qdrant_search_tool_on_invoke(tool_context: ToolContext[dict], args_json_string: str) -> str:
    """
    Function to be passed as on_invoke_tool for FunctionTool.
    Parses arguments from LLM, searches Qdrant, and returns results.
    """
    print(f"ToolContext received: {tool_context}") # You can inspect context if needed

    try:
        # Parse the JSON arguments provided by the LLM
        args = json.loads(args_json_string)
        query = args.get("query")
        top_k = args.get("top_k", 5) # Default to 5 if top_k is not provided by LLM

        if not query:
            return "Error: 'query' parameter is missing for QdrantPDFSearch tool."

    except json.JSONDecodeError:
        return f"Error: Invalid JSON arguments provided to QdrantPDFSearch tool: {args_json_string}"
    except Exception as e:
        return f"Error parsing arguments for QdrantPDFSearch tool: {e}"

    print(f"Qdrant Search Tool: Searching for '{query}' with top_k={top_k}")
    query_embedding = await generate_embedding(query)
    search_result = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_vectors=False,
        with_payload=True,
    )

    if not search_result:
        return "No relevant information found in the document via Qdrant search."

    retrieved_content = []
    for hit in search_result:
        content = hit.payload.get("content", "")
        page = hit.payload.get("metadata", {}).get("page", "unknown")
        score = hit.score
        retrieved_content.append(f"Content (Page {page}, Score: {score:.2f}):\n{content}\n---")

    return "\n\n".join(retrieved_content)

# if __name__ == "__main__":
#     asyncio.run(main())
