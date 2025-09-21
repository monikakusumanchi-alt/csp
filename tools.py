import asyncio
import os
import io
import json # Import json for parsing the arguments
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from agents import Agent, Runner, trace
# Import ToolContext if you plan to use it within on_invoke_tool
from agents.tool import FunctionTool, ToolContext 


# Load environment variables
load_dotenv()

# --- Qdrant Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "https://dec76c01-c3b9-4df2-9258-e637aaa81b52.us-west-2-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ca7mlWcfsg_V6196gwkQ6fIfB2py3fq31OwkmE99H6Q")
QDRANT_COLLECTION_NAME = "pdf_embeddings_collection"

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536 # Dimension for text-embedding-3-small

# Initialize OpenAI client for embeddings
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) # Use AsyncOpenAI instead of OpenAI

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

async def generate_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using OpenAI's API."""
    response = await openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

async def prepare_qdrant_collection():
    """Ensures the Qdrant collection exists and has the necessary indexes."""
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")
        
        # Try to create index for 'metadata.page' if it doesn't exist
        try:
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name="metadata.page",
                field_schema=PayloadSchemaType.INTEGER
            )
            print(f"Payload index for 'metadata.page' ensured.")
        except Exception as e:
            print(f"Warning: Could not create payload index for 'metadata.page' (might already exist or schema conflict): {e}")

    except Exception: # Collection does not exist
        print(f"Collection '{QDRANT_COLLECTION_NAME}' not found. Creating it...")
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
        )
        # Create index for metadata.page immediately after collection creation
        qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION_NAME,
            field_name="metadata.page",
            field_schema=PayloadSchemaType.INTEGER
        )
        print(f"Collection '{QDRANT_COLLECTION_NAME}' created with 'metadata.page' index.")
        
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


async def main():
    pdf_file_path = "/workspace/OpenAI_SDK/sample_menu.pdf"

    # Check if the PDF file exists
    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file not found at {pdf_file_path}")
        return

    # 1. Prepare Qdrant collection
    await prepare_qdrant_collection()

    # 2. Upload PDF content to Qdrant (this will also embed it)
    await upload_pdf_to_qdrant(pdf_file_path)

    # 3. Create the custom Qdrant search tool with proper schema and on_invoke_tool
    qdrant_search_tool = FunctionTool(
        name="QdrantPDFSearch",
        description="Searches a Qdrant vector database for information within PDF documents. "
                    "Input is a JSON object with 'query' (string) and optional 'top_k' (integer). "
                    "Returns relevant text chunks from the document.",
        params_json_schema={ # Define the expected JSON schema for the tool's parameters
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string to find information in the PDF."
                },
                "top_k": {
                    "type": "integer",
                    "description": "The maximum number of relevant results to retrieve (default is 5).",
                    "default": 5
                }
            },
            "required": ["query"] # 'query' is a mandatory parameter
        },
        on_invoke_tool=qdrant_search_tool_on_invoke # Pass the new function conforming to the signature
    )

    # Create an agent that uses the custom Qdrant search tool
    agent = Agent(
        name="Qdrant PDF Searcher",
        instructions="You are a helpful agent. Your primary role is to answer questions based "
                     "on information retrieved from PDF documents stored in a Qdrant vector database. "
                     "Use the 'QdrantPDFSearch' tool to find relevant content. Always cite page numbers if available. "
                     "If the information is not in the document, state that you cannot find it.",
        tools=[qdrant_search_tool],
    )

    with trace("Qdrant PDF search example"):
        print("\n### Asking the agent a question based on PDF content from Qdrant:\n")
        query = "What is the main topic discussed in the document?" # Customize this for your PDF content

        result = await Runner.run(
            agent, query
        )

        print("\n### Final output:\n")
        print(result.final_output)

        print("\n### Output items (raw search results from tool):\n")
        if result.new_items:
            for item in result.new_items:
                print(str(item.raw_item))
                print("-" * 20)
        else:
            print("No new items generated by the tool.")


# if __name__ == "__main__":
#     asyncio.run(main())