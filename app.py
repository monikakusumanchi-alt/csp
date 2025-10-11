import asyncio
import os
import time
import streamlit as st
from agents import Agent, Runner, SQLiteSession, ModelSettings
from agents.tool import WebSearchTool
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from agents.tool import Tool
from agno.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.reader.pdf_reader import PDFReader
from tools import qdrant_search_tool_on_invoke, QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY, generate_embedding
from agents.tool import FunctionTool, ToolContext 
import datetime

# RAG dependencies for file upload
from typing import List
from pypdf import PdfReader

# Setup components
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
from agno.tools import tool
import re
import yagmail

# ------------------------------
# File Upload RAG Functions
# ------------------------------
def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks by characters"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start += chunk_size - overlap
    
    return chunks

async def process_uploaded_file(file, qdrant_client, collection_name: str):
    """Process uploaded PDF or text files and add to Qdrant using OpenAI embeddings"""
    try:
        chunks = []
        
        if file.type == "application/pdf":
            reader = PdfReader(file)
            # Process each page separately to maintain page context
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    # Split page into smaller chunks if it's too large
                    page_chunks = split_text(page_text)
                    for chunk_idx, chunk in enumerate(page_chunks):
                        chunks.append({
                            "text": chunk,
                            "page": page_num + 1,
                            "chunk_idx": chunk_idx,
                            "source": file.name
                        })
                    
        elif file.type.startswith("text/"):
            text = file.read().decode("utf-8")
            text_chunks = split_text(text)
            for chunk_idx, chunk in enumerate(text_chunks):
                chunks.append({
                    "text": chunk,
                    "page": 1,  # Text files don't have pages
                    "chunk_idx": chunk_idx,
                    "source": file.name
                })
        else:
            return 0, f"Unsupported file type: {file.type}"
        
        if not chunks:
            return 0, "No text content found in file"
        
        # Generate embeddings and upload to Qdrant
        points = []
        
        # Get current point count to generate unique IDs
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            start_id = collection_info.points_count
        except:
            start_id = 0
        
        # Process chunks and generate embeddings
        for idx, chunk_data in enumerate(chunks):
            # Generate embedding using OpenAI (same as tools.py)
            embedding = await generate_embedding(chunk_data["text"])
            
            point = PointStruct(
                id=start_id + idx,
                vector=embedding,
                payload={
                    "content": chunk_data["text"],
                    "metadata": {
                        "page": chunk_data["page"],
                        "source_file": chunk_data["source"],
                        "chunk_index": chunk_data["chunk_idx"],
                        "uploaded_at": datetime.datetime.now().isoformat()
                    }
                }
            )
            points.append(point)
        
        # Upload to Qdrant in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=batch
            )
        
        return len(chunks), f"Successfully processed {file.name}"
        
    except Exception as e:
        return 0, f"Error processing {file.name}: {str(e)}"

def send_support_email(subject: str, body: str) -> str:
    """Send escalation emails to support team when customer issue requires human intervention."""
    try:
        sender_email = "kusumonika033@gmail.com"
        sender_password = "nobd atmo sjcs vwyr"

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, sender_email):
            return "Invalid email format."

        yag = yagmail.SMTP(user=sender_email, password=sender_password)
        yag.send(to=sender_email, subject=subject, contents=body)
        return "Email sent successfully!"

    except Exception as e:
        return f"Error sending email: {e}"

import json

async def send_support_email_async(tool_context, args):
    if isinstance(args, str):
        args = json.loads(args)
    return await asyncio.to_thread(
        send_support_email,
        subject=args["subject"],
        body=args["body"]
    )

send_support_email_tool = FunctionTool(
    name="SendSupportEmail",
    description=(
        "Sends an escalation email to the human support team when a customer issue requires manual intervention. "
        "Use this tool when the issue is complex, unusual, or outside the scope of the knowledge base. "
        "The email should summarize the customer's problem, list any troubleshooting already attempted, "
        "and provide recommendations or next steps for the support team."
    ),
    params_json_schema={
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["subject", "body"]
    },
    on_invoke_tool=send_support_email_async
)

knowledge_base_tool = FunctionTool(
    name="ProductKnowledgeSearch",
    description="Searches the company's product knowledge base for information about products, features, troubleshooting, "
                "common issues, setup instructions, and frequently asked questions. "
                "Use this tool to find accurate product information, solutions to customer problems, "
                "and detailed explanations from the company's official documentation. "
                "Input is a JSON object with 'query' (string) and optional 'top_k' (integer). "
                "Returns relevant information from product manuals, FAQs, and support documentation.",
    params_json_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for product information. Can include product names, features, error messages, setup steps, troubleshooting terms, or specific customer issues."
            },
            "top_k": {
                "type": "integer",
                "description": "The maximum number of relevant knowledge base entries to retrieve (default is 6 for comprehensive support).",
                "default": 6
            }
        },
        "required": ["query"]
    },
    on_invoke_tool=qdrant_search_tool_on_invoke
)


class CustomerSupportSystem:
    """Customer Support Multi-Agent System with Knowledge Base Integration"""

    def __init__(
        self,
        model="gpt-4o",
        temperature=0.1,
        session_id=None,
        db_file="customer_support_conversations.db"
    ):
        self.model = model
        self.temperature = temperature
        self.session_id = session_id or f"support_session_{int(time.time())}"
        self.db_file = db_file

        # ---- Tools ----
        self.web_search_tool = WebSearchTool()

        # ---- Memory Session ----
        self.session = SQLiteSession(self.session_id, self.db_file)

        # ---- Agents ----
        self.knowledge_agent = self._create_knowledge_agent()
        self.escalation_agent = self._create_escalation_agent()
        self.support_router = self._create_support_router()

    def _create_knowledge_agent(self):
        return Agent(
            name="ProductKnowledgeAgent",
            instructions="""
                You are the Product Knowledge Expert for customer support.
                Your role is to provide accurate, helpful information about our products using the knowledge base.
                
                Your responsibilities:
                - Answer product-related questions using the knowledge base search
                - Provide step-by-step troubleshooting guidance
                - Explain product features and functionality
                - Help with setup and installation instructions
                - Address common customer concerns and FAQs
                - Provide technical specifications when requested
                
                How to handle customer queries:
                1. Always search the knowledge base first for relevant information
                2. Use specific keywords from the customer's question
                3. If initial search doesn't yield results, try alternative search terms
                4. Combine multiple searches if needed for complex questions
                
                Response guidelines:
                - Be friendly, professional, and empathetic
                - Provide clear, step-by-step instructions when applicable
                - Include relevant details from the knowledge base
                - If information is not available, be honest and offer alternatives
                - Format responses for easy reading (use bullet points, numbered lists)
                - Always maintain a helpful, solution-oriented tone
                - Acknowledge customer frustration and show understanding
                
                When you cannot find specific information:
                - Clearly state what information you couldn't find
                - Suggest alternative solutions or workarounds if possible
                - Recommend escalation to human support for complex issues
                
                Remember: Your goal is to resolve customer issues quickly and accurately using our official documentation.
            """,
            tools=[knowledge_base_tool],
            model=self.model,
            model_settings=ModelSettings(temperature=self.temperature),
        )
    
    def _create_escalation_agent(self):
        return Agent(
            name="EscalationAgent",
            instructions="""
                You are the Escalation Specialist for customer support.
                Your role is to handle complex issues that require additional research or human intervention.

                When an escalation is required:
                - Use the `send_support_email` tool to notify the human support team.
                - Provide a clear summary of the issue, troubleshooting steps, and customer details in the email body.
            """,
            tools=[self.web_search_tool, send_support_email_tool],
            model=self.model,
            model_settings=ModelSettings(temperature=self.temperature),
        )

    def _create_support_router(self):
        return Agent(
            name="CustomerSupportRouter",
            instructions="""
                You are the Customer Support Router Agent.
                Your job is to analyze customer queries and route them to the appropriate specialist agent.

                Routing Guidelines:

                → Route to ProductKnowledgeAgent for:
                - Product feature questions and explanations
                - How-to questions and setup instructions
                - Troubleshooting common issues
                - Technical specifications inquiries
                - FAQ-type questions
                - Product comparison questions
                - General product usage questions
                - Error messages or issues that might be documented

                → Route to EscalationAgent for:
                - Complex technical issues not in documentation
                - Unusual error messages or behaviors
                - Questions about product roadmap or updates
                - Issues that might require web research
                - Problems that seem to need human intervention
                - Requests for features not currently available
                - Any situation where the support team should be notified by email
                (the EscalationAgent has access to `send_support_email` tool)

                Initial Response Protocol:
                1. Greet the customer warmly and professionally
                2. Acknowledge their query or concern
                3. Quickly assess the type of support needed
                4. Route to appropriate agent with context

                For ambiguous queries:
                - Ask clarifying questions to understand the specific issue
                - Gather relevant details (product version, error messages, steps taken)
                - Then route based on the clarified information

                Always ensure customers feel welcomed and that their concerns are being taken seriously.
            """,
            handoffs=[self.knowledge_agent, self.escalation_agent],
            model=self.model,
            model_settings=ModelSettings(temperature=self.temperature),
        )

    async def handle_support_query(self, query: str, customer_name: str = "Customer"):
        """Handle customer support query with personalized greeting"""
        contextual_query = f"Customer {customer_name} asks: {query}"
        
        result = await Runner.run(self.support_router, contextual_query, session=self.session)
        try:
            return result.final_output
        except AttributeError:
            return str(result)

    async def search_knowledge_base(self, query: str):
        """Direct knowledge base search for testing"""
        result = await Runner.run(self.knowledge_agent, f"Please search for information about: {query}", session=self.session)
        try:
            return result.final_output
        except AttributeError:
            return str(result)


# Streamlit UI Application
def main():
    st.set_page_config(
        page_title="Customer Support Assistant",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'support_system' not in st.session_state:
        st.session_state.support_system = CustomerSupportSystem()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'customer_name' not in st.session_state:
        st.session_state.customer_name = ""
    
    if 'uploaded_files_count' not in st.session_state:
        st.session_state.uploaded_files_count = 0
    
    if 'qdrant_client' not in st.session_state:
        # Initialize Qdrant client using your configuration
        try:
            if QDRANT_URL and QDRANT_API_KEY:
                st.session_state.qdrant_client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    timeout=30
                )
                # Test connection
                collections = st.session_state.qdrant_client.get_collections()
                st.sidebar.success(f"✅ Connected to Qdrant")
            else:
                st.session_state.qdrant_client = None
                st.sidebar.error("⚠️ Qdrant credentials not found in .env")
        except Exception as e:
            st.session_state.qdrant_client = None
            st.sidebar.error(f"⚠️ Could not connect to Qdrant: {str(e)}")

    # Sidebar
    with st.sidebar:
        st.title("🎧 Support Settings")
        
        # Customer Information
        st.subheader("Customer Information")
        customer_name = st.text_input(
            "Customer Name (Optional)", 
            value=st.session_state.customer_name,
            placeholder="Enter customer name..."
        )
        st.session_state.customer_name = customer_name

        # File Upload Section
        st.markdown("---")
        st.subheader("📄 Upload Documents to Knowledge Base")
        st.caption(f"Collection: `{QDRANT_COLLECTION_NAME}`")
        
        uploaded_files = st.file_uploader(
            "Upload Support Documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload product manuals, FAQs, or any support documentation. Files will be embedded using OpenAI and stored in Qdrant."
        )
        
        if uploaded_files and st.session_state.qdrant_client:
            if st.button("🚀 Process & Upload to Qdrant", type="primary"):
                with st.spinner("Processing documents and generating embeddings..."):
                    total_chunks = 0
                    success_count = 0
                    progress_bar = st.progress(0)
                    
                    for idx, file in enumerate(uploaded_files):
                        st.write(f"📄 Processing: **{file.name}**")
                        
                        # Process file asynchronously
                        chunks, message = asyncio.run(
                            process_uploaded_file(
                                file,
                                st.session_state.qdrant_client,
                                QDRANT_COLLECTION_NAME
                            )
                        )
                        
                        total_chunks += chunks
                        if chunks > 0:
                            success_count += 1
                            st.success(f"✅ {message} ({chunks} chunks)")
                        else:
                            st.error(f"❌ {message}")
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    if success_count > 0:
                        st.session_state.uploaded_files_count += success_count
                        st.balloons()
                        st.success(f"🎉 Successfully uploaded {success_count} files with {total_chunks} total chunks to Qdrant!")
                        st.info("💡 The knowledge base is now updated. Agents can search this content immediately!")
        
        elif uploaded_files and not st.session_state.qdrant_client:
            st.error("⚠️ Qdrant client not initialized. Check your .env file for QDRANT_URL and QDRANT_API_KEY")
        
        if st.session_state.uploaded_files_count > 0:
            st.info(f"📊 Files uploaded this session: **{st.session_state.uploaded_files_count}**")

        # Session Information
        st.markdown("---")
        st.subheader("Session Info")
        st.info(f"Session ID: `{st.session_state.support_system.session_id}`")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.support_system = CustomerSupportSystem()
            st.rerun()

        # Support Statistics
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Uploads", st.session_state.uploaded_files_count)
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        if st.button("🔍 Search Knowledge Base"):
            st.session_state.show_kb_search = True

    # Main Interface
    st.title("🎧 Customer Support Assistant")
    st.markdown("Welcome to our AI-powered customer support! Ask me anything about our products.")

    # Knowledge Base Search (if triggered)
    if hasattr(st.session_state, 'show_kb_search') and st.session_state.show_kb_search:
        with st.expander("🔍 Knowledge Base Search", expanded=True):
            kb_query = st.text_input("Search our knowledge base directly:")
            if st.button("Search") and kb_query:
                with st.spinner("Searching knowledge base..."):
                    try:
                        kb_result = asyncio.run(st.session_state.support_system.search_knowledge_base(kb_query))
                        st.success("Knowledge Base Results:")
                        st.write(kb_result)
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
            
            if st.button("Close Search"):
                st.session_state.show_kb_search = False
                st.rerun()

    # Chat Interface
    st.subheader("💬 Support Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"⏰ {message['timestamp']}")

    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"⏰ {timestamp}")

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    response = asyncio.run(
                        st.session_state.support_system.handle_support_query(
                            prompt, 
                            customer_name or "Customer"
                        )
                    )
                    st.markdown(response)
                    
                    # Add assistant response to messages
                    response_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": response_timestamp
                    })
                    st.caption(f"⏰ {response_timestamp}")
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try again or contact human support."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                    })

    # Footer
    st.markdown("---")
    st.markdown("🤖 Powered by AI | 📧 Need human support? Contact support@yourcompany.com")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Streamlit not running, falling back to command line mode. Error: {e}")
        system = CustomerSupportSystem()
        print("Customer Support System - Command Line Mode")
        print("Type 'exit' to quit, 'kb:query' to search knowledge base directly")

        async def interactive():
            while True:
                q = input("\n🎧 Customer: ").strip()
                if q.lower() in ("exit", "quit"):
                    break
                
                if q.startswith("kb:"):
                    kb_query = q[3:].strip()
                    print("🔍 Searching knowledge base...")
                    resp = await system.search_knowledge_base(kb_query)
                else:
                    print("🤖 Processing support request...")
                    resp = await system.handle_support_query(q, "Test Customer")
                
                print(f"\n🎧 Support Agent: {resp}")

        asyncio.run(interactive())