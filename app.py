import asyncio
import os
import time
import datetime
import re
import streamlit as st
from dotenv import load_dotenv
from agents import Agent, Runner, SQLiteSession, ModelSettings
from agents.tool import WebSearchTool, FunctionTool
from tools import (
    qdrant_search_tool_on_invoke,
    upload_pdf_to_qdrant,
    ensure_qdrant_collection,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    QDRANT_API_KEY,
)
import yagmail
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# ---------------------------------------------------
# Email Utility
# ---------------------------------------------------
def send_support_email(subject: str, body: str, sender_email=None, sender_password=None) -> str:
    """Send escalation emails to support team."""
    try:
        sender_email = sender_email or os.environ.get("SENDER_EMAIL")
        sender_password = sender_password or os.environ.get("SENDER_PASSWORD")

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, sender_email):
            return "Invalid email format."

        yag = yagmail.SMTP(user=sender_email, password=sender_password)
        yag.send(to=sender_email, subject=subject, contents=body)
        return "✅ Email sent successfully!"
    except Exception as e:
        return f"❌ Error sending email: {e}"


# ---------------------------------------------------
# Customer Support Multi-Agent System
# ---------------------------------------------------
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

        self.web_search_tool = WebSearchTool()
        self.session = SQLiteSession(self.session_id, self.db_file)
        self.knowledge_agent = self._create_knowledge_agent()

    def _create_knowledge_agent(self):
        return Agent(
            name="ProductKnowledgeAgent",
            instructions="""
            You are a customer support agent for RUNO, a company that provides SIM-based call management systems.
            Your role is to help RUNO’s clients by answering questions about our services, features, technical setup, troubleshooting, and best practices.

            Important: The Qdrant tool contains up-to-date, frequently asked questions and answers from our client support logs.
            Always search this knowledge base first to find relevant, accurate information.

            Guidelines:
            - Be friendly, professional, and empathetic in all your responses.
            - Format answers clearly using bullet points or numbered lists for easy reading.
            - If you can't find an exact answer, offer alternative suggestions based on your knowledge.
            - Ensure customers understand you are here to help them with any issue related to RUNO’s SIM call management systems.

            Your primary goal is to resolve client queries quickly and accurately using the Qdrant knowledge base of frequent customer questions.
        """,
            tools=[FunctionTool(
                name="ProductKnowledgeSearch",
                description="Searches the RUNO company’s product knowledge base.",
                params_json_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 6}
                    },
                    "required": ["query"]
                },
                on_invoke_tool=qdrant_search_tool_on_invoke
            )],
            model=self.model,
            model_settings=ModelSettings(temperature=self.temperature),
        )

    async def handle_support_query(self, query: str, customer_name: str = "Customer"):
        contextual_query = f"Customer {customer_name} asks: {query}"
        result = await Runner.run(self.knowledge_agent, contextual_query, session=self.session)
        return getattr(result, "final_output", str(result))

    async def search_knowledge_base(self, query: str):
        result = await Runner.run(
            self.knowledge_agent,
            f"Search for: {query}",
            session=self.session
        )
        return getattr(result, "final_output", str(result))


# ---------------------------------------------------
# Streamlit UI Application
# ---------------------------------------------------
# Streamlit UI Application
def main():
    st.set_page_config(
        page_title="Customer Support Assistant",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---- Sidebar Credentials ----
    st.sidebar.title("🎧 Support Settings")
    st.sidebar.subheader("🔑 API & Email Credentials")
    # Example: set defaults for local/dev, remove/change for prod
    # OpenAI API Key
    DEV_DEFAULT_OPENAI_API_KEY = "sk-proj-..............."

    # Email Configuration
    DEV_DEFAULT_SENDER_EMAIL = "name@gmail.com"
    DEV_DEFAULT_SENDER_PASSWORD = "password123"

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=DEV_DEFAULT_OPENAI_API_KEY)
    sender_email = st.sidebar.text_input("Sender Email", value=DEV_DEFAULT_SENDER_EMAIL)
    sender_password = st.sidebar.text_input("Email App Password", type="password", value=DEV_DEFAULT_SENDER_PASSWORD)

    # openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    # sender_email = st.sidebar.text_input("Sender Email")
    # sender_password = st.sidebar.text_input("Email App Password", type="password")
    st.session_state.openai_api_key = openai_api_key
    st.session_state.sender_email = sender_email
    st.session_state.sender_password = sender_password
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["SENDER_EMAIL"] = sender_email
    os.environ["SENDER_PASSWORD"] = sender_password

    # ---- Customer Info ----
    st.sidebar.subheader("👤 Customer Information")
    customer_name = st.sidebar.text_input("Customer Name (Optional)", value=st.session_state.get('customer_name', ""))
    st.session_state.customer_name = customer_name

    # ---- Escalate Case Section ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Escalate Support Case")
    if "escalate_active" not in st.session_state:
        st.session_state.escalate_active = False

    if st.sidebar.button("Escalate a Case"):
        st.session_state.escalate_active = True

    if st.session_state.escalate_active:
        subject = st.sidebar.text_input("Escalation Subject", value="")
        body = st.sidebar.text_area(
            "Escalation Email Body", 
            value="Describe the issue and prior troubleshooting steps here."
        )
        escalate_clicked = st.sidebar.button("Send Escalation Email", key="send_escalation")
        if escalate_clicked:
            result = send_support_email(
                subject=subject,
                body=body,
                sender_email=st.session_state.sender_email,
                sender_password=st.session_state.sender_password
            )
            st.sidebar.success(f"Escalation Result: {result}")
            st.session_state.escalate_active = False

    # ---- Knowledge Base/Chat Setup ----
    if 'support_system' not in st.session_state:
        st.session_state.support_system = CustomerSupportSystem()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploaded_files_count' not in st.session_state:
        st.session_state.uploaded_files_count = 0
    if 'qdrant_client' not in st.session_state:
        try:
            if QDRANT_URL and QDRANT_API_KEY:
                st.session_state.qdrant_client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    timeout=30
                )
                st.sidebar.success(f"✅ Connected to Qdrant")
            else:
                st.session_state.qdrant_client = None
                st.sidebar.error("⚠️ Qdrant credentials not found in .env")
        except Exception as e:
            st.session_state.qdrant_client = None
            st.sidebar.error(f"⚠️ Could not connect to Qdrant: {str(e)}")

    # ---- File Upload Section ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Upload Documents to Knowledge Base")
    st.sidebar.caption(f"Collection: `{QDRANT_COLLECTION_NAME}`")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Support Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload product manuals, FAQs, or any support documentation."
    )

    # File upload processing
    if uploaded_files and st.session_state.qdrant_client:
        if st.sidebar.button("🚀 Process & Upload to Qdrant", type="primary"):
            with st.spinner("Processing documents and generating embeddings..."):
                total_chunks = 0
                success_count = 0
                progress_bar = st.progress(0)

                for idx, file in enumerate(uploaded_files):
                    st.write(f"📄 Processing: **{file.name}**")
                    chunks, message = asyncio.run(
                        upload_pdf_to_qdrant(file,
                         st.session_state.qdrant_client)
                    )
                    total_chunks += chunks
                    if chunks > 0:
                        success_count += 1
                        st.success(f"✅ {message} ({chunks} chunks)")
                    else:
                        st.error(f"❌ {message}")
                    progress_bar.progress((idx + 1) / len(uploaded_files))

                if success_count > 0:
                    st.session_state.uploaded_files_count += success_count
                    st.balloons()
                    st.success(f"🎉 Successfully uploaded {success_count} files with {total_chunks} total chunks to Qdrant!")
                    st.info("💡 The knowledge base is now updated. Agents can search this content immediately!")

    if uploaded_files and not st.session_state.qdrant_client:
        st.error("⚠️ Qdrant client not initialized. Check your .env file for QDRANT_URL and QDRANT_API_KEY")

    if st.session_state.uploaded_files_count > 0:
        st.info(f"📊 Files uploaded this session: **{st.session_state.uploaded_files_count}**")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Info")
    st.sidebar.info(f"Session ID: `{st.session_state.support_system.session_id}`")

    # ---- Main Chat UI ----
    st.title("🎧 Customer Support Assistant")
    st.markdown("Welcome to our AI-powered customer support! Ask me anything about our products.")

    st.subheader("💬 Support Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"⏰ {message['timestamp']}")

    prompt = st.chat_input("How can I help you today?")
    if prompt:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": timestamp
        })
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"⏰ {timestamp}")

        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    response = asyncio.run(
                        st.session_state.support_system.handle_support_query(
                            prompt, customer_name or "Customer"
                        )
                    )
                    st.markdown(response)
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

    st.markdown("---")
    st.markdown("🤖 Powered by AI | 📧 Need human support? Use 'Escalate Case' above!")

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