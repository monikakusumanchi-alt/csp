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
def main():
    st.set_page_config(
        page_title="Customer Support Assistant",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar Settings
    st.sidebar.title("🎧 Support Settings")
    st.sidebar.subheader("🔑 API & Email Credentials")

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    sender_email = st.sidebar.text_input("Sender Email")
    sender_password = st.sidebar.text_input("Email App Password", type="password")

    # Store credentials
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["SENDER_EMAIL"] = sender_email
    os.environ["SENDER_PASSWORD"] = sender_password
    st.session_state.openai_api_key = openai_api_key

    # Customer Info
    st.sidebar.subheader("👤 Customer Information")
    customer_name = st.sidebar.text_input("Customer Name", value=st.session_state.get('customer_name', ""))
    st.session_state.customer_name = customer_name

    # Escalate Case
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Escalate Support Case")
    if st.sidebar.button("Escalate a Case"):
        subject = st.sidebar.text_input("Subject", value="")
        body = st.sidebar.text_area("Email Body", value="Describe the issue and previous steps.")
        if st.sidebar.button("Send Escalation Email"):
            result = send_support_email(
                subject=subject,
                body=body,
                sender_email=sender_email,
                sender_password=sender_password
            )
            st.sidebar.success(result)

    # Initialize Qdrant client
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Knowledge Base Setup")

    if 'qdrant_client' not in st.session_state:
        try:
            st.session_state.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            ensure_qdrant_collection()
            st.sidebar.success(f"✅ Connected to Qdrant: `{QDRANT_COLLECTION_NAME}`")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to connect Qdrant: {e}")
            st.session_state.qdrant_client = None

    # PDF Upload Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Upload Documents to Knowledge Base")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Support PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.qdrant_client:
        if st.sidebar.button("🚀 Upload PDFs"):
            with st.spinner("Processing and uploading PDFs to Qdrant..."):
                for file in uploaded_files:
                    file_path = f"/tmp/{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    asyncio.run(upload_pdf_to_qdrant(file_path))
                    st.success(f"✅ Uploaded {file.name}")
                st.balloons()

    # Initialize support system
    if 'support_system' not in st.session_state:
        st.session_state.support_system = CustomerSupportSystem()

    # Main Chat Interface
    st.title("🎧 Customer Support Assistant")
    st.markdown("Welcome to RUNO Support! Ask any question about our SIM-based CRM system.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "timestamp" in msg:
                st.caption(f"⏰ {msg['timestamp']}")

    # Chat Input
    prompt = st.chat_input("How can I help you today?")
    if prompt:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"⏰ {timestamp}")

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    response = asyncio.run(
                        st.session_state.support_system.handle_support_query(prompt, customer_name or "Customer")
                    )
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                    })
                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)

    st.markdown("---")
    st.caption("🤖 Powered by AI | 💬 Escalate via sidebar if needed.")


if __name__ == "__main__":
    main()