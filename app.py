import asyncio
import os
import time
import streamlit as st
from agents import Agent, Runner, SQLiteSession, ModelSettings
from agents.tool import WebSearchTool
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from agents.tool import Tool
from agno.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.reader.pdf_reader import PDFReader
from tools import qdrant_search_tool_on_invoke # Assuming 'tools.py' contains this function
from agents.tool import FunctionTool, ToolContext 
import datetime

# Setup components
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Enhanced Knowledge Base search tool for customer support
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
        temperature=0.1,  # Slightly higher for more natural responses
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
                
                Your responsibilities:
                - Handle queries that couldn't be resolved with the knowledge base
                - Search for additional information online when internal docs are insufficient
                - Provide general troubleshooting advice for uncommon issues
                - Gather information for human support escalation
                - Research industry-standard solutions for technical problems
                
                When to use web search:
                - Customer reports an error not found in knowledge base
                - Need to check for known issues or recent updates
                - Looking for general troubleshooting approaches
                - Verifying compatibility information
                - Finding workarounds for known limitations
                
                Escalation criteria:
                - Account-specific issues (billing, subscriptions, personal data)
                - Product defects or warranty claims
                - Complex technical issues requiring specialized expertise
                - Requests for customization or special features
                - Complaints requiring management attention
                
                Response format for escalations:
                - Summarize the customer's issue clearly
                - List troubleshooting steps already attempted
                - Provide relevant information gathered
                - Recommend next steps for human support team
                - Set appropriate customer expectations
                
                Always maintain professionalism and ensure customers feel heard and supported.
            """,
            tools=[self.web_search_tool],
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
        # Add customer context to the query
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

        # Session Information
        st.subheader("Session Info")
        st.info(f"Session ID: {st.session_state.support_system.session_id}")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.support_system = CustomerSupportSystem()
            st.rerun()

        # Support Statistics
        st.subheader("📊 Session Stats")
        st.metric("Messages", len(st.session_state.messages))
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        if st.button("Search Knowledge Base"):
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
    # Check if running in Streamlit
    try:
        # This will work if we're in Streamlit
        main()
    except Exception as e:
        # Fallback for command line testing
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
                    # Direct knowledge base search
                    kb_query = q[3:].strip()
                    print("🔍 Searching knowledge base...")
                    resp = await system.search_knowledge_base(kb_query)
                else:
                    # Full support query
                    print("🤖 Processing support request...")
                    resp = await system.handle_support_query(q, "Test Customer")
                
                print(f"\n🎧 Support Agent: {resp}")

        asyncio.run(interactive())