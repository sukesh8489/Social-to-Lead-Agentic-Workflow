import json
import os
from typing import Annotated, TypedDict, Literal

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import dotenv

dotenv.load_dotenv()

# ==========================================
# 1. RAG Setup
# ==========================================
def setup_rag():
    try:
        with open("knowledge.json", "r") as f:
            data = json.load(f)
            
        docs = []
        # Parse pricing
        metrics = data["product_info"]
        
        # Convert JSON structure to documents
        text_content = ""
        
        # Basic Plan
        b_plan = metrics["AutoStream_Pricing_Features"]["Basic_Plan"]
        text_content += f"AutoStream Basic Plan:\n- Price: {b_plan['price']}\n- Video Limit: {b_plan['videos_allowed']}\n- Resolution: {b_plan['resolution']}\n\n"
        
        # Pro Plan
        p_plan = metrics["AutoStream_Pricing_Features"]["Pro_Plan"]
        text_content += f"AutoStream Pro Plan:\n- Price: {p_plan['price']}\n- Video Limit: {p_plan['videos_allowed']}\n- Resolution: {p_plan['resolution']}\n- Extra Features: {', '.join(p_plan['features'])}\n\n"
        
        docs.append(Document(page_content=text_content, metadata={"source": "Pricing"}))
        
        # Policies
        policies = metrics["Company_Policies"]
        policy_text = f"AutoStream Company Policies:\n- Refunds: {policies['refunds']}\n- Support: {policies['support']}\n"
        docs.append(Document(page_content=policy_text, metadata={"source": "Policies"}))

        # Check API key first, if none return None (prevents crashing at startup if env missing)
        if not os.environ.get("GOOGLE_API_KEY"):
             return None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        print(f"Warning: RAG setup failed (Error: {e})")
        return None

retriever = setup_rag()

@tool
def query_knowledge_base(query: str) -> str:
    """Use this tool to search the knowledge base for information about AutoStream pricing, features, and company policies."""
    if not retriever:
        return "The knowledge base is currently unavailable."
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# ==========================================
# 2. Tool Execution - Lead Capture
# ==========================================
@tool
def mock_lead_capture(name: str, email: str) -> str:
    """
    Call this tool ONLY when you have identified the user as a high-intent lead 
    and you have collected their name and email.
    """
    print(f"\n[SYSTEM] Lead captured successfully: {name}, {email}")
    return "Lead information successfully submitted to the CRM."

# ==========================================
# 3. Agent Setup (LangGraph)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tools = [query_knowledge_base, mock_lead_capture]
tool_node = ToolNode(tools)

# Optionally use GEMINI_API_KEY if GOOGLE_API_KEY is not set but needed by Gemini
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are Inflx, a Conversational AI Agent for AutoStream, a SaaS product providing automated video editing tools for content creators.
Your capabilities:
1. Intent Identification: Actively identify if the user is making a:
   - Casual greeting (respond politely, proactively ask for their Name and Email if you don't have it yet, and ask how you can help)
   - Product or pricing inquiry (use the 'query_knowledge_base' tool, and ask for their Name and Email if not provided yet)
   - High-intent lead (e.g., ready to sign up, wants to try a plan).
2. Lead Qualification: You MUST ask for their Name and Email as early as possible during the conversation.
   Do not call the lead capture tool until you have gathered BOTH details.
3. RAG-Powered Knowledge Retrieval: Always use the 'query_knowledge_base' tool to answer questions about AutoStream's plans, prices, features, and policies. Do not hallucinate capabilities.

Be concise, friendly, and helpful. Always wait for the user's input to clarify details.
"""

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, stop
    return "__end__"

def call_model(state: AgentState):
    messages = state["messages"]
    # Prepend the system prompt if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()
