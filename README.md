<h1 align="center">Social-to-Lead Agentic Workflow 🤖💼</h1>

<p align="center">
  <em>An autonomous Conversational AI Agent built for SaaS platforms to handle support, fetch knowledge base data via RAG, and instantly qualify high-intent leads.</em>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-1C1C1E?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img alt="Gemini API" src="https://img.shields.io/badge/Gemini_API-8E75B2?style=for-the-badge&logo=google&logoColor=white" />
</p>

---

## 📖 Overview

This repository contains an implementation of a Conversational AI Agent designed for **AutoStream**, a fictional automated video editing SaaS product. The workflow dynamically adapts to user intent, ensuring smooth handoffs between casual chat, product FAQs, and lead capture.

### Key Features
* 🧠 **Dynamic Intent Detection:** Identifies whether a user is making a casual greeting, asking about the product, or showing high purchase intent.
* 📚 **RAG-Powered Knowledge Retrieval:** Utilizes an in-memory FAISS vector store to fetch and answer questions regarding AutoStream's pricing, features, and company policies directly from `knowledge.json`.
* 🎯 **Autonomous Lead Capture:** Proactively collects the user's **Name and Email**. Once successfully gathered, it seamlessly triggers an automated mock Lead Capture tool to submit the data to a CRM.

## 🚀 How to Run Locally

**Prerequisites:**
- Python 3.9+
- A Google Gemini API Key

**Setup Steps:**

1. **Clone the repository** and navigate into the project directory:
   ```bash
   git clone <your-repo-url>
   cd PROJECT
   ```

2. **Create a Virtual Environment** (Highly Recommended):
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**:
   * Create a `.env` file in the root directory.
   * Add your Gemini API Key representing your Google AI Studio configuration:
     ```env
     GOOGLE_API_KEY=your_gemini_api_key_here
     ```

5. **Run the CLI Application**:
   ```bash
   python app.py
   ```
   *Type your messages in the terminal to interact with Inflx, the AI Agent!*

---

## 🏗️ Architecture Explanation

This application leans on **LangGraph** because it natively supports cyclic, stateful multi-agent workflows, which are essential for maintaining tool calling and conversation history during RAG processes. LangGraph treats the conversation as a graph where nodes execute logic (calling an LLM or running a tool) and edges manage the control flow.

* **State Management:** State is managed via a LangGraph `TypedDict` (`AgentState`). Specifically, the `messages` array records the entire conversation history context. In `app.py`, the message list gets continuously populated with Human, AI, and Tool messages and is fed back into the graph, ensuring state is retained accurately across conversation turns.
* **RAG Architecture:** The text data regarding AutoStream is ingested into an in-memory `FAISS` vector store utilizing `GoogleGenerativeAIEmbeddings`. This store acts as a retriever and is securely bound to the LLM via the `query_knowledge_base` tool.
* **Tool Execution:** The lead capture function is wrapped as a strict LangChain `@tool`. Through system prompts, the LLM is instructed only to trigger this tool when high-intent parameters (Name and Email) are fully collected.

---

## 📱 WhatsApp Integration Path

**How to deploy this agent to WhatsApp utilizing Webhooks:**

To take this from a CLI demo to a production WhatsApp Agent, we would typically use the WhatsApp Business API (via Meta for Developers or Twilio).

1. **Expose a Webhook Endpoint:** Wrap the `agent.py` executable behind a web framework like **FastAPI** or **Flask** to expose a `POST /webhook` REST endpoint.
2. **Meta Webhook Configuration:** Configure the WhatsApp Business app environment to point its event webhook URL to our deployed FastAPI endpoint.
3. **Receive Cloud Messages:** When a user messages the WhatsApp number, Meta dispatches a JSON payload to the webhook containing the user's phone number and the text content.
4. **State Management per User:** Instead of a single local array in `app.py`, the incoming user's phone number would serve as a distinct `thread_id`. We would use LangGraph's standard `MemorySaver` (backed by Redis, Postgres, or SQLite) to fetch and resume the conversation state using `thread_id`, ensuring the agent remembers the chat history for that specific mobile user.
5. **Graph Invocation:** We invoke the graph using the retrieved context: `graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": phone_number}})`.
6. **Delivery Response:** Once the graph yields a response item, the backend issues an asynchronous HTTP POST request back to the WhatsApp API sending the AI's generated response directly to the user's phone screen.
