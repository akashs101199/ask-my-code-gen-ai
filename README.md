# 💬 Ask My Code - GenAI

> An intelligent code assistant powered by open-source LLMs and RAG architecture. Understand, explore, and interact with your codebase through natural language conversations.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Mistral](https://img.shields.io/badge/Mistral-7B-orange.svg)](https://mistral.ai/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green.svg)](https://www.langchain.com/)

## 🎯 Project Overview

**Learning Goals:**
- Master RAG (Retrieval Augmented Generation) architecture for code understanding
- Build semantic code search using vector embeddings
- Create conversational interfaces over technical documentation
- Implement chunking strategies for large codebases

**What It Does:**  
A conversational AI assistant that lets you explore and understand codebases through natural language queries. No more grep searches or manual file navigation—just ask questions about your code.

## ✨ Key Features

### 🔍 **Semantic Code Search**
- **Natural language queries** over your entire codebase
- **Vector embeddings** for intelligent code similarity matching
- Find functions, classes, or patterns without knowing exact file locations
- Example: "Show me all database connection logic" → Returns relevant code snippets

### 💬 **Conversational Code Exploration**
- Ask questions about code structure, logic, and dependencies
- Get explanations in plain English, not just code snippets
- Follow-up questions with conversation memory
- Example conversation:
```
  You: "What does the user authentication function do?"
  AI: "The authenticate_user() function in auth.py validates credentials..."
  You: "Where is it called?"
  AI: "It's called in 3 places: login endpoint, API middleware, and admin panel..."
```

### 📚 **Documentation Generation**
- Auto-generate explanations for complex code blocks
- Create README sections from existing code
- Summarize file purposes and module relationships

### 🧠 **Code Understanding**
- Explain what code does in business terms
- Identify code patterns and anti-patterns
- Suggest refactoring opportunities
- Trace data flow through functions

### 🔐 **Privacy-First Architecture**
- Runs completely locally with Ollama
- Your code never leaves your machine
- Zero API costs, zero data leakage
- Open-source LLMs (Mistral/CodeLlama)

## 🏗️ Architecture
```
┌─────────────────────┐
│   User Query        │
│ "Show me auth code" │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Query Understanding│  ◄─── LLM (Mistral/CodeLlama)
│  & Intent Detection │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Vector Search      │  ◄─── Embeddings (sentence-transformers)
│  Top-K Retrieval    │       ChromaDB/FAISS
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Context Assembly   │  ◄─── Relevant code chunks
│  + Metadata         │       (file, line, function)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  LLM Response       │  ◄─── RAG Chain (LangChain)
│  Generation         │       Context + Query → Answer
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Formatted Answer   │
│  + Code Snippets    │
└─────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Mistral 7B / CodeLlama 7B (Ollama) | Code understanding & generation |
| **Framework** | LangChain | RAG orchestration, chains |
| **Vector DB** | ChromaDB / FAISS | Code embedding storage |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Semantic code search |
| **Code Parser** | tree-sitter / AST | Intelligent code chunking |
| **Frontend** | Streamlit / Gradio | Interactive chat interface |
| **Backend** | FastAPI | API endpoints (optional) |

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.9+
python --version

# Ollama for local LLM inference
# Install from: https://ollama.ai
ollama --version
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/akashs101199/ask-my-code-gen-ai.git
cd ask-my-code-gen-ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Pull LLM models**
```bash
# Primary model for code understanding
ollama pull mistral:7b

# Alternative: Specialized code model
ollama pull codellama:7b
```

5. **Index your codebase** (first-time setup)
```bash
# Point to the codebase you want to explore
python scripts/index_codebase.py --path /path/to/your/project

# This creates vector embeddings of your code
# Only needs to run once (or when code changes significantly)
```

### Running the Application

**Option 1: Streamlit UI (Recommended)**
```bash
streamlit run app.py
```

**Option 2: Command Line**
```bash
python cli.py --query "Show me all API endpoints"
```

**Option 3: Python API**
```python
from ask_my_code import CodeAssistant

assistant = CodeAssistant(codebase_path="./my_project")
response = assistant.ask("What does the main.py file do?")
print(response)
```

## 💬 Usage Examples

### Example 1: Find Functionality
