
# LangGraph Self-RAG

A **Self-RAG style retrieval system** built with **LangGraph**, **ChromaDB**, **BM25**, **Groq LLM**, and tool calling.

The system retrieves relevant documents, checks their relevance, rewrites the query when retrieval is insufficient, and allows the LLM to use external tools when needed.

---

## Architecture

```text
User Question
      │
      ▼
┌──────────────┐
│  Retrieval   │
│ Dense + BM25 │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Relevance Check │
└───────┬─────────┘
        │
   ┌────┴─────┐
   │          │
 YES          NO
   │          │
   ▼          ▼
 LLM      Rewrite Query
 Tools         │
   ▲           │
   └─────◄─────┘
   │
   ▼
Tool Decision
   │
 ┌─┴─────────────┐
 │               │
Tools          Answer
 │
 ▼
Approval
 │
 ▼
LLM
```

---

## 1. Document Loading

The project loads a PDF using `PyPDFLoader`.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "Why_Language_Models_Hallucinate_Explainer.pdf"
)

pages = loader.load()
```

The loaded document is then split into smaller chunks.

---

## 2. Text Chunking

The project uses `RecursiveCharacterTextSplitter`.

```python
RecursiveCharacterTextSplitter(
    chunk_size=1400,
    chunk_overlap=180
)
```

Each chunk receives:

* Document content
* Metadata
* MD5-based ID

```python
ids = [
    hashlib.md5(
        chunk.encode("utf-8")
    ).hexdigest()
    for chunk in chunks
]
```

---

## 3. Vector Database

The project uses **ChromaDB** with the `all-MiniLM-L6-v2` embedding model.

```python
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./Hybrid_RAG"
)

collection = client.get_or_create_collection(
    name="Hybrid_RAG",
    embedding_function=embedding_function
)
```

Documents are stored persistently in:

```text
./Hybrid_RAG
```

---

## 4. Hybrid Retrieval

The retrieval system combines:

### Dense Retrieval

ChromaDB performs semantic retrieval:

```python
result = collection.query(
    query_texts=[query_re],
    n_results=5
)
```

A distance threshold is applied:

```python
threshold = 1.5
```

### Sparse Retrieval

BM25 performs keyword-based retrieval:

```python
tokens = BM25Okapi(token_corpus)
scores = tokens.get_scores(query_re.split())
```

### Reciprocal Rank Fusion

Dense and sparse results are combined using RRF.

```text
RRF score = Σ 1 / (rank + 60)
```

The highest-ranked documents are selected as the final context.

---

## 5. Query Rewriting

If the retrieved context is not relevant, the system asks the LLM to rewrite the user's query.

```python
Rewrite Query
      │
      ▼
Better Search Query
      │
      ▼
Hybrid Retrieval
```

The system allows a maximum of **2 retries** before moving forward.

---

## 6. Relevance Checking

The retrieved documents are evaluated by the LLM.

The grader returns only:

```text
YES
```

or

```text
NO
```

If the answer is `YES`, the system continues.

If the answer is `NO`, the query is rewritten and retrieval happens again.

---

## 7. LLM + Tools

The Groq model is configured as:

```python
ChatGroq(
    model="openai/gpt-oss-120b"
)
```

Available tools:

### Calculator

Uses `numexpr` for arithmetic operations.

```python
calculator(exec: str)
```

### Web Search

Uses DuckDuckGo.

```python
Web_search(query: str)
```

The LLM decides whether a tool is required.

---

## 8. Human Approval / Interrupt

Before tool execution, the graph can interrupt execution for approval.

```python
decision = interrupt({
    "type": "approval",
    "reason": "Model is about to answer user's question"
})
```

The graph is compiled with:

```python
interrupt_before=["tools"]
```

Execution can then be resumed using:

```python
from langgraph.types import Command

response = builder.invoke(
    Command(resume=True),
    config=config
)
```

---

## 9. LangGraph State

The graph maintains:

```python
class State(TypedDict):
    messages: list
    retrieved_context: list
    relevance: str
    search_query: str
    retries: int
```

### State responsibilities

| Field                 | Purpose                      |
| --------------------- | ---------------------------- |
| `messages`          | User and model messages      |
| `retrieved_context` | Retrieved documents          |
| `relevance`         | Retrieval quality decision   |
| `search_query`      | Rewritten retrieval query    |
| `retries`           | Number of retrieval attempts |

---

## 10. Graph Flow

The LangGraph nodes are:

```text
START
  ↓
retrival
  ↓
relevance_check
  ↓
 ┌─────────────────┐
 │                 │
YES                NO
 │                 │
 ▼                 ▼
LLM_tools      rewrite_query
 │                 │
 │                 ▼
 │              retrival
 │
 ▼
Tool Condition
 │
 ├── tools
 │     ↓
 │   approval
 │     ↓
 │   LLM_tools
 │
 └── END
```

---

## 11. Persistent Memory

The graph uses SQLite checkpointing:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory_context = SqliteSaver.from_conn_string(
    "langgraph_memory.db"
)
```

A thread ID is used to identify the conversation:

```python
config = {
    "configurable": {
        "thread_id": "id_3"
    },
    "recursion_limit": 12
}
```

---

## 12. Technologies Used

* Python
* LangChain
* LangGraph
* ChromaDB
* Sentence Transformers
* BM25
* Groq
* DuckDuckGo
* NumExpr
* SQLite

---

## 13. Main Concepts Demonstrated

This project demonstrates:

* Document loading
* Text chunking
* Embeddings
* Vector databases
* Dense retrieval
* Sparse retrieval
* Hybrid RAG
* Reciprocal Rank Fusion
* Query rewriting
* Retrieval relevance grading
* Agentic tool calling
* LangGraph state management
* Conditional routing
* Graph loops
* Interrupts
* Human approval
* SQLite checkpointing
* Thread-based execution
* Recursion limits

---

## 14. Project Structure

```text
LangGraph_Self_RAG/
│
├── LangGraph_Self_RAG.ipynb
├── Why_Language_Models_Hallucinate_Explainer.pdf
├── Hybrid_RAG/
│   └── ChromaDB persistent data
│
└── langgraph_memory.db
```

---

## 15. Example

The notebook tests the system with:

```python
question = (
    "What are the main factors discussed in the research "
    "that affect economic growth?"
)
```

The graph then performs:

```text
Question
   ↓
Hybrid Retrieval
   ↓
Relevance Check
   ↓
Query Rewrite if needed
   ↓
LLM
   ↓
Tool decision
   ↓
Approval
   ↓
Final response
```

---

## Key Takeaway

This project extends traditional RAG into an **iterative LangGraph workflow**.

Instead of:

```text
Question → Retrieve → Answer
```

the system implements:

```text
Question
   ↓
Retrieve
   ↓
Check relevance
   ↓
Rewrite if needed
   ↓
Retrieve again
   ↓
Generate
   ↓
Use tools when required
   ↓
Human approval
   ↓
Final answer
```

The important idea is that the system can **evaluate its retrieval and take another action instead of blindly generating an answer from the first retrieved documents**.
