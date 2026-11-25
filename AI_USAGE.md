# AI Usage Disclosure — Task 8 (LangGraph)

## Tool Used
- **Antigravity** (agentic coding assistant)

## How AI Was Used
1. **Environment & Dependency Setup:**
   - Verified Python 3.13 environment, installed modern LangGraph (`1.2.24`) and LangChain (`1.5.2`) ecosystem packages, and pinned all exact package versions in `task8-langgraph/requirements.txt`.
   - Configured `.env` (with Groq API key and `openai/gpt-oss-120b` model) and `.gitignore` to keep secrets and sandbox/checkpoint artifacts untracked.
2. **Code Scaffolding & Implementation (`task8-langgraph/`):**
   - `tools.py`: Implemented `reset_sandbox()`, `query_database` (strictly read-only SQLite URI + SELECT-only validation), `search_policies` (chunking all 14 `novatech_docs/*.md` files with `RecursiveCharacterTextSplitter`, embedding via `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")`, and retrieving `k=3` chunks from `InMemoryVectorStore`), `calculate_days`, `cancel_order` (pre-checking `status == "processing"` before calling LangGraph `interrupt()` and updating `novatech_sandbox.db` only on approval), and `--flaky` transient failure simulation.
   - `graph.py`: Implemented a manual `StateGraph` (`AgentState`, `agent_node`, `ToolNode(TOOLS)`, `route_after_agent`, `MemorySaver` checkpointer, `init_chat_model` on Groq `openai/gpt-oss-120b` with `temperature=0`, and Mermaid diagram export) without using prebuilt agents (`create_agent`, `create_react_agent`, or `AgentExecutor`). Refined `SYSTEM_PROMPT` so Returns Policy v2.0 (`returns_policy.md`) supersedes `returns_policy_v1.3.md`.
   - `run.py`: Implemented CLI modes `--mode task6`, `--mode conversations` (parsing `task8_conversations.md` and saving `c1_trace.txt`), `--mode interactive`, and `--mode c7` (`SqliteSaver` cross-process pause/resume + checkpoint history), plus `--flaky` and `--scripted-manager`.
3. **Verification & Documentation:**
   - Executed automated tests covering Mermaid export, tool unit checks, all 10 Task 6 queries, `--flaky` retry behavior, interactive-stdin conversations `C1`–`C6`, `C7` cross-process `SqliteSaver` persistence, and SHA-256 verification confirming `novatech_store.db` remained completely untouched.
   - Populated `task8-langgraph/graph-notes.md` with the `task6-agent/agent.py` → LangGraph mapping table, Mermaid graph diagram, Task 6 (`task6-agent/agent-notes.md`) vs Task 8 comparison table, `C1`–`C7` results table, and the full interactive `C1` trace.

## Human-Only Sections (No AI Used)
- All **5 theory questions** in `task8-langgraph/graph-notes.md` were written independently by the student without AI assistance and pasted verbatim into the file.
