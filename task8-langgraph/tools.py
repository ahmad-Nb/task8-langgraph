"""Tools and sandbox database utilities for the Task 8 NovaTech LangGraph agent."""

from __future__ import annotations

import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import interrupt

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
STORE_DB = REPO_ROOT / "novatech_store.db"
SANDBOX_DB = REPO_ROOT / "novatech_sandbox.db"
DOCS_DIR = REPO_ROOT / "novatech_docs"

_VECTOR_STORE: InMemoryVectorStore | None = None
_FLAKY_MODE: bool = False
_FLAKY_FAILED_ONCE: set[str] = set()

_FORBIDDEN_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|TRIGGER|TRANSACTION|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)


def set_flaky_mode(enabled: bool) -> None:
    """Enable or disable simulated transient tool failures."""
    global _FLAKY_MODE
    _FLAKY_MODE = enabled
    _FLAKY_FAILED_ONCE.clear()


def reset_flaky_state() -> None:
    """Reset per-run flaky failure tracking so tools can fail once again."""
    _FLAKY_FAILED_ONCE.clear()


def _maybe_fail_flaky(tool_name: str) -> str | None:
    """Simulate a one-time transient tool failure when --flaky is active."""
    if _FLAKY_MODE and tool_name not in _FLAKY_FAILED_ONCE:
        _FLAKY_FAILED_ONCE.add(tool_name)
        return (
            f"Error: Transient tool timeout in {tool_name} "
            "(simulated --flaky failure). Please retry the tool call."
        )
    return None


def reset_sandbox() -> Path:
    """Copy the pristine novatech_store.db to novatech_sandbox.db."""
    if not STORE_DB.exists():
        raise FileNotFoundError(f"Source database not found: {STORE_DB}")
    shutil.copy2(STORE_DB, SANDBOX_DB)
    return SANDBOX_DB


def _sandbox_ro_uri() -> str:
    """Return a strict read-only SQLite URI for novatech_sandbox.db."""
    if not SANDBOX_DB.exists():
        reset_sandbox()
    return f"{SANDBOX_DB.as_uri()}?mode=ro"


def _strip_sql_comments_and_literals(sql: str) -> tuple[str, str]:
    """Return (sql_without_comments, sql_without_comments_or_literals)."""
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_comments = re.sub(r"--[^\n]*", " ", no_block).strip()
    no_literals = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " '' ", no_comments)
    return no_comments, no_literals


def _validate_select_only(sql: str) -> str | None:
    """Validate that sql is a single read-only SELECT/WITH statement."""
    cleaned, no_literals = _strip_sql_comments_and_literals(sql)
    if not cleaned:
        return "Error: Rejected empty SQL query."

    # Allow a single optional trailing semicolon, reject multiple statements.
    body = cleaned.rstrip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    no_lit_body = no_literals.rstrip()
    if no_lit_body.endswith(";"):
        no_lit_body = no_lit_body[:-1].rstrip()

    if ";" in no_lit_body:
        return "Error: Rejected query. Multiple SQL statements are not permitted."

    if not re.match(r"^(SELECT|WITH)\b", body, re.IGNORECASE):
        return "Error: Rejected query. Only read-only SELECT statements are permitted."

    match = _FORBIDDEN_SQL_RE.search(no_lit_body)
    if match:
        return (
            f"Error: Rejected query containing forbidden keyword '{match.group(1).upper()}'. "
            "Only read-only SELECT statements are permitted."
        )
    return None


def _get_vector_store() -> InMemoryVectorStore:
    """Lazily load, chunk, embed, and index all 14 NovaTech policy docs."""
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown docs found in {DOCS_DIR}")

    docs = [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        )
        for path in md_files
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    _VECTOR_STORE = InMemoryVectorStore.from_documents(chunks, embeddings)
    return _VECTOR_STORE


@tool
def query_database(sql: str) -> str:
    """Run a read-only SQL SELECT query against the NovaTech sandbox SQLite database.

    Use this to look up customers, products, orders, order_items, and returns.
    Only single SELECT (or WITH ... SELECT) statements are permitted.
    """
    flaky_err = _maybe_fail_flaky("query_database")
    if flaky_err:
        return flaky_err

    validation_err = _validate_select_only(sql)
    if validation_err:
        return validation_err

    try:
        with sqlite3.connect(_sandbox_ro_uri(), uri=True) as conn:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in (cursor.description or [])]
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        return f"SQL Error: {exc}"

    if not rows:
        return "Query returned 0 rows."

    max_rows = 50
    shown_rows = rows[:max_rows]
    lines = [" | ".join(columns), "-" * max(20, len(" | ".join(columns)))]
    for row in shown_rows:
        lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows truncated; total {len(rows)})")
    return "\n".join(lines)


@tool
def search_policies(query: str) -> str:
    """Search NovaTech store policy documents (returns, warranty, refunds, cancellation, loyalty, shipping).

    Returns the top 3 matching document chunks with their source filenames.
    """
    flaky_err = _maybe_fail_flaky("search_policies")
    if flaky_err:
        return flaky_err

    store = _get_vector_store()
    results = store.similarity_search(query, k=3)
    if not results:
        return "No matching policy sections found."

    formatted = [
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content.strip()}"
        for doc in results
    ]
    return "\n\n---\n\n".join(formatted)


@tool
def calculate_days(start_date: str, end_date: str) -> str:
    """Calculate the exact number of days from start_date to end_date (both YYYY-MM-DD)."""
    try:
        start = datetime.strptime(start_date.strip(), "%Y-%m-%d").date()
        end = datetime.strptime(end_date.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        return f"Date Error: expected YYYY-MM-DD format ({exc})."

    diff = (end - start).days
    return f"{diff} days (from {start.isoformat()} to {end.isoformat()})"


def _parse_manager_decision(decision: Any) -> tuple[bool, str]:
    """Normalize Order Manager interrupt resume payload into (approved, reason)."""
    if isinstance(decision, dict):
        raw_choice = str(decision.get("decision", "")).strip().lower()
        reason = str(decision.get("reason", "")).strip()
    else:
        raw_choice = str(decision).strip().lower()
        reason = ""

    approved = raw_choice in {"approve", "approved", "yes", "y"}
    return approved, reason


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order if it is currently in 'processing' status, after Order Manager approval.

    Always checks the database first. If the order does not exist or its status is not
    'processing', returns an error immediately without requesting approval or modifying data.
    """
    oid = order_id.strip()
    if not oid:
        return "Error: order_id cannot be empty."

    with sqlite3.connect(_sandbox_ro_uri(), uri=True) as conn:
        row = conn.execute(
            """
            SELECT o.order_id, c.full_name, o.order_date, o.status, o.payment_method, o.total_jod
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.order_id = ?
            """,
            (oid,),
        ).fetchone()

    if row is None:
        return f"Error: Order '{oid}' was not found in the database. No changes were made."

    found_id, customer_name, order_date, status, payment_method, total_jod = row
    if status != "processing":
        return (
            f"Error: Order '{found_id}' (customer: {customer_name}, total: {total_jod:.2f} JOD) "
            f"cannot be cancelled because its status is '{status}'. "
            "Only orders in 'processing' status can be cancelled."
        )

    # IMPORTANT: interrupt() must NOT be inside a try/except Exception block.
    decision = interrupt(
        {
            "action": "cancel_order",
            "order_id": found_id,
            "customer_name": customer_name,
            "order_date": order_date,
            "status": status,
            "payment_method": payment_method,
            "total_jod": total_jod,
            "message": (
                f"Order Manager approval required to cancel order {found_id} "
                f"for {customer_name} (status={status}, total={total_jod:.2f} JOD, "
                f"payment_method={payment_method})."
            ),
        }
    )

    approved, reason = _parse_manager_decision(decision)
    if not approved:
        reason_text = f" Reason: {reason}" if reason else ""
        return (
            f"Cancellation rejected by Order Manager for order '{found_id}'.{reason_text} "
            f"Order status remains '{status}'."
        )

    with sqlite3.connect(SANDBOX_DB) as conn:
        cursor = conn.execute(
            "UPDATE orders SET status = 'cancelled' WHERE order_id = ? AND status = 'processing'",
            (found_id,),
        )
        conn.commit()
        updated = cursor.rowcount

    if updated != 1:
        return (
            f"Error: Order '{found_id}' could not be updated (it may no longer be in 'processing' status)."
        )

    return (
        f"Order '{found_id}' for {customer_name} ({total_jod:.2f} JOD, payment method: "
        f"{payment_method}) has been approved by the Order Manager and cancelled. "
        "Database status is now 'cancelled'."
    )


TOOLS = [query_database, search_policies, calculate_days, cancel_order]
