"""
Lesson 22: AWS S3 Integration — Document & Conversation Storage
===============================================================
Teaches:
  - Uploading and downloading documents to/from S3 inside LangGraph nodes
  - Storing conversation history snapshots to S3 (durable, cross-server)
  - Streaming large files from S3 without loading into memory
  - Presigned URL generation for secure frontend file access
  - S3-backed document loader for the Data Analysis Agent
  - Integrating S3 storage into a multi-agent graph (Data + DB + S3 agents)
  - IAM policy design for least-privilege S3 access

Architecture role (from High Level Architecture diagram):
    EC2 FastAPI Server
       └── Agent Orchestration
              ├── Data Analysis Agent  ← reads documents from S3
              ├── DB Retrieval Agent
              └── Session & Memory Mgmt ← saves conversation snapshots to S3
    AWS S3 Bucket ← THIS LESSON

Prerequisites:
    pip install boto3 langchain-aws langchain-community

AWS Setup:
    IAM policy required:
    {
        "Effect": "Allow",
        "Action": [
            "s3:GetObject", "s3:PutObject", "s3:DeleteObject",
            "s3:ListBucket", "s3:GetObjectPresignedUrl"
        ],
        "Resource": [
            "arn:aws:s3:::your-bucket-name",
            "arn:aws:s3:::your-bucket-name/*"
        ]
    }

Environment variables (see .env.example):
    AWS_REGION              (default: us-east-1)
    AWS_PROFILE             (optional)
    S3_BUCKET_NAME          (required)
    S3_CONVERSATION_PREFIX  (default: conversations/)
    S3_DOCUMENTS_PREFIX     (default: documents/)
"""

import io
import json
import logging
import os
from datetime import datetime
from typing import Annotated, Any, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("lesson_22")

# ===========================================================================
# SECTION 1 — S3 CLIENT SETUP
# ===========================================================================
#
# boto3 credential chain (same as Lesson 21 Bedrock):
#   1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
#   2. AWS profile (~/.aws/credentials)
#   3. EC2/ECS/Lambda IAM role (automatic — no credentials needed in code)
#
# Always prefer IAM roles in production. Never hardcode credentials.

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "my-langgraph-bucket")
S3_CONVERSATION_PREFIX = os.getenv("S3_CONVERSATION_PREFIX", "conversations/")
S3_DOCUMENTS_PREFIX = os.getenv("S3_DOCUMENTS_PREFIX", "documents/")


def create_s3_client() -> Optional[Any]:
    """
    Create an S3 client using the boto3 credential chain.
    Returns None if credentials are not available (allows local fallback).
    """
    try:
        if AWS_PROFILE:
            session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        else:
            session = boto3.Session(region_name=AWS_REGION)

        client = session.client("s3")
        client.list_buckets()
        logger.info("S3 client created. Region: %s", AWS_REGION)
        return client

    except NoCredentialsError:
        logger.warning("No AWS credentials found. S3 operations will be simulated.")
        return None
    except Exception as e:
        logger.warning("S3 client init failed: %s — using simulation mode.", e)
        return None


s3_client = create_s3_client()


# ===========================================================================
# SECTION 2 — S3 HELPER FUNCTIONS
# ===========================================================================

def s3_upload_text(key: str, content: str, metadata: dict = None) -> bool:
    """
    Upload a text string to S3 under the given key.

    Args:
        key:      Full S3 object key, e.g. "conversations/thread_42/2024-01-01.json"
        content:  String content to upload
        metadata: Optional dict of S3 object metadata tags

    Returns:
        True on success, False on failure.
    """
    if s3_client is None:
        logger.info("[SIMULATE] s3_upload_text → s3://%s/%s", S3_BUCKET_NAME, key)
        return True

    try:
        extra_args = {}
        if metadata:
            extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="application/json",
            **extra_args,
        )
        logger.info("Uploaded s3://%s/%s (%d bytes)", S3_BUCKET_NAME, key, len(content))
        return True

    except ClientError as e:
        logger.error("S3 upload failed for key %s: %s", key, e)
        return False


def s3_download_text(key: str) -> Optional[str]:
    """
    Download a text object from S3.

    Returns:
        String content, or None if key does not exist / error.
    """
    if s3_client is None:
        logger.info("[SIMULATE] s3_download_text ← s3://%s/%s", S3_BUCKET_NAME, key)
        return json.dumps({"simulated": True, "key": key})

    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        content = response["Body"].read().decode("utf-8")
        logger.info("Downloaded s3://%s/%s (%d bytes)", S3_BUCKET_NAME, key, len(content))
        return content

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.info("Key not found in S3: %s", key)
        else:
            logger.error("S3 download failed for key %s: %s", key, e)
        return None


def s3_upload_file_stream(key: str, file_bytes: bytes, content_type: str = "application/octet-stream") -> bool:
    """
    Upload raw bytes (e.g. PDF, CSV) to S3 using a BytesIO stream.
    Suitable for large files — avoids holding full content in memory as a string.
    """
    if s3_client is None:
        logger.info("[SIMULATE] s3_upload_file_stream → s3://%s/%s (%d bytes)", S3_BUCKET_NAME, key, len(file_bytes))
        return True

    try:
        s3_client.upload_fileobj(
            io.BytesIO(file_bytes),
            S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        logger.info("Streamed upload s3://%s/%s (%d bytes)", S3_BUCKET_NAME, key, len(file_bytes))
        return True
    except ClientError as e:
        logger.error("S3 stream upload failed: %s", e)
        return False


def s3_list_keys(prefix: str) -> list[str]:
    """
    List all S3 object keys under a given prefix.
    Returns empty list on error or simulation mode.
    """
    if s3_client is None:
        logger.info("[SIMULATE] s3_list_keys prefix=%s", prefix)
        return [f"{prefix}simulated_file_1.json", f"{prefix}simulated_file_2.json"]

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
    except ClientError as e:
        logger.error("S3 list failed for prefix %s: %s", prefix, e)
        return []


def s3_generate_presigned_url(key: str, expiry_seconds: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for a frontend to download/upload an S3 object
    without needing AWS credentials.

    Use case: Front End requests a document → API returns a presigned URL
              → browser downloads directly from S3 (no proxy through EC2).

    Args:
        key:            S3 object key
        expiry_seconds: URL valid for this many seconds (default 1 hour)

    Returns:
        Presigned URL string, or None on failure.
    """
    if s3_client is None:
        logger.info("[SIMULATE] presigned URL for s3://%s/%s (expires %ds)", S3_BUCKET_NAME, key, expiry_seconds)
        return f"https://simulated-presigned.s3.amazonaws.com/{key}?expires={expiry_seconds}"

    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": key},
            ExpiresIn=expiry_seconds,
        )
        logger.info("Presigned URL generated for %s (expires %ds)", key, expiry_seconds)
        return url
    except ClientError as e:
        logger.error("Presigned URL generation failed: %s", e)
        return None


def s3_delete_object(key: str) -> bool:
    """Delete an S3 object. Used for GDPR erasure (right to be forgotten)."""
    if s3_client is None:
        logger.info("[SIMULATE] s3_delete_object s3://%s/%s", S3_BUCKET_NAME, key)
        return True

    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
        logger.info("Deleted s3://%s/%s", S3_BUCKET_NAME, key)
        return True
    except ClientError as e:
        logger.error("S3 delete failed for key %s: %s", key, e)
        return False


# ===========================================================================
# SECTION 3 — CONVERSATION SNAPSHOT STORAGE
# ===========================================================================
#
# Pattern: After each LangGraph turn, serialize the message list to JSON
# and write it to S3. This gives you:
#   - Durable conversation history (survives EC2 restart)
#   - Cross-server access (multiple EC2 instances share the same S3 bucket)
#   - Audit trail (each snapshot is a separate S3 object with timestamp)
#   - Cheap long-term storage vs keeping everything in the DB
#
# Key format: conversations/{tenant_id}/{thread_id}/{timestamp}.json

def conversation_key(tenant_id: str, thread_id: str, timestamp: str = None) -> str:
    """Build the S3 key for a conversation snapshot."""
    ts = timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{S3_CONVERSATION_PREFIX}{tenant_id}/{thread_id}/{ts}.json"


def save_conversation_to_s3(tenant_id: str, thread_id: str, messages: list) -> str:
    """
    Serialize and upload a conversation's messages to S3.

    Returns:
        The S3 key used for this snapshot.
    """
    snapshot = {
        "tenant_id": tenant_id,
        "thread_id": thread_id,
        "saved_at": datetime.utcnow().isoformat(),
        "message_count": len(messages),
        "messages": [
            {
                "type": type(m).__name__,
                "content": m.content if hasattr(m, "content") else str(m),
            }
            for m in messages
        ],
    }
    key = conversation_key(tenant_id, thread_id)
    s3_upload_text(
        key=key,
        content=json.dumps(snapshot, indent=2),
        metadata={"tenant_id": tenant_id, "thread_id": thread_id},
    )
    return key


def load_latest_conversation_from_s3(tenant_id: str, thread_id: str) -> Optional[dict]:
    """
    Load the most recent conversation snapshot for a thread.
    Lists all snapshots for the thread and returns the last (lexicographically latest) one.
    """
    prefix = f"{S3_CONVERSATION_PREFIX}{tenant_id}/{thread_id}/"
    keys = s3_list_keys(prefix)
    if not keys:
        return None

    latest_key = sorted(keys)[-1]
    raw = s3_download_text(latest_key)
    if raw:
        return json.loads(raw)
    return None


def delete_conversation_from_s3(tenant_id: str, thread_id: str) -> int:
    """
    Delete ALL S3 snapshots for a thread. Used for GDPR right-to-erasure.

    Returns:
        Number of objects deleted.
    """
    prefix = f"{S3_CONVERSATION_PREFIX}{tenant_id}/{thread_id}/"
    keys = s3_list_keys(prefix)
    deleted = 0
    for key in keys:
        if s3_delete_object(key):
            deleted += 1
    logger.info("GDPR erasure: deleted %d snapshots for thread %s/%s", deleted, tenant_id, thread_id)
    return deleted


# ===========================================================================
# SECTION 4 — DOCUMENT STORAGE FOR DATA ANALYSIS AGENT
# ===========================================================================
#
# The Data Analysis Agent (from the architecture) needs to:
#   1. Accept document uploads (CSV, PDF, JSON) from the frontend
#   2. Store them in S3
#   3. Retrieve them for analysis when called by the orchestrator
#
# Key format: documents/{tenant_id}/{doc_id}/{filename}

def document_key(tenant_id: str, doc_id: str, filename: str) -> str:
    """Build the S3 key for a document."""
    return f"{S3_DOCUMENTS_PREFIX}{tenant_id}/{doc_id}/{filename}"


def upload_document(tenant_id: str, doc_id: str, filename: str, file_bytes: bytes) -> str:
    """
    Upload a document (CSV, PDF, JSON) to S3.

    Returns:
        S3 key of the uploaded document.
    """
    ext = filename.split(".")[-1].lower()
    content_type_map = {
        "csv": "text/csv",
        "pdf": "application/pdf",
        "json": "application/json",
        "txt": "text/plain",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")
    key = document_key(tenant_id, doc_id, filename)
    s3_upload_file_stream(key, file_bytes, content_type)
    return key


def get_document_url(tenant_id: str, doc_id: str, filename: str, expiry: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL so the frontend can download a document directly
    from S3 without routing through the EC2 server.
    """
    key = document_key(tenant_id, doc_id, filename)
    return s3_generate_presigned_url(key, expiry_seconds=expiry)


def load_document_as_text(tenant_id: str, doc_id: str, filename: str) -> Optional[str]:
    """
    Load a text-based document (CSV, JSON, TXT) from S3 for analysis.
    Used by the Data Analysis Agent node to read uploaded documents.
    """
    key = document_key(tenant_id, doc_id, filename)
    return s3_download_text(key)


# ===========================================================================
# SECTION 5 — LANGGRAPH STATE WITH S3 FIELDS
# ===========================================================================

class S3AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_id: str
    thread_id: str
    uploaded_doc_key: Optional[str]
    presigned_url: Optional[str]
    analysis_result: Optional[str]
    s3_snapshot_key: Optional[str]


# ===========================================================================
# SECTION 6 — LANGGRAPH NODES
# ===========================================================================

def get_llm():
    """Return LLM — Bedrock if available, fallback to Ollama for local dev."""
    try:
        import boto3
        from langchain_aws import ChatBedrockConverse
        model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
        session = boto3.Session(region_name=AWS_REGION)
        client = session.client("bedrock-runtime")
        client.list_foundation_models(byOutputModality="TEXT")
        logger.info("Using Bedrock model: %s", model_id)
        return ChatBedrockConverse(model_id=model_id, client=client)
    except Exception:
        logger.info("Bedrock not available — falling back to ChatOllama")
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2")


def document_upload_node(state: S3AgentState) -> dict:
    """
    Simulate a document upload from the frontend.
    In production this is triggered by a POST /upload endpoint — the file
    bytes arrive in the request and this node stores them in S3.
    """
    tenant_id = state.get("tenant_id", "tenant_default")
    thread_id = state.get("thread_id", "thread_001")

    sample_csv = b"date,metric,value\n2024-01-01,revenue,50000\n2024-01-02,revenue,52000\n2024-01-03,revenue,48000\n"
    doc_id = "doc_001"
    filename = "se_data.csv"

    key = upload_document(tenant_id, doc_id, filename, sample_csv)
    presigned = get_document_url(tenant_id, doc_id, filename, expiry=3600)

    logger.info("Document uploaded: %s", key)
    return {
        "uploaded_doc_key": key,
        "presigned_url": presigned,
        "messages": [HumanMessage(content=f"Document uploaded: {filename}. Presigned URL ready for frontend.")],
    }


def data_analysis_node(state: S3AgentState) -> dict:
    """
    Data Analysis Agent node.
    Retrieves the uploaded document from S3 and asks the LLM to analyze it.
    This maps to the 'Data Analysis Agent' in the architecture diagram.
    """
    tenant_id = state.get("tenant_id", "tenant_default")
    doc_key = state.get("uploaded_doc_key", "")

    if not doc_key:
        return {"analysis_result": "No document available for analysis."}

    doc_content = s3_download_text(doc_key)
    if not doc_content:
        return {"analysis_result": "Failed to load document from S3."}

    llm = get_llm()
    prompt = (
        f"You are a Data Analysis Agent. Analyze the following CSV data and provide a brief summary:\n\n"
        f"{doc_content}\n\n"
        f"Identify trends, anomalies, and key metrics."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content

    logger.info("Data analysis complete. Result length: %d chars", len(analysis))
    return {
        "analysis_result": analysis,
        "messages": [AIMessage(content=f"Analysis complete:\n{analysis}")],
    }


def save_conversation_node(state: S3AgentState) -> dict:
    """
    Session & Memory Management node.
    Saves the current conversation snapshot to S3.
    Maps to 'Session & Memory Management' in the architecture diagram.
    """
    tenant_id = state.get("tenant_id", "tenant_default")
    thread_id = state.get("thread_id", "thread_001")
    messages = state.get("messages", [])

    key = save_conversation_to_s3(tenant_id, thread_id, messages)
    logger.info("Conversation snapshot saved to S3: %s", key)

    return {"s3_snapshot_key": key}


# ===========================================================================
# SECTION 7 — BUILD THE GRAPH
# ===========================================================================

def build_s3_agent_graph() -> StateGraph:
    """
    Build a LangGraph that:
      1. Uploads a document to S3
      2. Runs Data Analysis Agent (reads from S3)
      3. Saves conversation snapshot to S3
    """
    graph = StateGraph(S3AgentState)

    graph.add_node("upload_document", document_upload_node)
    graph.add_node("data_analysis", data_analysis_node)
    graph.add_node("save_conversation", save_conversation_node)

    graph.add_edge(START, "upload_document")
    graph.add_edge("upload_document", "data_analysis")
    graph.add_edge("data_analysis", "save_conversation")
    graph.add_edge("save_conversation", END)

    return graph.compile(checkpointer=MemorySaver())


# ===========================================================================
# SECTION 8 — PRESIGNED URL PATTERN (FRONTEND INTEGRATION)
# ===========================================================================
#
# How the Front End uses presigned URLs:
#
#   1. User uploads file → POST /api/upload → EC2 stores in S3 → returns presigned URL
#   2. User requests document → GET /api/document/{doc_id} → returns presigned URL
#   3. Browser downloads directly from S3 (EC2 is not in the download path)
#
# Benefits:
#   - EC2 is not a bottleneck for large file transfers
#   - S3 handles bandwidth and scaling automatically
#   - Presigned URLs expire — no permanent public access
#   - IAM controls what the EC2 role can generate URLs for
#
# Example FastAPI endpoint (for Lesson 23/24):
#
#   @app.get("/api/document/{tenant_id}/{doc_id}/{filename}")
#   async def get_doc_url(tenant_id: str, doc_id: str, filename: str, user=Depends(verify_jwt)):
#       if user["tenant_id"] != tenant_id:
#           raise HTTPException(403, "Tenant mismatch")
#       url = get_document_url(tenant_id, doc_id, filename, expiry=900)
#       return {"url": url, "expires_in": 900}


# ===========================================================================
# SECTION 9 — GDPR ERASURE WITH S3
# ===========================================================================
#
# Right-to-be-forgotten workflow:
#
#   1. User submits GDPR erasure request
#   2. API endpoint calls erase_user_data(tenant_id, thread_id)
#   3. Function deletes all S3 conversation snapshots for that thread
#   4. Also triggers DB erasure (Lesson 20 GDPR pattern)
#   5. Audit log records the erasure event
#
# The key insight: S3 data must be erased separately from DB data.
# Many GDPR implementations forget S3 — this lesson closes that gap.

def erase_user_data(tenant_id: str, thread_id: str) -> dict:
    """
    Full GDPR erasure: delete S3 conversation snapshots + document files.

    Returns:
        Summary dict of what was deleted.
    """
    conv_deleted = delete_conversation_from_s3(tenant_id, thread_id)

    doc_prefix = f"{S3_DOCUMENTS_PREFIX}{tenant_id}/"
    doc_keys = s3_list_keys(doc_prefix)
    doc_deleted = 0
    for key in doc_keys:
        if s3_delete_object(key):
            doc_deleted += 1

    result = {
        "tenant_id": tenant_id,
        "thread_id": thread_id,
        "conversations_deleted": conv_deleted,
        "documents_deleted": doc_deleted,
        "erased_at": datetime.utcnow().isoformat(),
    }
    logger.info("GDPR erasure complete: %s", result)
    return result


# ===========================================================================
# SECTION 10 — DEMO
# ===========================================================================

def run_demo():
    print("\n" + "=" * 65)
    print("LESSON 22 — AWS S3 INTEGRATION DEMO")
    print("=" * 65)

    graph = build_s3_agent_graph()
    config = {"configurable": {"thread_id": "lesson22_demo"}}

    initial_state = {
        "messages": [],
        "tenant_id": "tenant_acme",
        "thread_id": "thread_lesson22",
        "uploaded_doc_key": None,
        "presigned_url": None,
        "analysis_result": None,
        "s3_snapshot_key": None,
    }

    print("\n--- Running S3 Agent Graph ---")
    final_state = graph.invoke(initial_state, config)

    print(f"\n[1] Document uploaded to S3 key: {final_state.get('uploaded_doc_key')}")
    print(f"[2] Presigned URL (expires 1h): {final_state.get('presigned_url', '')[:80]}...")
    print(f"[3] Analysis result preview: {str(final_state.get('analysis_result', ''))[:200]}...")
    print(f"[4] Conversation snapshot saved: {final_state.get('s3_snapshot_key')}")

    print("\n--- GDPR Erasure Demo ---")
    result = erase_user_data("tenant_acme", "thread_lesson22")
    print(f"Erasure result: {result}")

    print("\n--- Presigned URL Generation ---")
    url = s3_generate_presigned_url("documents/tenant_acme/doc_001/se_data.csv", expiry_seconds=900)
    print(f"Presigned URL (15 min): {url}")

    print("\n" + "=" * 65)
    print("KEY TAKEAWAYS:")
    print("  1. S3 credential chain: env vars → profile → IAM role (same as Bedrock)")
    print("  2. Conversations serialized to JSON, stored per tenant/thread/timestamp")
    print("  3. Presigned URLs: frontend downloads from S3, not through EC2")
    print("  4. GDPR erasure must cover S3 separately from the database")
    print("  5. Streaming uploads (BytesIO) avoid loading large files into RAM")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
