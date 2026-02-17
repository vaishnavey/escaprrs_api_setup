#!/usr/bin/env python3
"""
FastAPI app that:
- downloads the private repo Vaishnavey/testprrs server-side using HUGGINGFACEHUB_API_TOKEN (secret)
- loads processor.py (PROCESSOR_PATH) and calls process_sequence(...) for /analyze
- returns JSON with summary, table, and base64-encoded plots (so clients never get raw files)
- optional API_KEY required via header X-API-KEY
"""
import os
import logging
import threading
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from importlib.machinery import SourceFileLoader

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="PRRS Private Repo Proxy")

# Config (from env)
REPO_ID = os.environ.get("REPO_ID", "Vaishnavey/testprrs")
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
PROCESSOR_PATH = os.environ.get("PROCESSOR_PATH", "processor.py")
PROCESSOR_FUNC = os.environ.get("PROCESSOR_FUNC", "process_sequence")
CACHE_DIR = os.environ.get("CACHE_DIR", "/app/repo_cache")
API_KEY = os.environ.get("API_KEY")  # optional simple client key

REPO_DIR: Optional[str] = None
USER_PROCESSOR = None

class AnalyzeRequest(BaseModel):
    sequence: str
    options: Dict[str, Any] = {}

def load_processor_from_path(repo_dir: str, rel_path: str, func_name: str):
    abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(repo_dir, rel_path)
    if not os.path.exists(abs_path):
        logger.error("Processor path not found: %s", abs_path)
        return None
    try:
        loader = SourceFileLoader("user_processor_module", abs_path)
        mod = loader.load_module()  # type: ignore
        func = getattr(mod, func_name, None)
        if func and callable(func):
            logger.info("Loaded processor function %s from %s", func_name, abs_path)
            return func
        logger.error("Processor function %s not found in %s", func_name, abs_path)
    except Exception:
        logger.exception("Failed to load processor from %s", abs_path)
    return None

def download_and_setup():
    global REPO_DIR, USER_PROCESSOR
    if not HF_TOKEN:
        logger.error("HUGGINGFACEHUB_API_TOKEN not set; cannot download repo.")
        return
    try:
        # Try modern param name first, fall back to older name for compatibility
        try:
            REPO_DIR = snapshot_download(repo_id=REPO_ID, cache_dir=CACHE_DIR, token=HF_TOKEN)
        except TypeError:
            REPO_DIR = snapshot_download(repo_id=REPO_ID, cache_dir=CACHE_DIR, use_auth_token=HF_TOKEN)
        logger.info("Downloaded repo snapshot to %s", REPO_DIR)
    except Exception:
        logger.exception("Failed to download repo snapshot")
        REPO_DIR = None
    if REPO_DIR:
        USER_PROCESSOR = load_processor_from_path(REPO_DIR, PROCESSOR_PATH, PROCESSOR_FUNC)
        if not USER_PROCESSOR:
            logger.warning("Using fallback behavior since processor failed to load.")

# background startup
@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=download_and_setup, daemon=True)
    t.start()

@app.get("/health")

def health():
    return {
        "status": "ok",
        "version": API_VERSION,
        "processor_has_closest_match_always": True  
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, x_api_key: Optional[str] = Header(None)):
    # API key guard (optional)
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    seq = (req.sequence or "").strip()
    if not seq:
        raise HTTPException(status_code=400, detail="Empty sequence")

    # Ensure processor is loaded
    if USER_PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Processor not loaded (check logs)")

    try:
        result = USER_PROCESSOR(seq, req.options or {})
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Processor returned unexpected type")
        out = {
            "summary": result.get("summary", ""),
            "table": result.get("table", []),
            "plots": result.get("plots", [])
        }
        return out
    except Exception as exc:
        logger.exception("Processor error")
        raise HTTPException(status_code=500, detail=f"Processor error: {exc}")
