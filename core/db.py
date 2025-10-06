"""Minimal MongoDB connection helpers.

This module centralizes reading MONGO_URI, MONGO_DB and GRIDFS_BUCKET from env (via python-dotenv)
and provides get_client(), get_db(), and get_gridfs().
"""
import os
from typing import Tuple, Optional
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "mmm_app_db")
GRIDFS_BUCKET = os.getenv("GRIDFS_BUCKET", "project_files")

# If running on Streamlit Cloud, prefer values provided in st.secrets
try:
    import streamlit as _st
    # st.secrets behaves like a dict-like mapping
    if hasattr(_st, "secrets") and _st.secrets:
        if _st.secrets.get("MONGO_URI"):
            MONGO_URI = _st.secrets.get("MONGO_URI")
        if _st.secrets.get("MONGO_DB"):
            MONGO_DB = _st.secrets.get("MONGO_DB")
        if _st.secrets.get("GRIDFS_BUCKET"):
            GRIDFS_BUCKET = _st.secrets.get("GRIDFS_BUCKET")
except Exception:
    # streamlit may not be available in non-Streamlit runtimes; ignore
    pass


_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI is not set in environment")
        _client = MongoClient(MONGO_URI)
    return _client


def get_db():
    client = get_client()
    return client[MONGO_DB]


def get_gridfs():
    db = get_db()
    return gridfs.GridFS(db, collection=GRIDFS_BUCKET)
