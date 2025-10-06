import os
import json
from typing import Optional, Dict, Any, List, Union
from bson import ObjectId
from core.db import get_db, get_gridfs, GRIDFS_BUCKET


def upload_file(bucket: Optional[str], filename: str, local_path: str, metadata: Optional[Dict] = None) -> ObjectId:
    """Upload a local file to GridFS. Returns the ObjectId of the stored file."""
    db = get_db()
    fs = get_gridfs()
    with open(local_path, "rb") as f:
        file_id = fs.put(f, filename=filename, metadata=metadata or {})
    return file_id


def upload_bytes(bucket: Optional[str], filename: str, data: bytes, metadata: Optional[Dict] = None) -> ObjectId:
    """Upload bytes content to GridFS. Returns the ObjectId."""
    db = get_db()
    fs = get_gridfs()
    file_id = fs.put(data, filename=filename, metadata=metadata or {})
    return file_id


def download_file(bucket: Optional[str], filename_or_id: Union[str, ObjectId], target_path: str) -> None:
    """Download a file (by filename or ObjectId) to local path."""
    db = get_db()
    fs = get_gridfs()
    file_obj = None
    try:
        if ObjectId.is_valid(str(filename_or_id)):
            file_obj = fs.get(ObjectId(str(filename_or_id)))
        else:
            file_obj = fs.find_one({"filename": filename_or_id})
    except Exception:
        file_obj = None

    if not file_obj:
        raise FileNotFoundError(f"File {filename_or_id} not found in GridFS bucket {GRIDFS_BUCKET}")

    with open(target_path, "wb") as f:
        f.write(file_obj.read())


def download_bytes(bucket: Optional[str], filename_or_id: Union[str, ObjectId]) -> bytes:
    """Return file bytes for filename or ObjectId."""
    db = get_db()
    fs = get_gridfs()
    file_obj = None
    try:
        if ObjectId.is_valid(str(filename_or_id)):
            file_obj = fs.get(ObjectId(str(filename_or_id)))
        else:
            file_obj = fs.find_one({"filename": filename_or_id})
    except Exception:
        file_obj = None

    if not file_obj:
        raise FileNotFoundError(f"File {filename_or_id} not found in GridFS bucket {GRIDFS_BUCKET}")
    return file_obj.read()


def list_files(bucket: Optional[str], prefix: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """List files stored in GridFS. Use prefix to filter by filename start."""
    db = get_db()
    fs = get_gridfs()
    query = {}
    if prefix:
        # GridFS filenames are stored in 'filename'
        query["filename"] = {"$regex": f"^{prefix}"}
    cursor = db[f"{GRIDFS_BUCKET}.files"].find(query).limit(limit)
    results = []
    for doc in cursor:
        results.append({
            "_id": doc.get("_id"),
            "filename": doc.get("filename"),
            "length": doc.get("length"),
            "uploadDate": doc.get("uploadDate"),
            "metadata": doc.get("metadata"),
        })
    return results


def delete_file(bucket: Optional[str], filename_or_id: Union[str, ObjectId]) -> None:
    """Delete a GridFS file by ObjectId or filename (deletes first match)."""
    db = get_db()
    fs = get_gridfs()
    if ObjectId.is_valid(str(filename_or_id)):
        fs.delete(ObjectId(str(filename_or_id)))
        return
    # find by filename
    doc = db[f"{GRIDFS_BUCKET}.files"].find_one({"filename": filename_or_id})
    if not doc:
        raise FileNotFoundError(f"File {filename_or_id} not found to delete")
    fs.delete(doc["_id"])


# --- Config helpers (store small JSON configs in a dedicated collection) ---

def save_config(name: str, value: Dict[str, Any]) -> Dict[str, Any]:
    """Insert or update a named config document.

    Returns the saved document.
    """
    db = get_db()
    fs = get_gridfs()
    coll = db.get_collection("configs")
    existing = coll.find_one({"name": name})
    if existing:
        coll.update_one({"_id": existing["_id"]}, {"$set": {"value": value, "updated_at": db.client.server_info().get('localTime')}})
        existing = coll.find_one({"_id": existing["_id"]})
        return existing
    doc = {"name": name, "value": value}
    res = coll.insert_one(doc)
    doc["_id"] = res.inserted_id
    return doc


def get_config(name: str) -> Optional[Dict[str, Any]]:
    db = get_db()
    fs = get_gridfs()
    coll = db.get_collection("configs")
    doc = coll.find_one({"name": name})
    return doc


def list_configs(limit: int = 100) -> List[Dict[str, Any]]:
    db = get_db()
    fs = get_gridfs()
    coll = db.get_collection("configs")
    docs = coll.find().limit(limit)
    return list(docs)


# --- Preview helpers ---
def save_preview(name: str, preview: Dict[str, Any]) -> Dict[str, Any]:
    """Save a small preview document for a recently uploaded file.

    Document shape: {name, preview, created_at}
    """
    db = get_db()
    fs = get_gridfs()
    coll = db.get_collection("previews")
    doc = {"name": name, "preview": preview}
    res = coll.insert_one(doc)
    doc["_id"] = res.inserted_id
    return doc


def list_previews(limit: int = 20) -> List[Dict[str, Any]]:
    db = get_db()
    fs = get_gridfs()
    coll = db.get_collection("previews")
    docs = list(coll.find().sort([("_id", -1)]).limit(limit))
    return docs
