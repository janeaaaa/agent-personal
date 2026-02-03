
import chromadb
from chromadb.config import Settings as ChromaSettings
import os
from pathlib import Path

chroma_dir = r'f:\BaiduNetdiskDownload\15套原创热门Agent项目源码\双十二\高精度多模态文档解析\AgenticRAGOCR\backend\data\chroma_db'
if not os.path.exists(chroma_dir):
    print(f"Chroma dir not found at {chroma_dir}")
    exit(1)

client = chromadb.PersistentClient(
    path=chroma_dir,
    settings=ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

doc_id = '432a54d9-e2ba-4f2f-9aba-c35821850d86'
collection_name = f"doc_{doc_id}"

try:
    collection = client.get_collection(name=collection_name)
    data = collection.get(limit=5, include=["documents", "metadatas"])
    print(f"Found {len(data['documents'])} sample documents in collection {collection_name}")
    for i in range(len(data['documents'])):
        print(f"DOC {i}: {repr(data['documents'][i])}")
        print(f"METADATA {i}: {data['metadatas'][i]}")
except Exception as e:
    print(f"Error: {e}")
