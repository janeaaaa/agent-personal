
import sqlite3
import os

db_path = r'f:\BaiduNetdiskDownload\15套原创热门Agent项目源码\双十二\高精度多模态文档解析\AgenticRAGOCR\backend\data\tasks.db'
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
doc_id = '432a54d9-e2ba-4f2f-9aba-c35821850d86'
cursor.execute('SELECT content FROM blocks WHERE doc_id=? LIMIT 5', (doc_id,))
rows = cursor.fetchall()
print(f"Found {len(rows)} blocks for doc {doc_id}")
for i, row in enumerate(rows):
    print(f"BLOCK {i}: {repr(row[0])}")
conn.close()
