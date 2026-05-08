"""
Biologist Agent — RAG retrieval layer.
Builds a ChromaDB vector store from knowledge_base.json and retrieves
relevant biological constraints for any user query.
"""

import json
import hashlib
from pathlib import Path
import chromadb

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator

BASE = Path(__file__).resolve().parents[2]
KB_FILE = BASE / "results" / "knowledge_base.json"
VECTOR_STORE_PATH = BASE / "data" / "vector_store"

COLLECTION_NAME = "bio_constraints"


def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


class BiologistAgent:
    def __init__(self, rebuild: bool = False):
        VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))

        existing = [c.name for c in self.client.list_collections()]
        if rebuild and COLLECTION_NAME in existing:
            self.client.delete_collection(COLLECTION_NAME)
            existing = []

        if COLLECTION_NAME not in existing:
            print("Building vector store from knowledge base...")
            self.collection = self.client.create_collection(COLLECTION_NAME)
            self._index_knowledge_base()
            print(f"  Indexed {self.collection.count()} documents.")
        else:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            print(f"Loaded existing vector store ({self.collection.count()} docs).")

    def _index_knowledge_base(self):
        with open(KB_FILE) as f:
            kb = json.load(f)

        documents, metadatas, ids = [], [], []

        for entry in kb:
            image = entry["image"]
            category = entry["category"]
            labels = entry.get("labels", [])

            # Index each constraint individually
            for constraint in entry.get("constraints", []):
                doc_id = _doc_id(constraint)
                if doc_id not in ids:
                    documents.append(constraint)
                    metadatas.append({"image": image, "type": "constraint", "category": category})
                    ids.append(doc_id)

            # Index the label set as a combined document (broader coverage)
            if labels:
                label_doc = f"biological diagram showing: {', '.join(labels[:12])}"
                doc_id = _doc_id(image + label_doc)
                if doc_id not in ids:
                    documents.append(label_doc)
                    metadatas.append({"image": image, "type": "labels", "category": category})
                    ids.append(doc_id)

        # Batch insert (ChromaDB handles up to ~5000 at once)
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )

    @traceable(name="biologist_retrieve", run_type="retriever")
    def retrieve(self, query: str, top_k: int = 8) -> list[str]:
        """Return top_k constraint strings most relevant to the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"type": "constraint"},
        )
        constraints = results["documents"][0] if results["documents"] else []
        return constraints

    @traceable(name="biologist_retrieve_with_scores", run_type="retriever")
    def retrieve_with_scores(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        """Return (constraint, distance) pairs — lower distance = more relevant."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"type": "constraint"},
            include=["documents", "distances"],
        )
        docs = results["documents"][0] if results["documents"] else []
        dists = results["distances"][0] if results["distances"] else []
        return list(zip(docs, dists))
