import os
import json
import numpy as np
from lightrag.llm.ollama import ollama_embed
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import re
import asyncio

class RagQuery:
    def __init__(self, edition, embedding_dim=1024, storage_dir=None):
        self.edition = edition
        self.embedding_dim = embedding_dim
        self.storage_dir = storage_dir or f"./DandD/{edition}/rag_storage/vector_store"
        self.meta_dir = storage_dir or f"./DandD/{edition}/rag_storage"
        self.vectors = []
        self.meta = []

        # Load all vectors and metadata
        vec_files = sorted(Path(self.storage_dir).glob("vec_*.npy"))
        for vec_file in vec_files:
            vectors = np.load(vec_file)
            self.vectors.append(vectors)

            idx = vec_file.stem.split("_")[1]
            meta_file = Path(self.meta_dir) / f"meta_{idx}.jsonl"
            if not meta_file.exists():
                continue
            with open(meta_file, "r", encoding="utf-8") as f:
                self.meta.extend([json.loads(line) for line in f])

        if self.vectors:
            self.vectors = np.vstack(self.vectors)
        else:
            self.vectors = np.zeros((0, embedding_dim), dtype=np.float32)

    async def query(self, query_text, top_k=5, max_chunks_per_file=3):
        """
        Query the RAG store and return LLM-friendly concise text.
        :param query_text: The user query string
        :param top_k: Number of top chunks to retrieve
        :param max_chunks_per_file: Limit of mini-chunks per matched file
        :return: Concatenated text for LLM consumption
        """
        if self.vectors.shape[0] == 0:
            print("Warning: Vector store is empty.")
            return ""

        # Embed query
        query_vec = await ollama_embed([query_text], embed_model=os.environ["EMBEDDED_OLLAMA_MODEL"])
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if query_vec.shape[1] != self.embedding_dim:
            query_vec = query_vec[:, :self.embedding_dim]

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.vectors)[0]

        # Get top-k matches
        top_indices = similarities.argsort()[::-1][:top_k]
        top_meta = [self.meta[idx] for idx in top_indices]

        llm_chunks = []
        seen_files = set()

        for r in top_meta:
            file_path = Path(r["source_path"])
            if file_path.is_absolute():
                file_path = Path(".") / Path(*file_path.parts[1:])
            if not file_path.exists() or file_path in seen_files:
                continue
            seen_files.add(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                # Split into smaller paragraph-level mini-chunks
                mini_chunks = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
                
                # Pick only the first few mini-chunks per file to keep output concise
                for chunk in mini_chunks[:max_chunks_per_file]:
                    llm_chunks.append(f"Source: {file_path.name}\n{chunk}")

        # Combine chunks with clear separators
        return "\n\n---\n\n".join(llm_chunks)


# Example usage
async def main():
    rq = RagQuery("5e")
    # Query for concise info about fireball
    llm_ready_text = await rq.query("cleric", top_k=5, max_chunks_per_file=2)

    # Save to file
    output_file = Path("./DandD/5e/rag_storage/query_results.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(llm_ready_text)

    print(f"LLM-ready query results saved to {output_file}")
    print(llm_ready_text)

asyncio.run(main())
