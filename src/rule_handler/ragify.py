import os, sys, asyncio, datetime, re, pathlib, json, shutil, logging, numpy as np
from logging import Logger, handlers
from tqdm import tqdm
from _tokenizer import Chunker
from lightrag.llm.ollama import ollama_embed

import torch


EDITION_DIR = "/DandD/{edition}/"
SPLIT_DIR = "/DandD/{edition}/split_ruleset/"
WORKING_DIR = "/DandD/{edition}/rag_storage/"
LOG_DIR = "/DandD/{edition}/logs/"

LOG_FOLDER = "/logs/"
os.makedirs(LOG_FOLDER, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(LOG_FOLDER, f"ragify_{timestamp}.log")

logger = Logger("ragify_logger")
file_handler = handlers.RotatingFileHandler(
    log_file_path, maxBytes=20_000_000, backupCount=5
)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
logger.propagate = False
console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
logger.addHandler(console)



def complete_dir(edition):
    return EDITION_DIR.format(edition=edition)


def split_dir(edition):
    return SPLIT_DIR.format(edition=edition)


def working_dir(edition):
    return WORKING_DIR.format(edition=edition)


def log_dir(edition):
    return LOG_DIR.format(edition=edition)


def cleanup_incomplete(edition_path):
    edition_path = pathlib.Path(edition_path)
    complete_file = edition_path / ".ragify_complete"
    if not complete_file.exists():
        for folder_name in ["split_ruleset", "rag_storage", "logs"]:
            folder_path = edition_path / folder_name
            if folder_path.exists():
                logger.info(f"Deleting incomplete folder: {folder_path}")
                shutil.rmtree(folder_path)


def is_dnd_chunk(chunk: str) -> bool:
    keywords = [
        "spell",
        "armor",
        "weapon",
        "creature",
        "class",
        "level",
        "hit points",
        "attack",
        "damage",
        "strength",
        "dexterity",
        "constitution",
        "intelligence",
        "wisdom",
        "charisma",
        "proficiency",
        "saving throw",
        "initiative",
        "spellcasting",
        "damage type",
    ]
    chunk_lower = chunk.lower()
    return True


class Ragify:
    """Original class name retained for main.py compatibility, with bulk embedding insert."""

    def __init__(self, path):
        self.edition = str(path.split("/")[-1])
        edition_path = complete_dir(self.edition)
        cleanup_incomplete(edition_path)

        os.makedirs(log_dir(self.edition), exist_ok=True)
        os.makedirs(working_dir(self.edition), exist_ok=True)

        if not os.path.exists(split_dir(self.edition)):
            logger.info(f"Chunking ruleset for edition {self.edition}")
            Chunker().process_file(
                input_path=complete_dir(self.edition) + "ruleset.md",
                output_path=split_dir(self.edition),
            )

    def collect_chunk_files(self):
        files = []
        for dp, _, fns in os.walk(split_dir(self.edition)):
            for fn in sorted(fns):
                if fn.lower().endswith(".txt"):
                    files.append(os.path.join(dp, fn))
        return files

    def load_chunks_from_files(self, files):
        all_chunks = []
        all_paths = []
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if not text:
                    continue
                chunks = re.split(r"\r?\n\s*-{3,}\s*\r?\n", text)
                for c in chunks:
                    c = c.strip()
                    if not c:
                        continue
                    if is_dnd_chunk(c):
                        all_chunks.append(c)
                        all_paths.append(fp)
        return all_chunks, all_paths

    async def run_async(self, batch_size=128, embed_batch_size=64, embedding_dim=768):
        files = self.collect_chunk_files()

        logger.info(f"Found {len(files)} chunk files")
        chunks, paths = self.load_chunks_from_files(files)
        total = len(chunks)
        logger.info(f"Total filtered D&D chunks to ingest: {total}")
        if total == 0:
            logger.warning("No chunks found to ingest. Exiting.")
            return

        store_path = pathlib.Path(working_dir(self.edition)) / "vector_store"
        store_path.mkdir(parents=True, exist_ok=True)

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_paths = paths[i : i + batch_size]

            vectors_all = []
            # Split batch into embedding sub-batches
            for j in range(0, len(batch_chunks), embed_batch_size):
                sub_batch = batch_chunks[j : j + embed_batch_size]
                vectors = await ollama_embed(sub_batch, embed_model=os.environ["EMBEDDED_OLLAMA_MODEL"])
                vectors = np.asarray(vectors, dtype=np.float32)
                # Ensure embedding dimension
                if vectors.shape[1] != embedding_dim:
                    vectors = vectors[:, :embedding_dim]
                vectors_all.append(vectors)

            vectors_all = np.vstack(vectors_all)

            vec_file = store_path / f"vec_{i}.npy"
            meta_file = pathlib.Path(working_dir(self.edition)) / f"meta_{i}.jsonl"
            np.save(vec_file, vectors_all)

            with open(meta_file, "w", encoding="utf-8") as f:
                for k, _ in enumerate(batch_chunks):
                    meta = {
                        "chunk_index": i + k,
                        "source_path": batch_paths[k],
                        "edition": self.edition,
                    }
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            # Use logger.info to report progress
            logger.info(f"Ingested {min(i + batch_size, total)}/{total} chunks")

        pathlib.Path(complete_dir(self.edition) + "/.ragify_complete").touch()
        logger.info("Marked .ragify_complete")
        logger.info("All chunks ingested successfully")


    def run(self):
        asyncio.run(self.run_async(int(os.environ["batch_size"]), int(os.environ["embed_batch_size"]), int(os.environ["embedding_dim"])))
