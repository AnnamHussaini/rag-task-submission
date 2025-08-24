# app/rag_pipeline.py
import os, json, uuid, re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from tqdm import tqdm
import requests

# If Tesseract is not on PATH (Windows), uncomment and set the correct path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SRC_DIR = os.path.join(DATA_DIR, 'source_pdfs')
PROC_DIR = os.path.join(DATA_DIR, 'processed')
INDEX_DIR = os.path.join(DATA_DIR, 'chroma')
COLLECTION_NAME = 'docs'

EMBED_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
QUERY_PREFIX = 'Query: '

@dataclass
class Page:
    text: str
    page_num: int
    source: str

# PDF → text (with OCR fallback)
def extract_pdf(path: str, ocr_dpi: int = 220) -> List[Page]:
    doc = fitz.open(path)
    pages: List[Page] = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text").strip()
        if len(text) < 20:  # likely scanned → OCR fallback
            mat = fitz.Matrix(ocr_dpi/72, ocr_dpi/72)
            pix = page.get_pixmap(matrix=mat)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang='eng')
        pages.append(Page(text=text, page_num=i+1, source=os.path.basename(path)))
    return pages

# Chunking
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# Build embeddings + Chroma index 
class Indexer:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.client = chromadb.PersistentClient(path=index_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def _emb(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs).tolist()

    def clear(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, docs: List[Dict]):
        ids = [d['id'] for d in docs]
        texts = [d['text'] for d in docs]
        metas = [d['metadata'] for d in docs]
        embs = self._emb(texts)
        self.collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    def search(self, query: str, k: int = 4):
        q = QUERY_PREFIX + query
        q_emb = self._emb([q])
        res = self.collection.query(query_embeddings=q_emb, n_results=k)
        out = []
        for text, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
            out.append({"text": text, "metadata": meta, "score": 1 - dist})
        return out

# Ingest pipeline 
def ingest_all(src_dir: str = SRC_DIR, save_intermediate: bool = True):
    os.makedirs(PROC_DIR, exist_ok=True)
    idx = Indexer(INDEX_DIR)

    all_docs = []
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(src_dir, fname)
        print(f"Extracting {fname} …")
        pages = extract_pdf(path)
        per_file = []
        for p in pages:
            for j, ch in enumerate(chunk_text(p.text)):
                doc = {
                    'id': str(uuid.uuid4()),
                    'text': ch,
                    'metadata': {
                        'source': p.source,
                        'page': p.page_num,
                        'chunk': j
                    }
                }
                per_file.append(doc)
                all_docs.append(doc)
        if save_intermediate:
            with open(os.path.join(PROC_DIR, f"{os.path.splitext(fname)[0]}_chunks.jsonl"), 'w', encoding='utf-8') as f:
                for d in per_file:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if all_docs:
        print("Indexing …")
        idx.add_documents(all_docs)
        print(f"Indexed {len(all_docs)} chunks.")
    else:
        print("No PDFs found in data/source_pdfs.")

# Generator via Ollama (local) 
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2')

def ollama_generate(prompt: str) -> str:
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=600)
        r.raise_for_status()
        return r.json().get('response', '').strip()
    except Exception as e:
        return f"[Generator unavailable] {e}"

# Ask 
ANSWER_PROMPT = """
You are a helpful assistant that answers ONLY using the provided context. If the answer is not present in the context, say "I don't know from the provided documents." After the answer, include citations like [source: <filename>, page <n>]. Keep it concise.

Question: {question}

Context:
{context}
"""

def build_context(hits: List[Dict]) -> Tuple[str, List[Tuple[str,int]]]:
    ctx_blocks = []
    cites = []
    for h in hits:
        src = h['metadata']['source']; page = h['metadata']['page']
        cites.append((src, page))
        header = f"[source={src} page={page}]"
        ctx_blocks.append(header + "\n" + h['text'])
    return ("\n\n---\n\n".join(ctx_blocks), cites)

# simple evidence highlighting
def highlight(text: str, query: str) -> str:
    tokens = [t.lower() for t in re.findall(r"\w+", query)]
    tokens = [t for t in tokens if len(t) > 2]
    def repl(m):
        w = m.group(0)
        return f"**{w}**" if w.lower() in tokens else w
    return re.sub(r"\w+", repl, text)

def ask(question: str, k: int = 4) -> Dict:
    idx = Indexer(INDEX_DIR)
    hits = idx.search(question, k=k)
    for h in hits:
        h['text'] = highlight(h['text'], question)

    context, cites = build_context(hits)
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    answer = ollama_generate(prompt)

    seen, ordered = set(), []
    for c in cites:
        if c not in seen:
            ordered.append(c); seen.add(c)

    return {
        'answer': answer,
        'citations': ordered,
        'contexts': hits,
    }

# CLI 
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')
    p_ing = sub.add_parser('ingest'); p_ing.add_argument('--src', default=SRC_DIR)
    p_ask = sub.add_parser('ask');   p_ask.add_argument('question'); p_ask.add_argument('-k', type=int, default=4)
    args = p.parse_args()

    if args.cmd == 'ingest':
        ingest_all(args.src)
    elif args.cmd == 'ask':
        res = ask(args.question, k=args.k)
        print("\nAnswer:\n", res['answer'])
        print("\nCitations:")
        for s, pg in res['citations']:
            print(f" - {s} (page {pg})")
        print("\nTop contexts (with highlighted evidence):")
        for i, h in enumerate(res['contexts'], 1):
            m = h['metadata']; print(f"[{i}] {m['source']} p.{m['page']} (score={h['score']:.3f})\n{h['text'][:800]}\n")
    else:
        p.print_help()

