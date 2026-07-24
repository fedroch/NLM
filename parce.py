import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import database as db
from model_class import percp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
_LOCAL_MODEL = os.path.join(_BASE, "model_qwen")
EMB_MODEL = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else "Qwen/Qwen3-Embedding-0.6B"
CKPT = os.path.join(_BASE, "compresser_128.pt")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_compressor(path=CKPT, device=DEVICE):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = percp(
        d_in=ck["in_dim"],
        d_out=ck["out_dim"],
        hidden1=ck["hidden1"],
        hidden2=ck["hidden2"],
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


class Ingestor:
    def __init__(self, ckpt=CKPT, emb_model=EMB_MODEL, device=DEVICE):
        self.device = device
        self.embedder = SentenceTransformer(emb_model, device=device)
        self.compressor = load_compressor(ckpt, device)

    def embed(self, texts, batch_size=32, progress=None):
        """Кодирует документы progress(done, total, phase)"""
        texts = list(texts)
        total = len(texts)
        if not total:
            return np.zeros((0, db.IN_DIM), dtype=np.float32)
        parts = []
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            emb = self.embedder.encode(
                batch, normalize_embeddings=True, batch_size=batch_size
            )
            parts.append(np.asarray(emb, dtype=np.float32))
            if progress:
                progress(min(i + len(batch), total), total, "Эмбеддинг")
        return np.vstack(parts)

    def embed_query(self, text, batch_size=32):
        """Кодирует поисковый запрос"""
        texts = [text] if isinstance(text, str) else list(text)
        emb = self.embedder.encode(
            texts,
            prompt_name="query",
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        return np.asarray(emb, dtype=np.float32)

    def compress(self, full):
        with torch.no_grad():
            t = torch.tensor(full, device=self.device)
            return self.compressor(t).cpu().numpy().astype(np.float32)

    def chunk_spans(self, text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        """Режет текст на пассажи по токенам модели"""
        enc = self.embedder.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        n = len(offs)
        if n == 0:
            return []
        if n <= chunk_size:
            return [(text, 0, len(text))]
        step = chunk_size - overlap
        spans = []
        for start in range(0, n, step):
            window = offs[start:start + chunk_size]
            if not window:
                break
            c0, c1 = window[0][0], window[-1][1]
            spans.append((text[c0:c1], c0, c1))
            if start + chunk_size >= n:
                break
        return spans

    def build_chunk_rows(self, docs, batch_size=32, progress=None):
        """(метаданные документов, строки чанков c char_start/char_end)"""
        meta, owners, offsets, to_embed, texts = [], [], [], [], []
        for d in docs:
            doc_id = d.get("doc_id")
            title = (d.get("title") or "").strip()
            meta.append((doc_id, d.get("title"), d.get("text"), d.get("tag")))
            for i, (piece, c0, c1) in enumerate(self.chunk_spans(d.get("text") or "")):
                owners.append((doc_id, i))
                offsets.append((c0, c1))
                texts.append(piece)
                to_embed.append(f"{title}\n{piece}" if title else piece)

        full = self.embed(to_embed, batch_size, progress)
        small = self.compress(full)
        rows = [
            (doc_id, no, t, o[0], o[1], f, s)
            for (doc_id, no), t, o, f, s in zip(owners, texts, offsets, full, small)
        ]
        return meta, rows

    def add_documents(self, docs, batch_size=32, progress=None):
        docs = list(docs)
        if not docs:
            return 0
        if progress:
            progress(0, 0, "Разбор на фрагменты")
        meta, rows = self.build_chunk_rows(docs, batch_size, progress)
        if progress:
            progress(len(rows), len(rows), "Запись в базу")
        db.upsert_documents_meta(meta)   # на них ссылается FK
        return db.replace_chunks(rows)


# if __name__ == "__main__":
    # from sqlalchemy import text as sql_text

    # db.get_engine()
    # with db.get_engine().begin() as conn:
    #     conn.execute(sql_text("TRUNCATE documents CASCADE"))

    # ing = Ingestor()
    # print("модели загружены")
    # long_text = (
    #     "Раздел про садоводство. Помидоры высаживают в грунт в мае, поливают тёплой водой. " * 40
    #     + "Раздел про автомобили. Двигатель внутреннего сгорания требует замены масла каждые 10 тысяч км. " * 40
    #     + "Раздел про астрономию. Юпитер — крупнейшая планета Солнечной системы, у неё десятки спутников. " * 40
    # )

    # docs = [
    #     {"doc_id": "d1", "title": "Кошки", "text": "Кошка — домашнее животное.", "tag": "био"},
    #     {"doc_id": "d2", "title": "Python", "text": "Python — язык программирования.", "tag": "it"},
    #     {"doc_id": "d3", "title": "Сборник", "text": long_text, "tag": "разное"},
    # ]
    # print(f"записано чанков: {ing.add_documents(docs)}")

    # with db.get_engine().begin() as conn:
    #     rows = conn.execute(sql_text(
    #         "SELECT d.doc_id, d.title, count(c.id) FROM documents d "
    #         "JOIN chunks c ON c.doc_id = d.doc_id GROUP BY d.doc_id, d.title ORDER BY d.doc_id"
    #     )).fetchall()
    # print("чанков на документ:")
    # for doc_id, title, cnt in rows:
    #     print(f"  {doc_id:4s} {title:10s} -> {cnt}")

    # db.create_index("chunks", "emb_small", "cosine")
    # q = ing.compress(ing.embed_query("сколько спутников у планеты Юпитер"))[0]
    # print("поиск про Юпитер (ждём пассаж об астрономии из d3):")
    # for doc_id, title, chunk_no, ctext, dist in db.search_chunks(q, k=3):
    #     print(f"  {doc_id:4s} {title:10s} чанк#{chunk_no} dist={dist:.4f} :: {ctext[:55]}...")
    # ing.add_documents([{"doc_id": "d3", "title": "Сборник", "text": "Теперь коротко.", "tag": "разное"}])
    # with db.get_engine().begin() as conn:
    #     cnt = conn.execute(sql_text("SELECT count(*) FROM chunks WHERE doc_id='d3'")).scalar()
    # print(f"чанков у d3 после укорачивания текста: {cnt} (было 9, ожидаем 1 — сироты удалены)")

    # db.stop_server()
