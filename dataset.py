import glob
import itertools
from datasets import load_dataset, interleave_datasets
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EMB = 300_000
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
OUT_FILE = "data.npz"
BATCH_SZ = 64
LANGS = ["en", "ru"]
MMARCO_N = 50_000
SHARD = 5_000
SHARD_DIR = "shards"
MIRACL_TOTAL = 9366


def mmarco_triples(n_max, lang):
    base = "hf://datasets/unicamp-dl/mmarco/data"

    def tsv(path, cols):
        return load_dataset("csv", data_files=f"{base}/{path}", delimiter="\t",
                            column_names=cols, streaming=True, split="train")
    triples, need_q, need_p = [], set(), set()
    for i, r in enumerate(tsv("triples.train.ids.small.tsv", ["qid", "pos", "neg"])):
        if i >= n_max:
            break
        qid, pos, neg = str(r["qid"]), str(r["pos"]), str(r["neg"])
        triples.append((qid, pos, neg))
        need_q.add(qid); need_p.add(pos); need_p.add(neg)
    qtext = {}
    for r in tsv(f"google/queries/train/{lang}_queries.train.tsv", ["qid", "text"]):
        k = str(r["qid"])
        if k in need_q:
            qtext[k] = r["text"]
            if len(qtext) == len(need_q):
                break
    ptext = {}
    for r in tsv(f"google/collections/{lang}_collection.tsv", ["pid", "text"]):
        k = str(r["pid"])
        if k in need_p:
            ptext[k] = r["text"]
            if len(ptext) == len(need_p):
                break
    for qid, pos, neg in triples:
        if qid in qtext and pos in ptext and neg in ptext:
            yield {"anchor": qtext[qid], "positive": ptext[pos], "negative": ptext[neg]}


def build_stream():
    streams = [
        load_dataset("sentence-transformers/miracl", f"{lang}-triplet",
                     split="train", streaming=True)
        for lang in LANGS
    ]
    ds = interleave_datasets(streams, stopping_strategy="all_exhausted")
    if MMARCO_N > 0:
        ds = itertools.chain(ds, mmarco_triples(MMARCO_N, "russian"),
                             mmarco_triples(MMARCO_N, "english"))
    return ds


def shard_paths():
    return sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npz")))


def count_done():
    return sum(int(os.path.basename(p).split("_")[2].split(".")[0]) for p in shard_paths())


def save_shard(idx, bq, bp, bn):
    if not bq:
        return idx
    q = np.concatenate(bq).astype(np.float32)
    p = np.concatenate(bp).astype(np.float32)
    n = np.concatenate(bn).astype(np.float32)
    os.makedirs(SHARD_DIR, exist_ok=True)
    np.savez(os.path.join(SHARD_DIR, f"shard_{idx:05d}_{len(q)}.npz"),
             query=q, positive=p, negative=n)
    bq.clear(); bp.clear(); bn.clear()
    print(f"  сохранён shard_{idx:05d} ({len(q)} пар)")
    return idx + 1


def encode_batch(embeder, qb, pb, nb, bq, bp, bn):
    if not qb:
        return
    bq.append(embeder.encode(qb, prompt_name="query", normalize_embeddings=True, batch_size=BATCH_SZ))
    bp.append(embeder.encode(pb, normalize_embeddings=True, batch_size=BATCH_SZ))
    bn.append(embeder.encode(nb, normalize_embeddings=True, batch_size=BATCH_SZ))
    qb.clear(); pb.clear(); nb.clear()


def merge():
    ss = shard_paths()
    if not ss:
        print("нет шардов для склейки")
        return
    np.savez(OUT_FILE,
             query=np.concatenate([np.load(s)["query"] for s in ss]),
             positive=np.concatenate([np.load(s)["positive"] for s in ss]),
             negative=np.concatenate([np.load(s)["negative"] for s in ss]))
    print(f"склеено шардов: {len(ss)} -> {OUT_FILE}")


if __name__ == "__main__":
    os.makedirs(SHARD_DIR, exist_ok=True)
    done = count_done()
    shard_i = len(shard_paths())
    N = MMARCO_N * 2 + MIRACL_TOTAL
    print(f"возобновляю с {done} / {N} (шардов уже {shard_i})")

    embeder = SentenceTransformer(EMB_MODEL, device=DEVICE)
    embeder.max_seq_length = 512

    ds = itertools.islice(build_stream(), done, None)

    qb, pb, nb = [], [], []
    bq, bp, bn = [], [], []
    processed = done
    try:
        for row in ds:
            if processed >= N_EMB:
                break
            qb.append(row["anchor"]); pb.append(row["positive"]); nb.append(row["negative"])
            processed += 1
            if processed % BATCH_SZ == 0:
                print(f"{processed} / {N}")
            if len(qb) == BATCH_SZ:
                encode_batch(embeder, qb, pb, nb, bq, bp, bn)
                if sum(len(a) for a in bq) >= SHARD:
                    shard_i = save_shard(shard_i, bq, bp, bn)
    except KeyboardInterrupt:
        print("\nпрервано — дописываю накопленное в шард")
    finally:
        encode_batch(embeder, qb, pb, nb, bq, bp, bn)
        shard_i = save_shard(shard_i, bq, bp, bn)

    merge()
    # Итератор streaming-датасета оставляет живой фоновый поток загрузки,
    # который падает на финализации интерпретатора (PyGILState_Release / abort).
    # Выходим в обход финализации.
    sys.stdout.flush()
    os._exit(0)
