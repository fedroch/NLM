import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.nn.functional import normalize, mse_loss
import sys
import optuna
from model_class import percp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_FILE = "data.npz"
IN_DIM = 1024

OUT_DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 128
print(f" dim: {OUT_DIM}")
LR = 1e-3
EPOCHS = 96
K = 10
BATCH = 4096
HIDDEN1 = 1024
HIDDEN2 = 1024
EVAL_CAP = 5000


# class percp(nn.Module):                                                              !!!!!!!!!!!уже определен в model_calss 
#     def __init__(self, d_in=IN_DIM, d_out=OUT_DIM, hidden1=HIDDEN1, hidden2=HIDDEN2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_in, hidden1),
#             nn.GELU(),
#             nn.Linear(hidden1, hidden2),
#             nn.GELU(),
#             nn.Linear(hidden2, d_out),
#         )

#     def forward(self, x):
#         x = self.net(x)
#         return normalize(x)


def qd_loss(q_full, d_full, model):
    zq = model(q_full)
    zd = model(d_full)
    s_full = q_full @ d_full.T
    s_small = zq @ zd.T
    return mse_loss(s_small, s_full)


def train_projector(q, pos, neg=None, EPOCHS=EPOCHS, hid1=HIDDEN1, hid2=HIDDEN2):
    model = percp(hidden1=hid1, hidden2=hid2).to(DEVICE)
    qT = torch.tensor(np.asarray(q, dtype=np.float32), device=DEVICE)
    pT = torch.tensor(np.asarray(pos, dtype=np.float32), device=DEVICE)
    nT = torch.tensor(np.asarray(neg, dtype=np.float32), device=DEVICE) if neg is not None else None
    n = qT.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    schedudler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            qb, pb = qT[idx], pT[idx]
            docs = pb if nT is None else torch.cat([pb, nT[idx]], dim=0)
            loss = qd_loss(qb, docs, model)
            opt.zero_grad()
            loss.backward()
            opt.step()
            schedudler.step()
            total += loss.item() * len(idx)
        print(f"epoch {ep+1:2d}/{EPOCHS}  loss {total/n:.6f}")
    return model


def retrieval_recall(q, d, k=K):
    """Доля запросов, чей собственный позитив d[i] попал в top-k ближайших"""
    q = np.asarray(q, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    kk = min(k, d.shape[0] - 1)
    sim = q @ d.T
    topk = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    gold = np.arange(len(q))[:, None]
    return float(np.mean((topk == gold).any(axis=1)))


def norm(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)


def project(model, x):
    """Прогон массива через компрессор -> numpy (N, OUT_DIM)"""
    model.eval()
    with torch.no_grad():
        t = torch.tensor(np.asarray(x, dtype=np.float32), device=DEVICE)
        return model(t).cpu().numpy()


def optuna_train(trial, q_tr, pos_tr, neg_tr, q_va, pos_va):
    epochs = trial.suggest_int("epochs", 5, 100)
    hidden1 = trial.suggest_int("hidden1", max(OUT_DIM, 256), 1024, step=64)
    hidden2 = trial.suggest_int("hidden2", OUT_DIM, hidden1, step=64)
    model = train_projector(q_tr, pos_tr, neg_tr, EPOCHS=epochs, hid1=hidden1, hid2=hidden2)
    qv, pv = q_va[:EVAL_CAP], pos_va[:EVAL_CAP]
    rec = retrieval_recall(project(model, qv), project(model, pv))
    with open("optuna_hits.txt", "a") as f:
        f.write(f"recall: {rec}, epochs: {epochs}, hidden1: {hidden1}, hidden2: {hidden2} \n")
    return rec


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    data = np.load(OUT_FILE)
    q = data["query"].astype(np.float32)
    pos = data["positive"].astype(np.float32)
    neg = data["negative"].astype(np.float32)
    idx = np.arange(len(q))
    tr, tv = train_test_split(idx, train_size=0.8, test_size=0.2, random_state=42)
    te, va = train_test_split(tv, train_size=0.5, random_state=42)
    q_tr, pos_tr, neg_tr = q[tr], pos[tr], neg[tr]
    q_va, pos_va = q[va], pos[va]
    q_te, pos_te = q[te], pos[te]

    if len(sys.argv) > 1:
        if "default" in sys.argv[1]:
            model = train_projector(q_tr, pos_tr, neg_tr)
        elif "optuna" in sys.argv[1]:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda t: optuna_train(t, q_tr, pos_tr, neg_tr, q_va, pos_va),
                           n_trials=100)
            bp = study.best_params
            model = train_projector(q_tr, pos_tr, neg_tr,
                                    EPOCHS=bp["epochs"], hid1=bp["hidden1"], hid2=bp["hidden2"])
            print(bp.values())
            with open("optuna_hits.txt", "a") as f:
                f.write(f"best params: epochs = {bp['epochs']}, hid1 = {bp['hidden1']}, hid2 = {bp['hidden2']}\n")
        else:
            print("первый аргумент должен быть default либо optuna (а второй отвечает за размерность выходного вектора)")
            sys.exit(1)
    else:
        model = train_projector(q_tr, pos_tr, neg_tr)

    if len(sys.argv) > 3 and sys.argv[3] == "save":
        torch.save({
            "state_dict": model.state_dict(),
            "in_dim": IN_DIM,
            "out_dim": OUT_DIM,
            "hidden1": model.net[0].out_features,
            "hidden2": model.net[2].out_features,
        }, f"compresser_{OUT_DIM}.pt")

    r = retrieval_recall(project(model, q_te), project(model, pos_te))
    ceiling = retrieval_recall(q_te, pos_te)
    print(f"recall@{K}: {r:.4f}  ({IN_DIM} -> {OUT_DIM})   [потолок 1024d: {ceiling:.4f}]")
    pca = PCA(n_components=OUT_DIM).fit(np.concatenate([q_tr, pos_tr]))
    pca_result = retrieval_recall(norm(pca.transform(q_te).astype(np.float32)),
                                  norm(pca.transform(pos_te).astype(np.float32)))
    print(f"PCA    recall@{K}: {pca_result:.4f}")

    trunc_result = retrieval_recall(norm(q_te[:, :OUT_DIM]), norm(pos_te[:, :OUT_DIM]))
    print(f"Trunc  recall@{K}: {trunc_result:.4f}")

    return {"model": r, "pca": pca_result, "trunc": trunc_result}


if __name__ == "__main__":
    model, pca, trunc = main().values()
    if len(sys.argv) > 1:
        with open("dims_result.txt", "a") as file:
            file.write(f"------------dim= {OUT_DIM}, model= {sys.argv[1]}------------ \nmodel: {model}, pca: {pca}, trunc: {trunc}\n")
