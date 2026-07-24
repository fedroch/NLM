import os
import pathlib
import subprocess
import sys

import numpy as np
import pgserver
import psycopg
from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, text

IN_DIM = 1024
DEFAULT_OUT_DIM = 128

def _app_dir():
    """Каталог данных приложения (pgdata, settings.json, context.md).

    Postgres под Windows плохо переносит не-ASCII в путях: кириллица в имени
    пользователя ломает initdb, а backend падает при загрузке расширения
    (CREATE EXTENSION vector -> 'server closed the connection unexpectedly').
    Поэтому если домашний путь не-ASCII, уходим в ProgramData — он всегда ASCII.
    """
    home = pathlib.Path.home() / ".local" / "share" / "dim_reducer"
    if sys.platform == "win32":
        try:
            str(home).encode("ascii")
        except UnicodeEncodeError:
            return pathlib.Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "Notebook"
    return home


APP_DIR = _app_dir()
PGDATA = APP_DIR / "pgdata"

_server = None
_engine = None


def _hide_subprocess_windows():
    """В собранном оконном приложении (console=False) каждый дочерний процесс,
    запущенный через subprocess (pg_ctl, initdb и, каскадом, сам postgres),
    получает НОВОЕ консольное окно — на экране висит лишний терминал. Вешаем всем
    детям флаг CREATE_NO_WINDOW, чтобы окон не было."""
    if sys.platform != "win32" or getattr(subprocess, "_no_window_patched", False):
        return
    CREATE_NO_WINDOW = 0x08000000
    orig_init = subprocess.Popen.__init__
    def init(self, *args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | CREATE_NO_WINDOW
        orig_init(self, *args, **kwargs)
    subprocess.Popen.__init__ = init
    subprocess._no_window_patched = True


def _bump_pg_ctl_timeout(seconds=120):
    """pgserver зашивает timeout=10 в `pg_ctl start`. На Windows этого мало: после
    нештатного завершения Postgres делает crash-recovery с fsync всего каталога, а
    антивирус + 'sharing violation' на файле pgdata\\log растягивают старт на
    десятки секунд -> таймаут -> приложение падает. Поднимаем ожидание старта.
    pg_ctl импортируется в postgres_server по имени, поэтому патчим там."""
    import pgserver.postgres_server as pgs
    if getattr(pgs, "_timeout_patched", False):
        return
    orig = pgs.pg_ctl
    def patched(args, *a, **kw):
        if kw.get("timeout") is not None:
            kw["timeout"] = max(kw["timeout"], seconds)
        return orig(args, *a, **kw)
    pgs.pg_ctl = patched
    pgs._timeout_patched = True


def get_server():
    """Поднимает локальный сервер в PGDATA либо возвращает уже живой"""
    global _server
    if _server is None:
        _hide_subprocess_windows()
        _bump_pg_ctl_timeout()
        PGDATA.mkdir(parents=True, exist_ok=True)
        _server = pgserver.get_server(PGDATA)
    return _server


def get_uri(sqlalchemy=False):
    uri = get_server().get_uri()
    if sqlalchemy:
        return uri.replace("postgresql://", "postgresql+psycopg://", 1)
    return uri


def get_engine(out_dim=DEFAULT_OUT_DIM):
    global _engine
    if _engine is None:
        _engine = create_engine(get_uri(sqlalchemy=True))
        init_schema(_engine, out_dim)
    return _engine


def init_schema(engine, out_dim=DEFAULT_OUT_DIM):
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id     BIGSERIAL PRIMARY KEY,
                doc_id TEXT UNIQUE,
                title  TEXT,
                text   TEXT,
                tag    TEXT
            )
        """))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS tag TEXT"))
        conn.execute(text("ALTER TABLE documents DROP COLUMN IF EXISTS emb_full"))
        conn.execute(text("ALTER TABLE documents DROP COLUMN IF EXISTS emb_small"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id         BIGSERIAL PRIMARY KEY,
                doc_id     TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                chunk_no   INT NOT NULL,
                text       TEXT,
                char_start INT,
                char_end   INT,
                emb_full   vector({IN_DIM}),
                emb_small  vector({out_dim}),
                UNIQUE (doc_id, chunk_no)
            )
        """))
        conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS char_start INT"))
        conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS char_end INT"))


def connect():
    conn = psycopg.connect(get_uri())
    register_vector(conn)
    return conn


def upsert_documents_meta(rows):
    """Метаданные документа без векторов: (doc_id, title, text, tag)"""
    sql = """
        INSERT INTO documents (doc_id, title, text, tag)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (doc_id) DO UPDATE SET
            title = EXCLUDED.title,
            text  = EXCLUDED.text,
            tag   = EXCLUDED.tag
    """
    rows = list(rows)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def replace_chunks(rows):
    """Полностью заменяет чанки перечисленных документов
    rows: (doc_id, chunk_no, text, char_start, char_end, emb_full, emb_small)"""
    rows = list(rows)
    if not rows:
        return 0
    doc_ids = sorted({r[0] for r in rows})
    sql = """
        INSERT INTO chunks (doc_id, chunk_no, text, char_start, char_end, emb_full, emb_small)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", (doc_ids,))
            cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def create_index(table="chunks", column="emb_small", metric="cosine"):
    """HNSW-индекс"""
    ops = {
        "cosine": "vector_cosine_ops",
        "l2": "vector_l2_ops",
        "ip": "vector_ip_ops",
    }[metric]
    with get_engine().begin() as conn:
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS {table}_{column}_hnsw "
            f"ON {table} USING hnsw ({column} {ops})"
        ))


def search_chunks(vec, k=10, column="emb_small"):
    """k ближайших пассажей вместе с документом, которому они принадлежат"""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT c.doc_id, d.title, c.chunk_no, c.text, "
            f"       c.char_start, c.char_end, c.{column} <=> %s AS dist "
            f"FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            f"ORDER BY c.{column} <=> %s LIMIT %s",
            (vec, vec, k),            
        )
        return cur.fetchall()


def stop_server():
    global _server, _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
    if _server is not None:
        _server.cleanup()
        _server = None


if __name__ == "__main__":
    # тесты:
    rng = np.random.default_rng(42)

    get_engine(out_dim=DEFAULT_OUT_DIM)
    print(f"сервер поднят, данные в {PGDATA}")

    with get_engine().begin() as conn:
        conn.execute(text("TRUNCATE documents CASCADE"))

    meta = [(f"doc{i}", f"заголовок {i}", f"текст документа {i}", "тест") for i in range(20)]
    print(f"документов: {upsert_documents_meta(meta)}")
    chunks = [
        (
            f"doc{i}",
            j,
            f"пассаж {j} документа {i}",
            j * 10, j * 10 + 8,          # char_start, char_end (синтетика)
            rng.random(IN_DIM, dtype=np.float32),
            rng.random(DEFAULT_OUT_DIM, dtype=np.float32),
        )
        for i in range(20)
        for j in range(3)
    ]
    print(f"чанков: {replace_chunks(chunks)}")

    create_index("chunks", "emb_small", "cosine")
    print("HNSW-индекс построен")

    print("поиск по вектору чанка doc7#1 (ожидаем его первым, dist≈0):")
    for doc_id, title, chunk_no, ctext, cs, ce, dist in search_chunks(chunks[22][6], k=3):
        print(f"  {doc_id:6s} чанк#{chunk_no} [{cs}:{ce}] dist={dist:.6f} :: {ctext}")

    replace_chunks(chunks)
    with get_engine().begin() as conn:
        n = conn.execute(text("SELECT count(*) FROM chunks")).scalar()
    print(f"чанков после повторной заливки: {n} (ожидаем 60)")

    stop_server()
    print("сервер остановлен")
