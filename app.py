import json
import os
import sys
import threading
import uuid
import numpy as np

FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    _BASE = sys._MEIPASS
    os.environ.setdefault("HF_HOME", os.path.join(_BASE, "models"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    HERE = _BASE
else:
    HERE = os.path.dirname(os.path.abspath(__file__))

def load_env_file(path):
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except OSError:
        pass


load_env_file(os.path.join(HERE, ".env"))

import webview
from sqlalchemy import text

import database as db
import secret

INDEX = os.path.join(HERE, "web", "index.html")
SETTINGS_FILE = db.APP_DIR / "settings.json"

DEFAULT_KEY = secret.load_default_key(HERE)
DEFAULT_MODEL = os.environ.get("MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.environ.get("BASE_URL", "")

os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")


def load_settings():
    """Настройки LLM из settings.json; при отсутствии — из переменных окружения"""
    data = {}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {
        "api_key": data.get("api_key", ""),
        "model": data.get("model") or DEFAULT_MODEL,
        "base_url": data.get("base_url") or DEFAULT_BASE_URL,
    }


def save_settings(data):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")


_PLAIN_EXT = {".txt", ".md", ".markdown"}
_markitdown = None


def _extract_text(path):
    """Достаёт текст из файла произвольного формата"""
    if os.path.splitext(path)[1].lower() in _PLAIN_EXT:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    global _markitdown
    if _markitdown is None:
        from markitdown import MarkItDown
        _markitdown = MarkItDown()
    return _markitdown.convert(path).text_content


class Api:
    def __init__(self):
        self._ingestor = None
        self._agent = None
        self._lock = threading.RLock()
        self._cancel = None
        self._loading = False
        self._error = None
        self._settings = load_settings()
        self._ingest = {"active": False, "done": 0, "total": 0,
                        "phase": "", "result": None}

    def _get_ingestor(self):
        with self._lock:
            if self._ingestor is None:
                from parce import Ingestor
                self._ingestor = Ingestor()
            return self._ingestor

    def _effective_key(self):
        """Ключ пользователя, а если он пуст — скрытый дефолтный"""
        return self._settings.get("api_key") or DEFAULT_KEY

    def _get_agent(self):
        with self._lock:
            if self._agent is None:
                from agent import Agent
                s = self._settings
                self._agent = Agent(
                    ingestor=self._get_ingestor(),
                    api_key=self._effective_key() or None,
                    model=s.get("model") or None,
                    base_url=s.get("base_url") or None,
                )
            return self._agent

    def warmup(self):
        """Прогрев моделей в фоне врум врум маквин готов"""
        def run():
            with self._lock:
                self._loading = True
            try:
                self._get_ingestor()
            except Exception as e:
                self._error = str(e)
            finally:
                with self._lock:
                    self._loading = False
        threading.Thread(target=run, daemon=True).start()

    def status(self):
        with self._lock:
            return {
                "ready": self._ingestor is not None,
                "loading": self._loading,
                "error": self._error,
            }

    def get_settings(self):
        with self._lock:
            user_key = self._settings.get("api_key", "")
            return {
                "api_key": user_key,
                "model": self._settings.get("model", ""),
                "base_url": self._settings.get("base_url", ""),
                "using_default": not user_key,
            }

    def save_settings(self, api_key, model, base_url):
        with self._lock:
            self._settings = {"api_key": (api_key or "").strip(),
                              "model": model or "", "base_url": base_url or ""}
            save_settings(self._settings)
            self._agent = None
        return {"ok": True}

    def get_document(self, doc_id):
        with db.get_engine().begin() as conn:
            row = conn.execute(text(
                "SELECT title, text, tag FROM documents WHERE doc_id = :d"
            ), {"d": doc_id}).fetchone()
        if not row:
            return {"ok": False, "error": "Документ не найден"}
        return {"ok": True, "title": row[0], "text": row[1], "tag": row[2]}

    def list_documents(self):
        with db.get_engine().begin() as conn:
            rows = conn.execute(text(
                "SELECT d.doc_id, d.title, d.tag, count(c.id) AS n "
                "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id "
                "GROUP BY d.doc_id, d.title, d.tag ORDER BY d.title"
            )).fetchall()
        return [{"doc_id": r[0], "title": r[1], "tag": r[2], "chunks": r[3]} for r in rows]

    def _start_ingest(self, docs):
        """Запускает обработку документов в фоне"""
        with self._lock:
            if self._ingest["active"]:
                return {"ok": False, "error": "Идёт обработка другого источника, подождите"}
            self._ingest = {"active": True, "done": 0, "total": 0,
                            "phase": "Подготовка", "result": None}

        def prog(done, total, phase=None):
            with self._lock:
                self._ingest["done"] = done
                self._ingest["total"] = total
                if phase:
                    self._ingest["phase"] = phase

        def run():
            try:
                n = self._get_ingestor().add_documents(docs, progress=prog)
                db.create_index("chunks", "emb_small", "cosine")
                result = {"ok": True, "files": len(docs), "chunks": n}
            except Exception as e:
                result = {"ok": False, "error": str(e)}
            finally:
                with self._lock:
                    self._ingest["active"] = False
                    self._ingest["result"] = result

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, "started": True, "files": len(docs)}

    def ingest_status(self):
        with self._lock:
            return dict(self._ingest)

    def get_tags(self):
        """Уникальные непустые теги уже загруженных источников"""
        with db.get_engine().begin() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT tag FROM documents "
                "WHERE tag IS NOT NULL AND tag <> '' ORDER BY tag"
            )).fetchall()
        return [r[0] for r in rows]

    def add_document(self, title, content, tag=None):
        if not (content or "").strip():
            return {"ok": False, "error": "Пустой документ"}
        doc = {"doc_id": uuid.uuid4().hex, "title": title or "Без названия",
               "text": content, "tag": (tag or "").strip() or None}
        return self._start_ingest([doc])

    def add_files(self, tag=None):
        window = webview.active_window()
        paths = window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=True,
            file_types=(
                "Поддерживаемые файлы "
                "(*.pdf;*.docx;*.pptx;*.xlsx;*.xls;*.html;*.htm;*.csv;*.epub;*.md;*.markdown;*.txt)",
                "PDF (*.pdf)",
                "Word (*.docx)",
                "PowerPoint (*.pptx)",
                "Excel (*.xlsx;*.xls)",
                "HTML (*.html;*.htm)",
                "Текст (*.txt;*.md;*.markdown)",
                "Все файлы (*.*)",
            ),
        )
        if not paths:
            return {"ok": False, "cancelled": True}
        docs, failed = [], []
        for p in paths:
            try:
                content = _extract_text(p)
            except Exception:
                failed.append(os.path.basename(p))
                continue
            if not (content or "").strip():
                continue
            docs.append({
                "doc_id": os.path.abspath(p),
                "title": os.path.splitext(os.path.basename(p))[0],
                "text": content,
                "tag": (tag or "").strip() or None,
            })
        if not docs:
            msg = "Не удалось извлечь текст"
            if failed:
                msg += ": " + ", ".join(failed)
            return {"ok": False, "error": msg}
        return self._start_ingest(docs)

    def delete_document(self, doc_id):
        with db.get_engine().begin() as conn:
            conn.execute(text("DELETE FROM documents WHERE doc_id = :d"), {"d": doc_id})
        return {"ok": True}

    def ask(self, question):
        if not (question or "").strip():
            return {"ok": False, "error": "Пустой вопрос"}
        agent = self._get_agent()
        import agent as agent_mod
        cancel = threading.Event()
        with self._lock:
            self._cancel = cancel
        try:
            return {"ok": True, **agent.answer(question, cancel=cancel)}
        except agent_mod.Cancelled:
            return {"ok": False, "cancelled": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            with self._lock:
                if self._cancel is cancel:
                    self._cancel = None

    def stop(self):
        """Прервать текущий ответ агента на ближайшей контрольной точке"""
        with self._lock:
            if self._cancel is not None:
                self._cancel.set()
        return {"ok": True}

    def new_chat(self):
        """Сбросить историю диалога"""
        with self._lock:
            if self._agent is not None:
                self._agent.reset_history()
        return {"ok": True}

    def mind_map(self, threshold=0.4, column="emb_small", min_chunks=1):
        """создает (необязательно связный) граф файлов, где рёбра отражают связанность документов по смыслу 
        +
        документы с одинаковымим тэгами находятся рядом
        """
        sql = f"""
        WITH doc_vec AS (
            SELECT doc_id, avg({column}) AS v
            FROM chunks
            GROUP BY doc_id
            HAVING count(*) >= %s
        ),
        pairs AS (
            SELECT a.doc_id AS a_id, b.doc_id AS b_id, a.v <=> b.v AS dist
            FROM doc_vec a
            JOIN doc_vec b ON a.doc_id < b.doc_id
        )
        SELECT p.a_id, da.title, p.b_id, db.title, p.dist
        FROM pairs p
        JOIN documents da ON da.doc_id = p.a_id
        JOIN documents db ON db.doc_id = p.b_id
        WHERE p.dist <= %s
        ORDER BY p.dist
        """
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (min_chunks, threshold))
            res = cur.fetchall()
        edges ={}
        for i in res:
            if not (edges.get(i[0])): edges[i[0]] = [(i[2], i[4])]
            else: edges[i[0]].append((i[2],i[4]))
        sql = f"""
        SELECT tag, doc_id, title FROM documents
        """ # достаем тэги, айдишники и названия всех документов
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(sql)
            tags = cur.fetchall()
        verts = {}
        for i in tags:
            if not (verts.get(i[0])): verts[i[0]] = [(i[1], i[2])]
            else: verts[i[0]].append((i[1],i[2]))      
        return edges, verts



def main():
    db.get_engine()
    api = Api()
    webview.create_window("Notebook", INDEX, js_api=api,
                          width=1100, height=740, min_size=(800, 560))
    api.warmup()
    gui = "qt" if sys.platform.startswith("linux") else None
    try:
        webview.start(gui=gui)
    finally:
        db.stop_server()


if __name__ == "__main__":
    main()
