import os

from openai import OpenAI, pydantic_function_tool
import json
import database as db
from parce import Ingestor
from pydantic import BaseModel, Field


class Cancelled(Exception):
    """Пользователь нажал «стоп» — ответ прерван на контрольной точке."""


DEFAULT_MODEL = os.environ.get("MODEL", "gpt-5.4-mini")

TOP_K = 10
DIST_THRESHOLD = 0.5
MAX_HISTORY_MSGS = 20

CONTEXT_FILE = db.APP_DIR / "context.md"

REWRITE_PROMPT = (
    "Ты помогаешь искать в базе документов. По вопросу пользователя составь "
    "короткий поисковый запрос: суть и ключевые слова того, что нужно найти. "
    "Убери всё лишнее (вежливость, контекст, рассуждения). "
    "Верни только запрос одной строкой, без пояснений и кавычек."
)

SYSTEM_PROMPT = (
    "Ты — ассистент. Если ниже есть контекст из базы документов — опирайся "
    "прежде всего на него и ссылайся на документы по заголовку. Если контекст "
    "пуст или в нём нет ответа — ответь по своим знаниям, но честно предупреди, "
    "что в загруженных документах этого нет. "
    "В контексте тебе даны только отдельные ФРАГМЕНТЫ документов. Если их не "
    "хватает для ответа или пользователь спрашивает про конкретный документ "
    "целиком — вызови инструмент read_file_by_name с точным заголовком нужного "
    "документа, чтобы получить его полный текст. "
    "Если из сообщения узнаёшь что-то важное о пользователе (имя, предпочтения, "
    "цели, факты о нём, историю вашего диалога) — сохрани это одной короткой заметкой через инструмент "
    "write_to_context, чтобы помнить это в следующих ответах."
)

class write_to_context_schema(BaseModel):
    text: str = Field(..., description="Текст про пользователя и контекст диалога, который нужно запомнить")
class read_file_by_name_schema(BaseModel):
    title: str = Field(..., description="Название файла, содержание которого нужно прочитать")

class Agent:
    def __init__(self, k=TOP_K, threshold=DIST_THRESHOLD, ingestor=None,
                 api_key=None, model=None, base_url=None, client=None):
        self.k = k
        self.threshold = threshold
        self.model = model or DEFAULT_MODEL
        self.ingestor = ingestor or Ingestor()
        self.client = client or OpenAI(
            api_key=api_key or os.environ.get("API_KEY"),
            base_url=base_url or os.environ.get("BASE_URL"),
        )
        self.history = []

    def reset_history(self):
        """Начать новый диалог — забыть предыдущие реплики."""
        self.history = []

    def _remember(self, question, answer):
        """Добавляет обмен в историю и обрезает её до последних MAX_HISTORY_MSGS."""
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > MAX_HISTORY_MSGS:
            self.history = self.history[-MAX_HISTORY_MSGS:]

    def write_to_context(self, text):
        """Дописывает заметку о пользователе/диалоге в общий файл памяти"""
        text = (text or "").strip()
        if not text:
            return
        CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONTEXT_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    def read_file_by_name(self, title):
        """Полный текст документа(ов) с указанным заголовком — для инструмента."""
        title = (title or "").strip()
        if not title:
            return "Не указано название файла."
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT title, text FROM documents WHERE lower(title) = lower(%s)",
                (title,),
            )
            rows = cur.fetchall()
        if not rows:
            return f"Документ с названием «{title}» не найден."
        return "\n\n".join(f"# {t}\n{txt}" for t, txt in rows)


    @staticmethod
    def _ck(cancel):
        """Контрольная точка отмены: бросает Cancelled, если нажали «стоп»."""
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    def _chat(self, system, user, use_memory=True, history=None, cancel=None):
        """Обмен с моделью. use_memory=True даёт инструмент write_to_context"""
        messages = [{"role": "system", "content": system}]
        if history:
            messages += history
        messages.append({"role": "user", "content": user})
        tools = [pydantic_function_tool(write_to_context_schema), pydantic_function_tool(read_file_by_name_schema)] if use_memory else None
        try:
            msg = None
            for _ in range(4):
                self._ck(cancel)
                kwargs = dict(model=self.model, messages=messages, temperature=0)
                if tools:
                    kwargs["tools"] = tools
                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                if not tools or not msg.tool_calls:
                    return msg.content or ""
                messages.append(msg)
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if tc.function.name == "read_file_by_name_schema":
                        result = self.read_file_by_name(args.get("title", ""))
                    else:                            # write_to_context_schema
                        self.write_to_context(args.get("text", ""))
                        result = "сохранено"
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": result})
            return (msg.content if msg else "") or ""
        except Cancelled:
            raise
        except Exception as e:
            raise RuntimeError(f"Ошибка вызова LLM: {e}") from e

    def rewrite_query(self, question, cancel=None):
        """модель переформулирует вопрос юзера в компактный поисковый запрос»"""
        return self._chat(REWRITE_PROMPT, question,
                          use_memory=False, history=self.history, cancel=cancel).strip()

    def retrieve(self, query, k=None, max_dist=None):
        """k ближайших чанков; при max_dist отсекает далёкие"""
        k = k or self.k
        vec = self.ingestor.compress(self.ingestor.embed_query(query))[0]
        hits = db.search_chunks(vec, k=k)
        if max_dist is not None:
            hits = [h for h in hits if h[-1] <= max_dist]
        return hits

    @staticmethod
    def build_context(hits):
        """Блоки найденных документов для промпта"""
        blocks = []
        for doc_id, title, chunk_no, text, cs, ce, dist in hits:
            blocks.append(f"[{title or doc_id} · фрагмент {chunk_no}]\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    def read_context():
        try:
            return CONTEXT_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def generate(self, question, hits, cancel=None):
        """Финальный ответ модели"""
        docs = self.build_context(hits) if hits else "(релевантных документов не найдено)"
        parts = []
        memory = self.read_context()
        if memory:
            parts.append(f"Что известно о пользователе и прошлом диалоге (эту информацию не нужно записывать в файл с контекстом):\n{memory}")
        parts.append(f"Контекст из документов:\n{docs}")
        parts.append(f"Вопрос: {question}")
        return self._chat(SYSTEM_PROMPT, "\n\n".join(parts), history=self.history, cancel=cancel)

    @staticmethod
    def _format_sources(hits):
        return [
            {"doc_id": did, "title": t, "chunk_no": cn, "text": txt,
             "start": cs, "end": ce, "dist": round(float(d), 4)}
            for (did, t, cn, txt, cs, ce, d) in hits
        ]

    def answer(self, question, k=None, cancel=None):
        """Возвращает {answer, query, sources}. cancel — Event: если выставлен,
        прерываемся на ближайшей точке и НЕ трогаем историю (Cancelled)"""
        self._ck(cancel)
        search_query = self.rewrite_query(question, cancel=cancel)
        self._ck(cancel)
        hits = self.retrieve(search_query, k=k, max_dist=self.threshold)
        self._ck(cancel)
        ans = self.generate(question, hits, cancel=cancel)
        self._ck(cancel)
        self._remember(question, ans)
        return {
            "answer": ans,
            "query": search_query,
            "sources": self._format_sources(hits),
        }


# if __name__ == "__main__":
#     agent = Agent()
#     q = "Слушай, а я вот давно хотел разобраться — сколько же лун крутится вокруг того гиганта Юпитера?"
#     print("Исходный вопрос:", q)

#     sq = agent.rewrite_query(q)
#     print("Поисковый запрос:", sq)

#     hits = agent.retrieve(sq, max_dist=agent.threshold)
#     print(f"\nОтобрано чанков (dist <= {agent.threshold}): {len(hits)}")
#     for doc_id, title, chunk_no, text, cs, ce, dist in hits:
#         print(f"  [{title} #{chunk_no}] dist={dist:.4f} :: {text[:60]}...")

#     print("\nОтвет:")
#     print(agent.answer(q)["answer"])
