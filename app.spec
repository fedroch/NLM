# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller-спека для десктоп-приложения (собирать НА Windows).

Режим — onedir (не onefile): бандл ~4+ ГБ (torch + модель + postgres), и
onefile каждый запуск распаковывал бы всё во временную папку — очень медленно.

Сборка:  pyinstaller app.spec
Результат:  dist/Notebook/Notebook.exe
"""
import os
import sys
from PyInstaller.utils.hooks import collect_all

# Qt нужен только под Linux: там pywebview рисует окно через QtWebEngine. Под
# Windows окно даёт системный WebView2, и Qt — мёртвый груз в сотни мегабайт,
# поэтому там его выкидываем.
GUI_EXCLUDES = ["PyQt5", "PyQtWebEngine", "qtpy"] if sys.platform == "win32" else []

datas, binaries, hiddenimports = [], [], []

# Тяжёлые пакеты со скрытыми импортами и файлами данных — тянем целиком.
for pkg in [
    "torch", "sentence_transformers", "transformers", "tokenizers",
    "huggingface_hub", "safetensors", "sklearn", "scipy", "pandas",
    "tqdm", "numpy", "pgvector", "psycopg", "sqlalchemy",
    "openai", "pydantic", "pydantic_core", "webview", "cryptography",
    # markitdown и его извлекатели форматов. onnxruntime — нативная либа,
    # magika несёт ONNX-модель как data-файл, pdfminer — таблицы cmap.
    "markitdown", "magika", "onnxruntime", "pdfminer", "pdfplumber",
    "mammoth", "pptx", "openpyxl", "markdownify", "bs4",
    "defusedxml", "charset_normalizer",
]:
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# pgserver несёт бинарники Postgres + pgvector в pginstall/ — кладём всё дерево
# как есть (collect_data_files их пропускает, т.к. .exe/.dll считаются бинарями).
import pgserver
pg_root = os.path.dirname(pgserver.__file__)
datas += [(os.path.join(pg_root, "pginstall"), "pgserver/pginstall")]
datas += [(os.path.join(pg_root, p), "pgserver")
          for p in os.listdir(pg_root) if p.endswith(".py")]

# Файлы приложения.
datas += [
    ("web", "web"),                       # фронтенд
    ("compresser_128.pt", "."),           # веса компрессора
]
# Зашифрованный дефолтный ключ LLM (см. secret.py). Бандлим, если создан.
if os.path.exists("default_key.enc"):
    datas += [("default_key.enc", ".")]
# .env в бандл кладём БЕЗ секретов: только BASE_URL/MODEL. Даже если в вашем .env
# есть API_KEY (для локальной разработки), в сборку он НЕ попадёт — дефолтный ключ
# распространяется только через зашифрованный default_key.enc. Санитизируем сами,
# чтобы не полагаться на «не забудь убрать ключ вручную».
if os.path.exists(".env"):
    import tempfile
    _safe = [ln.rstrip("\n") for ln in open(".env", encoding="utf-8")
             if ln.split("=", 1)[0].strip().upper() in ("BASE_URL", "MODEL")]
    if _safe:
        _envdir = os.path.join(tempfile.gettempdir(), "nlm_build_env")
        os.makedirs(_envdir, exist_ok=True)
        with open(os.path.join(_envdir, ".env"), "w", encoding="utf-8") as _f:
            _f.write("\n".join(_safe) + "\n")
        datas += [(os.path.join(_envdir, ".env"), ".")]

# Забандленная модель эмбеддера (офлайн). Кладём ПЛОСКУЮ копию с реальными
# файлами, а не HF-кэш: в кэше snapshots/ — это симлинки на blobs/, и PyInstaller
# упаковывает их как пустые 0-байтные файлы (модель потом падает на пустом JSON).
# Скачать заранее без симлинков:  hf download Qwen/Qwen3-Embedding-0.6B --local-dir model_qwen
model_dir = "model_qwen"
if os.path.isdir(model_dir):
    datas += [(model_dir, "model_qwen")]
else:
    raise SystemExit(
        "Модель не найдена: ./model_qwen\n"
        "Скачайте её плоской копией на сборочной машине перед сборкой:\n"
        "  hf download Qwen/Qwen3-Embedding-0.6B --local-dir model_qwen"
    )

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # optuna больше не нужен: percp вынесен в model_class.py, приложение не тянет
    # dim_reducer. datasets НЕ исключаем — sentence_transformers импортирует его
    # лениво при загрузке модели, иначе падает с "No module named datasets".
    excludes=["optuna", "tkinter", "matplotlib", *GUI_EXCLUDES],
    # transformers лениво ищет свои подмодули как файлы на диске. noarchive
    # раскладывает чистый Python в _internal/ россыпью .pyc, а не в архив,
    # иначе на рантайме "не удаётся найти путь ...transformers\models\__init__.pyc".
    noarchive=True,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Notebook",
    console=False,            # оконное приложение, без консоли
    disable_windowed_traceback=False,
    icon="icon.ico",          # вшивается в .exe → иконка едет с файлом на любой ПК
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="Notebook",
)
