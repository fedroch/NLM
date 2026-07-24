import base64
import os

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_PARTS = ("dim", "-reduc", "er", "-notebook", "-2026")
_SALT = b"dr::notebook::salt::v1"
_ENC_FILE = "default_key.enc"


def _fernet():
    passphrase = "".join(_PARTS).encode()
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=_SALT,
                     iterations=200_000)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(passphrase)))


def encrypt(plaintext):
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(token):
    return _fernet().decrypt(token.encode()).decode()


def load_default_key(here="."):
    """Дефолтный ключ из зашифрованного файла рядом с приложением. '' если нет."""
    path = os.path.join(here, _ENC_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            token = f.read().strip()
        return decrypt(token) if token else ""
    except (OSError, InvalidToken):
        return ""


if __name__ == "__main__":
    import getpass

    key = getpass.getpass("Вставьте API-ключ (ввод скрыт): ").strip()
    if not key:
        raise SystemExit("пусто, ничего не записано")
    with open(_ENC_FILE, "w", encoding="utf-8") as f:
        f.write(encrypt(key))
    ok = load_default_key() == key
    print(f"Записано в {_ENC_FILE}. Проверка расшифровки: {'OK' if ok else 'FAIL'}")
