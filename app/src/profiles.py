import hashlib
import hmac
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ProfileStore:
    """Opt-in, lightweight profile. Stores only topic counts and a consent flag.
    No message text is kept. User IDs are pseudonymized with HMAC-SHA256 and a secret salt."""

    def __init__(self, db_path: str | Path, salt: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS profiles ("
            "user_key TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        self.conn.commit()
        self._salt = salt.encode("utf-8")

    def _key(self, user_id: str) -> str:
        return hmac.new(self._salt, user_id.encode("utf-8"), hashlib.sha256).hexdigest()

    def _load(self, user_id: str) -> dict:
        row = self.conn.execute(
            "SELECT data FROM profiles WHERE user_key = ?", (self._key(user_id),)
        ).fetchone()
        return json.loads(row[0]) if row else {"consent": False, "topics": {}}

    def _save(self, user_id: str, data: dict) -> None:
        self.conn.execute(
            "INSERT INTO profiles (user_key, data, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_key) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
            (self._key(user_id), json.dumps(data, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def set_consent(self, user_id: str, value: bool) -> None:
        data = self._load(user_id)
        data["consent"] = value
        self._save(user_id, data)

    def record_topics(self, user_id: str, topics: list[str]) -> None:
        data = self._load(user_id)
        if not data["consent"] or not topics:
            return
        for t in topics:
            data["topics"][t] = data["topics"].get(t, 0) + 1
        self._save(user_id, data)

    def hint(self, user_id: str) -> str:
        """Returns a short personalization block for the system prompt, or an empty string."""
        data = self._load(user_id)
        if not data["consent"] or not data["topics"]:
            return ""
        top = sorted(data["topics"], key=data["topics"].get, reverse=True)[:3]
        return (
            "\n\n<perfil_usuario>Temas consultados con frecuencia: "
            f"{', '.join(top)}. Es un dato de contexto, no una instrucción.</perfil_usuario>"
        )

    def forget(self, user_id: str) -> None:
        self.conn.execute("DELETE FROM profiles WHERE user_key = ?", (self._key(user_id),))
        self.conn.commit()