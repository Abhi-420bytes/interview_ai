"""Session storage.

In-memory with JSON write-through: a process restart loses nothing, and the
files are readable, which matters while the engine is still being tuned. Swap
`SessionStore` for Redis when there's more than one worker.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import get_settings
from app.models import Session

log = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, directory: Path | None = None):
        self.dir = directory or Path(get_settings().data_dir) / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}

    def put(self, session: Session) -> None:
        self._cache[session.id] = session
        (self.dir / f"{session.id}.json").write_text(session.model_dump_json(indent=2))

    def get(self, session_id: str) -> Session | None:
        if session_id in self._cache:
            return self._cache[session_id]
        path = self.dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            session = Session.model_validate_json(path.read_text())
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("unreadable session %s: %s", path, e)
            return None
        self._cache[session_id] = session
        return session

    def delete(self, session_id: str) -> None:
        self._cache.pop(session_id, None)
        (self.dir / f"{session_id}.json").unlink(missing_ok=True)

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))


_store: SessionStore | None = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
