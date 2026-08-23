"""
Auth + RBAC for the Sovereign Workbench.

Design goals:
- Stdlib-only crypto. `hashlib.scrypt` for password hashing, `hmac` for
  signed session cookies. No new heavy deps (bcrypt/itsdangerous/pyjwt).
- File-backed user store at data/users.yaml. Never commit that file.
- Bootstrap admin from env vars on first startup; refuse to start with
  an empty user store, so the app cannot be silently open to the world.
- The LLM is never the security boundary. This module produces a
  `User` on every request; callers use `permits_doc()` to gate retrieval.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fastapi import Cookie, HTTPException, Request

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
USERS_PATH = DATA_DIR / "users.yaml"
SECRET_PATH = DATA_DIR / "session_secret"

SESSION_COOKIE = "sac_session"
SESSION_TTL_S = 12 * 3600

ROLES = ("admin", "manager", "employee")
DEPARTMENTS = (
    "hr", "operations", "maintenance", "safety",
    "finance", "it", "engineering", "general",
)
ACCESS_LEVELS = ("all", "department", "admin")  # doc.access_level values

# ────────────────────────────────────────────────────────────── password


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    n, r, p = 2 ** 14, 8, 1
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = _b64d(salt_b64)
        expected = _b64d(hash_b64)
        computed = hashlib.scrypt(
            password.encode("utf-8"), salt=salt,
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
        )
        return hmac.compare_digest(computed, expected)
    except Exception:
        return False


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


# ─────────────────────────────────────────────────────────────── user model


@dataclass
class User:
    email: str
    name: str
    role: str
    department: str
    password_hash: str
    disabled: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


# ─────────────────────────────────────────────────────────────── user store


class UserStore:
    def __init__(self, path: Path = USERS_PATH) -> None:
        self.path = path
        self._users: dict[str, User] = {}
        self.reload()

    def reload(self) -> None:
        if not self.path.exists():
            self._users = {}
            return
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        out: dict[str, User] = {}
        for entry in raw.get("users", []):
            u = User(
                email=str(entry["email"]).lower(),
                name=entry.get("name", entry["email"]),
                role=entry.get("role", "employee"),
                department=entry.get("department", "general"),
                password_hash=entry["password_hash"],
                disabled=bool(entry.get("disabled", False)),
            )
            out[u.email] = u
        self._users = out

    def all(self) -> list[User]:
        return list(self._users.values())

    def get(self, email: str) -> User | None:
        return self._users.get((email or "").lower())

    def authenticate(self, email: str, password: str) -> User | None:
        u = self.get(email)
        if not u or u.disabled:
            return None
        if not verify_password(password, u.password_hash):
            return None
        return u

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {"users": [{
            "email": u.email, "name": u.name, "role": u.role,
            "department": u.department, "password_hash": u.password_hash,
            "disabled": u.disabled,
        } for u in self._users.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        tmp.replace(self.path)

    def upsert(self, user: User) -> None:
        self._users[user.email.lower()] = user
        self.save()

    def bootstrap_admin_from_env(self) -> User | None:
        """
        Create the initial admin from env vars if the user store has no admin.
        Returns the created admin, or None if not applicable.
        """
        if any(u.role == "admin" and not u.disabled for u in self._users.values()):
            print(f"[auth] bootstrap skipped: admin already exists in {self.path}")
            return None
        email_raw = os.environ.get("BOOTSTRAP_ADMIN_EMAIL", "")
        password = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD", "")
        email = email_raw.strip().lower()
        name = os.environ.get("BOOTSTRAP_ADMIN_NAME", "").strip() or email
        print(
            f"[auth] bootstrap check: email={email!r} "
            f"password_len={len(password)} name={name!r}"
        )
        if not email or not password:
            print("[auth] bootstrap aborted: email or password empty")
            return None
        try:
            admin = User(
                email=email, name=name, role="admin", department="general",
                password_hash=hash_password(password),
            )
            self.upsert(admin)
        except Exception as exc:
            print(f"[auth] bootstrap FAILED writing users file: {exc!r}")
            raise
        print(f"[auth] bootstrap wrote {self.path} with admin {email}")
        return admin


# ─────────────────────────────────────────────────────────────── session


def _load_secret() -> bytes:
    override = os.environ.get("SESSION_SECRET")
    if override:
        return override.encode("utf-8")
    if SECRET_PATH.exists():
        return SECRET_PATH.read_bytes()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    SECRET_PATH.write_bytes(secret)
    return secret


_SECRET = _load_secret()


def issue_cookie(email: str) -> str:
    payload = f"{email.lower()}|{int(time.time())}"
    sig = hmac.new(_SECRET, payload.encode("utf-8"), hashlib.sha256).digest()
    return f"{_b64e(payload.encode('utf-8'))}.{_b64e(sig)}"


def verify_cookie(cookie: str) -> str | None:
    """Return the email if the cookie is valid and not expired."""
    if not cookie or "." not in cookie:
        return None
    try:
        payload_b64, sig_b64 = cookie.split(".", 1)
        payload = _b64d(payload_b64)
        expected_sig = hmac.new(_SECRET, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, _b64d(sig_b64)):
            return None
        email, issued = payload.decode("utf-8").split("|", 1)
        if int(time.time()) - int(issued) > SESSION_TTL_S:
            return None
        return email
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────── permissions


def permits_doc(user: User, doc_meta: dict[str, Any]) -> bool:
    """
    Return True if `user` may see the document described by `doc_meta`.

    doc_meta expects keys: department, access_level.
    Defaults for legacy rows: department='general', access_level='all'.
    """
    dept = (doc_meta.get("department") or "general").lower()
    level = (doc_meta.get("access_level") or "all").lower()

    if user.is_admin:
        return True
    if level == "admin":
        return False
    if level == "all":
        return True
    # access_level == 'department'
    return dept == "general" or dept == (user.department or "general").lower()


def allowed_doc_ids(user: User, all_docs: list[dict[str, Any]]) -> list[str]:
    return [d["doc_id"] for d in all_docs if permits_doc(user, d)]


# ─────────────────────────────────────────────────────────────── FastAPI dep


class AuthContext:
    """Bound at app startup so FastAPI deps can find the user store."""
    store: UserStore | None = None


def _resolve_user(cookie_val: str | None) -> User:
    if AuthContext.store is None:
        raise HTTPException(500, "auth not initialized")
    email = verify_cookie(cookie_val or "")
    if not email:
        raise HTTPException(401, "not authenticated")
    user = AuthContext.store.get(email)
    if not user or user.disabled:
        raise HTTPException(401, "user no longer valid")
    return user


def current_user(request: Request) -> User:
    """FastAPI dependency: pulls the session cookie and returns the User."""
    return _resolve_user(request.cookies.get(SESSION_COOKIE))


def require_admin(request: Request) -> User:
    u = current_user(request)
    if not u.is_admin:
        raise HTTPException(403, "admin only")
    return u
