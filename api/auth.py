"""api/auth.py — JWT utilities."""

from __future__ import annotations
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

SECRET_KEY = os.getenv("JWT_SECRET", "change-this-in-production-use-long-random-string")
ALGORITHM = "HS256"


@dataclass
class TokenData:
    username: str
    role: str


def create_token(data: dict, expires_delta: timedelta = None) -> str:
    try:
        from jose import jwt
    except ImportError:
        raise ImportError("python-jose not installed. Run: pip install python-jose[cryptography]")

    payload = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=8))
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[TokenData]:
    try:
        from jose import jwt, JWTError
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(
            username=payload.get("sub", ""),
            role=payload.get("role", "viewer"),
        )
    except Exception:
        return None
