"""
JWT authentication utilities for ChickenCare AI.
Handles password hashing and token creation/validation.
"""

import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# Hard error at import time if JWT_SECRET is missing — intentional fail-fast.
SECRET_KEY = os.environ["JWT_SECRET"]
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24h default

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the hashed password."""
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: int, username: str) -> str:
    """
    Create a signed JWT for the given user.
    Payload: {"sub": str(user_id), "username": username, "exp": ...}
    """
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT. Returns the payload dict.
    Raises jose.JWTError if the token is invalid or expired.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
