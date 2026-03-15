"""api/rbac.py — Role-based access control."""

from __future__ import annotations
from enum import Enum
from fastapi import HTTPException, status


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


ROLE_HIERARCHY = {Role.ADMIN: 3, Role.ANALYST: 2, Role.VIEWER: 1}


def require_role(user, required_roles: list[Role]):
    user_level = ROLE_HIERARCHY.get(Role(user.role), 0)
    required_level = max(ROLE_HIERARCHY.get(r, 0) for r in required_roles)
    if user_level < required_level:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{user.role}' insufficient. Required: {[r.value for r in required_roles]}",
        )
