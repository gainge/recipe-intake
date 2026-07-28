"""Bearer token auth.

One user, one token. No user table, no sessions, no signup — those would be
machinery guarding nothing.

The token is not really protecting recipe data; it is protecting a paid API
behind an open endpoint, which is the actual cost risk (DESIGN.md §7). The rate
limit and spend ceiling are the backstops for when it leaks anyway.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings

UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="missing or invalid bearer token",
    headers={"WWW-Authenticate": "Bearer"},
)


def require_token(request: Request, settings: Settings = Depends(get_settings)) -> None:
    if not settings.api_token:
        # Refuse rather than allow. A blank token in the environment is a
        # misconfiguration, and treating it as "no auth required" would expose
        # the Anthropic key's spending power to the internet.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server has no API_TOKEN configured",
        )

    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise UNAUTHORIZED
    if not secrets.compare_digest(presented, settings.api_token):
        raise UNAUTHORIZED
