"""Health check router for the RetentionAI API."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple health status payload for the API."""
    return {
        "status": "healthy",
        "api": "RetentionAI",
        "version": "1.0",
    }
