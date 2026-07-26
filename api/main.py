"""FastAPI application entry point for the RetentionAI API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.health import router as health_router
from api.routers.predict import router as predict_router

app = FastAPI(
    title="RetentionAI API",
    version="1.0",
    description="Production-ready Customer Churn Prediction API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
