# API Design

## Planned Endpoints
- POST /predict
- GET /health
- GET /model-info

## Request and Response Schema Descriptions
- POST /predict: accepts a customer feature payload and returns a prediction plus confidence score.
- GET /health: returns the service status and readiness information.
- GET /model-info: returns the currently registered model metadata and artifact details.

## Error Handling Strategy
- Return structured error responses for invalid input, missing artifacts, or unavailable model state.
- Use clear status codes such as 400 for bad requests, 404 for missing resources, and 500 for unexpected failures.
