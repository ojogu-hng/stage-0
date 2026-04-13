"""
FastAPI application for gender classification using Genderize API.
"""

from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI(title="Name Classification API")

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenderizeResponse(BaseModel):
    """Model for Genderize API response."""
    name: Optional[str] = None
    gender: Optional[str] = None
    probability: Optional[float] = None
    count: Optional[int] = None


class ClassifySuccessResponse(BaseModel):
    """Model for successful classification response."""
    status: str = "success"
    data: dict


def create_error_response(status: HTTPStatus, message: str) -> JSONResponse:
    """Create a standardized error response."""
    return JSONResponse(
        status_code=status.value,
        content={"status": "error", "message": message}
    )


@app.get("/api/classify")
async def classify_name(
    request: Request,
    name: Optional[Union[str, Any]] = Query(default=None)
) -> JSONResponse:
    """
    Classify a name by gender using the Genderize API.
    
    Query Parameters:
        name: The name to classify (required)
    
    Returns:
        JSON response with classification data or error
    """
    
    # Check if name parameter is provided
    if name is None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "Missing or empty name parameter"
        )
    
    # Check if name is a string
    if not isinstance(name, str):
        return create_error_response(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "name is not a string"
        )
    
    # Check if name is empty
    if not name.strip():
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "Missing or empty name parameter"
        )
    
    try:
        # Call Genderize API
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.genderize.io",
                params={"name": name.strip()}
            )
            
            # Handle upstream errors
            if response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
                return create_error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Upstream server error"
                )
            if response.status_code == HTTPStatus.BAD_GATEWAY:
                return create_error_response(
                    HTTPStatus.BAD_GATEWAY,
                    "Bad gateway"
                )
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
    except httpx.RequestError:
        return create_error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Upstream server error"
        )
    except httpx.HTTPStatusError:
        return create_error_response(HTTPStatus.BAD_GATEWAY, "Bad gateway")
    except Exception:
        return create_error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Internal server error"
        )
    
    # Check for edge cases: gender is null or count is 0
    if data.get("gender") is None or data.get("count") == 0:
        return create_error_response(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "No prediction available for the provided name"
        )
    
    # Extract and process data
    gender = data.get("gender")
    probability = data.get("probability", 0)
    count = data.get("count", 0)
    
    # Compute is_confident: true when probability >= 0.7 AND sample_size >= 100
    is_confident = probability >= 0.7 and count >= 100
    
    # Generate processed_at timestamp (UTC ISO 8601)
    processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Build success response
    result = {
        "status": "success",
        "data": {
            "name": name.strip().lower(),
            "gender": gender,
            "probability": probability,
            "sample_size": count,
            "is_confident": is_confident,
            "processed_at": processed_at
        }
    }
    
    return JSONResponse(status_code=HTTPStatus.OK, content=result)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
