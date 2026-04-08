# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Customer Support Environment.

This module creates an HTTP server that exposes the SupportTicketEnvironment
over HTTP and WebSocket endpoints, compatible with OpenEnv clients and HF Spaces.

Endpoints:
    - POST /reset: Reset the environment and start new episode
    - POST /step: Execute an action in the environment
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /health: Health check endpoint
    - WS /ws: WebSocket endpoint for persistent sessions

Features:
    - Full OpenEnv compatibility
    - WebSocket support for persistent sessions
    - Health check for monitoring
    - CORS enabled for web clients
    - HuggingFace Spaces compatible
    - Multi-instance support

Usage:
    # Development with auto-reload:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Docker:
    docker build -t support-env .
    docker run -p 8000:8000 support-env

    # Direct execution:
    python -m server.app
    python -m server.app --port 8001
"""

import logging
import sys
import os
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv package is required. Install with: pip install openenv"
    ) from e

try:
    from models import SupportAction, SupportObservation
    from support_env import SupportTicketEnvironment
except ImportError as e:
    raise ImportError(f"Failed to import models or environment: {e}") from e

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create the OpenEnv HTTP server app
app = create_app(
    SupportTicketEnvironment,
    SupportAction,
    SupportObservation,
    env_name="support_ticket_environment",
    max_concurrent_envs=1,  # Single concurrent environment for now
)

logger.info("Customer Support Environment app initialized")


# Explicit health endpoint for HF Spaces compatibility
@app.get("/health", status_code=200)
async def health():
    """Health check endpoint returns 200."""
    return {"status": "healthy", "service": "support_ticket_environment"}


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution.

    Supports multiple invocation methods:
        - python -m server.app
        - python -m server.app --port 8002
        - uv run --project . server
        - uv run --project . server -- --port 8002
        - PORT=8002 python -m server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8001, or from PORT env var)

    For production, use uvicorn with multiple workers:
        uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
    """
    import sys
    import argparse
    
    # Check environment variable first
    default_port = int(os.getenv("PORT", port))
    
    # Parse command-line arguments if provided
    parser = argparse.ArgumentParser(
        description="Customer Support Environment Server",
        add_help=False  # Avoid conflict with existing args
    )
    parser.add_argument("--host", type=str, default=host, help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port number (default: {default_port})")
    args, _ = parser.parse_known_args()
    
    import uvicorn

    logger.info(f"Starting Customer Support Environment server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
