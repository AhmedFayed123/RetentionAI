"""Entrypoint for training the RetentionAI models.

Runs the training pipeline and reports status. Designed to be used in
production-like environments: logs progress, handles exceptions, and
returns appropriate exit codes.
"""
from __future__ import annotations

import logging
import sys
import traceback
from typing import Tuple

from src.pipelines.training_pipeline import run_training_pipeline


def setup_logging() -> None:
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    """Run the training pipeline and return an exit code.

    Returns
    -------
    int
        Exit code (0 on success, non-zero on failure).
    """
    logger = logging.getLogger("main")
    setup_logging()

    logger.info("Starting training pipeline")
    print("Training pipeline: STARTED")

    try:
        best_model, best_name, comparison_df = run_training_pipeline()
        logger.info("Training pipeline completed successfully")
        print("Training pipeline: COMPLETED")
        print(f"Selected best model: {best_name}")
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level exception handler for CLI
        logger.error("Training pipeline failed: %s", exc)
        traceback.print_exc()
        print("Training pipeline: FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
