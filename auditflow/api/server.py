# ============================================================
# auditflow/api/server.py
# FastAPI Application for AuditFlow
# ============================================================

from fastapi import FastAPI, BackgroundTasks, HTTPException
from typing import Any
from auditflow.api.schemas import PipelineRunRequest
from auditflow.core.config import (
    PipelineConfig,
    DataConfig,
    CleaningConfig,
    FeatureConfig,
    ModelConfig,
    ReportConfig,
)
import uuid
from auditflow.pipeline.runner import Pipeline
from auditflow.core.cache import RedisCache
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import logging

app = FastAPI(
    title="AuditFlow API",
    description="REST API for the AuditFlow Data Analysis Platform",
    version="0.1.0",
)
FastAPIInstrumentor.instrument_app(app)
logger = logging.getLogger("auditflow.api")
cache = RedisCache()


@app.get("/health")
def health_check() -> Any:
    """Health check endpoint to verify service is running."""
    return {"status": "ok", "service": "AuditFlow API"}


def run_pipeline_task(req: PipelineRunRequest, job_id: str) -> None:
    """Background task to convert pydantic models and run the pipeline."""
    config = PipelineConfig(
        data=DataConfig(**req.data.model_dump()),
        cleaning=CleaningConfig(**req.cleaning.model_dump()),
        features=FeatureConfig(**req.features.model_dump()),
        model=ModelConfig(**req.model.model_dump()),
        report=ReportConfig(**req.report.model_dump()),
    )
    pipeline = Pipeline(config, job_id=job_id)
    try:
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


@app.post("/v1/pipeline/run")
def trigger_pipeline(req: PipelineRunRequest, background_tasks: BackgroundTasks) -> Any:
    """
    Triggers a pipeline run in the background.
    """
    job_id = str(uuid.uuid4())
    cache.set_pipeline_status(job_id, "PENDING", "Pipeline enqueued")
    background_tasks.add_task(run_pipeline_task, req, job_id)
    return {"status": "accepted", "job_id": job_id, "message": "Pipeline run triggered in the background"}

@app.get("/v1/pipeline/{job_id}/status")
def get_pipeline_status(job_id: str) -> Any:
    """
    Get the status of a pipeline run.
    """
    status = cache.get_pipeline_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Pipeline job not found")
    return status
