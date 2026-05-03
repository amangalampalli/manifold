"""FastAPI application for the closed-loop showcase."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from manifold.api.artifacts import ArtifactNotFoundError, ClosedLoopArtifactStore


def create_app(root: str | Path | None = None) -> FastAPI:
    store = ClosedLoopArtifactStore(root)
    app = FastAPI(title="Manifold Closed-Loop Showcase API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/runs")
    def list_runs() -> dict[str, object]:
        return {"runs": store.list_runs()}

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict[str, object]:
        return store.summary(run_id)

    @app.get("/api/runs/{run_id}/graph")
    def run_graph(run_id: str) -> dict[str, object]:
        return store.graph(run_id)

    @app.get("/api/runs/{run_id}/rollout")
    def run_rollout(
        run_id: str,
        policy: str = Query(pattern="^(random|greedy|neural|chatgpt)$"),
        run_idx: int = 0,
    ) -> dict[str, object]:
        return store.rollout(run_id, policy, run_idx)

    @app.get("/api/runs/{run_id}/telemetry")
    def run_telemetry(run_id: str, run_idx: int = 0) -> dict[str, object]:
        return store.telemetry(run_id, run_idx)

    @app.exception_handler(ArtifactNotFoundError)
    def artifact_not_found(_: object, exc: ArtifactNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    dist_dir = Path("frontend/dist")
    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("manifold.api.server:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
