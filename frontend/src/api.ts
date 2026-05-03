import type { GraphPayload, Policy, RolloutPayload, RunSummary, TelemetryLine } from "./types";

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchRuns() {
  return getJson<{ runs: RunSummary[] }>("/api/runs");
}

export async function fetchGraph(runId: string) {
  return getJson<GraphPayload>(`/api/runs/${runId}/graph`);
}

export async function fetchRollout(runId: string, policy: Policy, runIdx = 0) {
  const params = new URLSearchParams({ policy, run_idx: String(runIdx) });
  return getJson<RolloutPayload>(`/api/runs/${runId}/rollout?${params.toString()}`);
}

export async function fetchTelemetry(runId: string, runIdx = 0) {
  return getJson<{ lines: TelemetryLine[] }>(`/api/runs/${runId}/telemetry?run_idx=${runIdx}`);
}

