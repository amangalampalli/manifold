import { create } from "zustand";
import { fetchGraph, fetchRollout, fetchRuns, fetchTelemetry } from "./api";
import type { GraphPayload, Policy, RolloutPayload, RunSummary, TelemetryLine } from "./types";

const MESSY_PRESET = {
  lambdaSheaf: 0.58,
  pIn: 0.015,
  pOut: 0.058
};

interface ShowcaseState {
  loading: boolean;
  error: string | null;
  runs: RunSummary[];
  run: RunSummary | null;
  runIdx: number;
  graph: GraphPayload | null;
  neural: RolloutPayload | null;
  greedy: RolloutPayload | null;
  chatgpt: RolloutPayload | null;
  telemetry: TelemetryLine[];
  step: number;
  playing: boolean;
  selectedPolicy: Exclude<Policy, "random">;
  lambdaSheaf: number;
  pIn: number;
  pOut: number;
  load: () => Promise<void>;
  loadRun: (runId: string, runIdx?: number) => Promise<void>;
  setStep: (step: number) => void;
  setPlaying: (playing: boolean) => void;
  setSelectedPolicy: (policy: Exclude<Policy, "random">) => void;
  setLambdaSheaf: (value: number) => void;
  setPIn: (value: number) => void;
  setPOut: (value: number) => void;
}

export const useShowcaseStore = create<ShowcaseState>((set, get) => ({
  loading: true,
  error: null,
  runs: [],
  run: null,
  runIdx: 0,
  graph: null,
  neural: null,
  greedy: null,
  chatgpt: null,
  telemetry: [],
  step: 0,
  playing: false,
  selectedPolicy: "neural",
  lambdaSheaf: MESSY_PRESET.lambdaSheaf,
  pIn: MESSY_PRESET.pIn,
  pOut: MESSY_PRESET.pOut,
  load: async () => {
    set({ loading: true, error: null });
    try {
      const { runs } = await fetchRuns();
      const run = runs[0];
      if (!run) {
        throw new Error("No closed-loop run artifacts found");
      }
      set({ runs });
      await get().loadRun(run.id, run.defaultRunIdx);
    } catch (error) {
      set({ loading: false, error: error instanceof Error ? error.message : String(error) });
    }
  },
  loadRun: async (runId, runIdx) => {
    set({ loading: true, error: null, playing: false, step: 0 });
    try {
      const runs = get().runs.length ? get().runs : (await fetchRuns()).runs;
      const run = runs.find((candidate) => candidate.id === runId);
      if (!run) {
        throw new Error(`Closed-loop run not found: ${runId}`);
      }
      const selectedRunIdx = runIdx ?? run.defaultRunIdx;
      const [graph, neural, greedy, chatgpt, telemetry] = await Promise.all([
        fetchGraph(run.id),
        fetchRollout(run.id, "neural", selectedRunIdx),
        fetchRollout(run.id, "greedy", selectedRunIdx),
        fetchRollout(run.id, "chatgpt", selectedRunIdx).catch(() => null),
        fetchTelemetry(run.id, selectedRunIdx)
      ]);
      set({
        loading: false,
        runs,
        run,
        runIdx: selectedRunIdx,
        graph,
        neural,
        greedy,
        chatgpt,
        telemetry: telemetry.lines,
        lambdaSheaf: MESSY_PRESET.lambdaSheaf,
        pIn: MESSY_PRESET.pIn,
        pOut: MESSY_PRESET.pOut
      });
    } catch (error) {
      set({ loading: false, error: error instanceof Error ? error.message : String(error) });
    }
  },
  setStep: (step) => set({ step }),
  setPlaying: (playing) => set({ playing }),
  setSelectedPolicy: (selectedPolicy) => set({ selectedPolicy }),
  setLambdaSheaf: (lambdaSheaf) => set({ lambdaSheaf }),
  setPIn: (pIn) => set({ pIn }),
  setPOut: (pOut) => set({ pOut })
}));

export function selectMaxStep(state: ShowcaseState) {
  return Math.max(0, (state.neural?.times.length ?? 1) - 1);
}
