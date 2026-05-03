import { ShieldAlert, Sparkles, TrendingDown } from "lucide-react";
import type { PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { useState } from "react";
import type { Policy, RolloutPayload, RunSummary } from "../types";

interface ReceiptsProps {
  run: RunSummary | null;
  neural: RolloutPayload | null;
  greedy: RolloutPayload | null;
  chatgpt: RolloutPayload | null;
  selectedPolicy: Exclude<Policy, "random">;
  step: number;
  setStep: (step: number) => void;
  setPlaying: (playing: boolean) => void;
}

export function Receipts({ run, neural, greedy, chatgpt, selectedPolicy, step, setStep, setPlaying }: ReceiptsProps) {
  const delta = run?.deltas.mean_error_auc_reduction_pct;
  const liveAdvantage = cumulativeAdvantage(greedy?.meanError, neural?.meanError, step);
  const currentRollout = selectedPolicy === "neural" ? neural : selectedPolicy === "chatgpt" ? chatgpt : greedy;
  const neuralCapture = capturedPerturbationMass(neural, step);
  const currentCapture = capturedPerturbationMass(currentRollout, step);
  const captureDelta = neuralCapture - currentCapture;
  const captureLabel = selectedPolicy === "neural" ? "neural captured" : `${selectedPolicy === "chatgpt" ? "gpt" : "greedy"} captured`;
  return (
    <section className="border-r border-white/10 bg-carbon-950/68 backdrop-blur-xl">
      <div className="border-b border-white/10 p-4 transition-all duration-300 hover:bg-white/[0.025]">
        <div className="font-mono text-[11px] uppercase tracking-[0.28em] text-white/45">
          closed-loop receipts
        </div>
        <div className="mt-2 text-4xl font-semibold text-white">
          {delta && delta > 0 ? `${delta.toFixed(1)}%` : "evidence"}
        </div>
        <div className="text-sm text-white/52">error mass removed versus greedy</div>
      </div>
      <div className="grid grid-rows-[1fr_auto]">
        <div className="border-b border-white/10 p-4 transition-all duration-300 hover:bg-white/[0.025]">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-white/55">
              <TrendingDown className="h-4 w-4 text-cyanOps" />
              cumulative error burden
            </div>
            <span className="font-mono text-xs text-cyanOps">
              live {Math.round(liveAdvantage)}%
            </span>
          </div>
          <div className="mt-4 border border-white/10 bg-white/[0.035] p-3 transition-all duration-300 hover:-translate-y-1 hover:border-cyanOps/25 hover:bg-cyanOps/[0.035] hover:shadow-[0_18px_50px_rgba(32,215,255,0.08)]">
            <AdvantageChart
              greedy={greedy?.meanError ?? []}
              neural={neural?.meanError ?? []}
              chatgpt={chatgpt?.meanError ?? []}
              step={step}
              setStep={setStep}
              setPlaying={setPlaying}
            />
          </div>
          <div className="mt-2 font-mono text-[10px] uppercase tracking-[0.18em] text-white/32">
            drag or click the burden curve to scrub the rollout
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2">
            <PolicyStat
              icon={<ShieldAlert className="h-4 w-4 text-amberOps" />}
              label="greedy auc"
              value={greedy?.meanErrorAuc.toFixed(3) ?? "--"}
            />
            <PolicyStat
              icon={<Sparkles className="h-4 w-4 text-violet-300" />}
              label="gpt auc"
              value={chatgpt?.meanErrorAuc.toFixed(3) ?? "--"}
            />
            <PolicyStat
              icon={<Sparkles className="h-4 w-4 text-cyanOps" />}
              label="sheaf auc"
              value={neural?.meanErrorAuc.toFixed(3) ?? "--"}
            />
          </div>
        </div>
        <div className="grid grid-cols-2 border-b border-white/10">
          <LiveMetric label={captureLabel} value={currentCapture ? `${Math.round(currentCapture)}%` : "--"} tone={selectedPolicy === "neural" ? "cyan" : "amber"} />
          <LiveMetric label="neural capture edge" value={`${captureDelta >= 0 ? "+" : ""}${Math.round(captureDelta)} pts`} tone="cyan" />
        </div>
      </div>
    </section>
  );
}

function AdvantageChart({
  greedy,
  neural,
  chatgpt,
  step,
  setStep,
  setPlaying
}: {
  greedy: number[];
  neural: number[];
  chatgpt: number[];
  step: number;
  setStep: (step: number) => void;
  setPlaying: (playing: boolean) => void;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [scrubbing, setScrubbing] = useState(false);
  const width = 242;
  const height = 210;
  const greedySeries = cumulativeSeries(greedy);
  const neuralSeries = cumulativeSeries(neural);
  const chatgptSeries = cumulativeSeries(chatgpt);
  const length = Math.min(greedySeries.length, neuralSeries.length, chatgptSeries.length || Number.POSITIVE_INFINITY);
  const max = Math.max(...greedySeries, ...neuralSeries, ...chatgptSeries, 1);
  const range = Math.max(max, 1e-6);
  const greedyPoints = greedySeries.slice(0, length).map((value, idx) => {
    const x = length <= 1 ? 0 : (idx / (length - 1)) * width;
    const y = height - (value / range) * height;
    return [x, y] as [number, number];
  });
  const neuralPoints = neuralSeries.slice(0, length).map((value, idx) => {
    const x = length <= 1 ? 0 : (idx / (length - 1)) * width;
    const y = height - (value / range) * height;
    return [x, y] as [number, number];
  });
  const chatgptPoints = chatgptSeries.slice(0, length).map((value, idx) => {
    const x = length <= 1 ? 0 : (idx / (length - 1)) * width;
    const y = height - (value / range) * height;
    return [x, y] as [number, number];
  });
  const gap = [...greedyPoints, ...[...neuralPoints].reverse()].map(([x, y]) => `${x},${y}`).join(" ");
  const markerIdx = Math.min(step, length - 1);
  const activeIdx = hoverIdx ?? markerIdx;
  const markerX = length <= 1 ? 0 : (markerIdx / (length - 1)) * width;
  const markerY = neuralPoints[markerIdx]?.[1] ?? height;
  const hoverX = length <= 1 ? 0 : (activeIdx / (length - 1)) * width;
  const hoverGreedy = greedySeries[activeIdx] ?? 0;
  const hoverNeural = neuralSeries[activeIdx] ?? 0;
  const hoverChatgpt = chatgptSeries[activeIdx] ?? 0;
  const hoverSaved = Math.max(0, hoverGreedy - hoverNeural);
  const hoverY = neuralPoints[activeIdx]?.[1] ?? height;
  const finalAdvantage = cumulativeAdvantage(greedy, neural, length - 1);
  const updateFromPointer = (event: ReactPointerEvent<SVGSVGElement>, commit: boolean) => {
    if (length <= 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * width;
    const idx = Math.min(length - 1, Math.max(0, Math.round((x / width) * (length - 1))));
    setHoverIdx(idx);
    if (commit) {
      setPlaying(false);
      setStep(idx);
    }
  };

  return (
    <svg
      className={`h-[230px] w-full overflow-visible ${scrubbing ? "cursor-grabbing" : "cursor-ew-resize"}`}
      viewBox={`0 0 ${width} ${height}`}
      onPointerLeave={() => {
        setHoverIdx(null);
        setScrubbing(false);
      }}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        setScrubbing(true);
        updateFromPointer(event, true);
      }}
      onPointerUp={(event) => {
        event.currentTarget.releasePointerCapture(event.pointerId);
        setScrubbing(false);
        updateFromPointer(event, true);
      }}
      onPointerCancel={() => setScrubbing(false)}
      onPointerMove={(event) => {
        updateFromPointer(event, scrubbing);
      }}
    >
      <defs>
        <linearGradient id="advantage-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="#20d7ff" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#20d7ff" stopOpacity="0.04" />
        </linearGradient>
      </defs>
      <line x1="0" x2={width} y1={height - 1} y2={height - 1} stroke="rgba(255,255,255,0.14)" />
      <polygon points={gap} fill="url(#advantage-fill)" />
      <polyline points={greedyPoints.map(([x, y]) => `${x},${y}`).join(" ")} fill="none" stroke="#ff9d2e" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" opacity="0.9" />
      {chatgptPoints.length ? (
        <polyline points={chatgptPoints.map(([x, y]) => `${x},${y}`).join(" ")} fill="none" stroke="#c084fc" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" opacity="0.88" />
      ) : null}
      <polyline points={neuralPoints.map(([x, y]) => `${x},${y}`).join(" ")} fill="none" stroke="#20d7ff" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" />
      <line x1={markerX} x2={markerX} y1="0" y2={height} stroke="rgba(255,255,255,0.32)" strokeDasharray="3 5" />
      <circle cx={markerX} cy={markerY} r="7" fill="#20d7ff" opacity="0.18">
        <animate attributeName="r" values="7;12;7" dur="1.5s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.18;0.04;0.18" dur="1.5s" repeatCount="indefinite" />
      </circle>
      <circle cx={markerX} cy={markerY} r="3.5" fill="#20d7ff" />
      {hoverIdx !== null ? (
        <g>
          <line x1={hoverX} x2={hoverX} y1="0" y2={height} stroke="#ffffff" strokeDasharray="2 4" opacity="0.42" />
          <circle cx={hoverX} cy={hoverY} r="4" fill="#ffffff" />
          <g transform={`translate(${Math.min(width - 108, Math.max(4, hoverX + 8))} ${Math.max(36, hoverY - 66)})`}>
            <rect width="104" height="64" fill="#030507" stroke="rgba(255,255,255,0.18)" />
            <text x="8" y="14" fill="#ffffff" fontFamily="monospace" fontSize="8" letterSpacing="1.5">STEP {activeIdx}</text>
            <text x="8" y="28" fill="#ff9d2e" fontFamily="monospace" fontSize="8" letterSpacing="1.3">G {hoverGreedy.toFixed(3)}</text>
            <text x="8" y="40" fill="#c084fc" fontFamily="monospace" fontSize="8" letterSpacing="1.3">GPT {hoverChatgpt.toFixed(3)}</text>
            <text x="8" y="52" fill="#20d7ff" fontFamily="monospace" fontSize="8" letterSpacing="1.3">ODE {hoverNeural.toFixed(3)}</text>
            <text x="62" y="52" fill="#57f287" fontFamily="monospace" fontSize="8" letterSpacing="1.3">Δ {hoverSaved.toFixed(3)}</text>
          </g>
        </g>
      ) : null}
      <text x="0" y="12" fill="#ff9d2e" fontFamily="monospace" fontSize="9" letterSpacing="2">GREEDY CUMULATIVE ERROR</text>
      <text x="0" y="28" fill="#c084fc" fontFamily="monospace" fontSize="9" letterSpacing="2">CHATGPT CUMULATIVE ERROR</text>
      <text x="0" y="44" fill="#20d7ff" fontFamily="monospace" fontSize="9" letterSpacing="2">SHEAF CUMULATIVE ERROR</text>
      <text x={width - 90} y="66" fill="#57f287" fontFamily="monospace" fontSize="9" letterSpacing="1.4">{Math.round(finalAdvantage)}% AUC SAVED</text>
    </svg>
  );
}

function PolicyStat({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="border border-white/10 bg-white/[0.035] p-3 transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.06]">
      <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-white/38">
        {icon}
        {label}
      </div>
      <div className="mt-2 font-mono text-2xl text-white">{value}</div>
    </div>
  );
}

function LiveMetric({ label, value, tone }: { label: string; value: string; tone: "cyan" | "amber" }) {
  const color = tone === "cyan" ? "#20d7ff" : "#ff9d2e";
  return (
    <div className="border-r border-white/10 p-4 transition-all duration-300 hover:bg-white/[0.03] last:border-r-0">
      <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-white/38">{label}</div>
      <div className="mt-2 font-mono text-2xl" style={{ color }}>{value}</div>
    </div>
  );
}

function cumulativeAdvantage(greedy?: number[], neural?: number[], step = 0) {
  if (!greedy?.length || !neural?.length) return 0;
  const idx = Math.min(step, greedy.length - 1, neural.length - 1);
  let greedyMass = 0;
  let removed = 0;
  for (let i = 0; i <= idx; i += 1) {
    greedyMass += greedy[i] ?? 0;
    removed += Math.max(0, (greedy[i] ?? 0) - (neural[i] ?? 0));
  }
  return greedyMass > 1e-6 ? (removed / greedyMass) * 100 : 0;
}

function cumulativeSeries(values: number[]) {
  let sum = 0;
  return values.map((value) => {
    sum += value;
    return sum;
  });
}

function capturedPerturbationMass(rollout: RolloutPayload | null, step: number) {
  const perturbation = rollout?.initialPerturbation;
  const observed = rollout?.observedNodes[Math.max(0, step - 1)] ?? [];
  if (!perturbation?.length || observed.length === 0) return 0;
  const total = perturbation.reduce((sum, value) => sum + value, 0);
  const captured = observed.reduce((sum, nodeId) => sum + (perturbation[nodeId] ?? 0), 0);
  return total > 1e-6 ? (captured / total) * 100 : 0;
}
