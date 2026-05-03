import { Activity, Crosshair, Pause, Play, RadioTower, SkipBack, SkipForward } from "lucide-react";
import type { ReactNode } from "react";
import type { RolloutPayload, RunSummary } from "../types";

interface ScoreboardProps {
  run: RunSummary | null;
  neural: RolloutPayload | null;
  greedy: RolloutPayload | null;
  chatgpt: RolloutPayload | null;
  step: number;
  maxStep: number;
  playing: boolean;
  setStep: (step: number) => void;
  setPlaying: (playing: boolean) => void;
}

export function Scoreboard({ run, neural, greedy, chatgpt, step, maxStep, playing, setStep, setPlaying }: ScoreboardProps) {
  const primary = run?.headlineMetrics[0];
  const secondary = run?.headlineMetrics[1];
  const jumpStep = (delta: number) => {
    setPlaying(false);
    setStep(Math.min(maxStep, Math.max(0, step + delta)));
  };

  return (
    <header className="relative z-10 grid grid-cols-1 gap-3 border-b border-white/10 bg-carbon-950/78 px-4 py-4 backdrop-blur-xl lg:grid-cols-[0.72fr_1.28fr_1.45fr] lg:px-5">
      <div className="flex min-w-0 items-center gap-2">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center border border-cyanOps/40 bg-cyanOps/10 shadow-cyan transition-all duration-300 hover:-translate-y-1 hover:scale-105 hover:bg-cyanOps/15">
          <RadioTower className="h-4 w-4 text-cyanOps" />
        </div>
        <div className="min-w-0">
          <div className="truncate font-mono text-[9px] uppercase tracking-[0.24em] text-cyanOps/80">
            closed-loop room
          </div>
          <h1 className="mt-1 truncate text-base font-semibold tracking-normal text-white">
            Manifold Control
          </h1>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricBlock
          icon={<Crosshair className="h-4 w-4" />}
          label={primary?.label ?? "Evidence Loading"}
          value={primary ? `${primary.value.toFixed(1)}${primary.unit}` : "--"}
          tone="cyan"
        />
        <MetricBlock
          icon={<Activity className="h-4 w-4" />}
          label={secondary?.label ?? "AUC Reduced"}
          value={secondary ? `${secondary.value.toFixed(1)}${secondary.unit}` : neural?.meanErrorAuc.toFixed(3) ?? "--"}
          tone="amber"
        />
      </div>

      <div className="grid grid-cols-[0.82fr_1fr_0.92fr_1fr_auto] items-center gap-2 text-center font-mono">
        <Readout label="step" value={`${step}/${maxStep}`} />
        <Readout label="neural" value={neural?.meanErrorAuc.toFixed(3) ?? "--"} />
        <Readout label="gpt" value={chatgpt?.meanErrorAuc.toFixed(3) ?? "--"} />
        <Readout label="greedy" value={greedy?.meanErrorAuc.toFixed(3) ?? "--"} />
        <div className="flex h-11 items-center overflow-hidden border border-cyanOps/30 bg-cyanOps/10 text-cyanOps shadow-[0_0_18px_rgba(32,215,255,0.08)]">
          <StepButton
            disabled={step <= 0}
            label="Previous step"
            onClick={() => jumpStep(-1)}
            icon={<SkipBack className="h-3.5 w-3.5" />}
          />
          <StepButton
            label={playing ? "Pause rollout" : "Play rollout"}
            onClick={() => setPlaying(!playing)}
            icon={playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            primary
          />
          <StepButton
            disabled={step >= maxStep}
            label="Next step"
            onClick={() => jumpStep(1)}
            icon={<SkipForward className="h-3.5 w-3.5" />}
          />
        </div>
      </div>
    </header>
  );
}

function MetricBlock({
  icon,
  label,
  value,
  tone
}: {
  icon: ReactNode;
  label: string;
  value: string;
  tone: "cyan" | "amber";
}) {
  const toneClass = tone === "cyan" ? "text-cyanOps shadow-cyan" : "text-amberOps shadow-amber";
  return (
    <div className={`flex min-h-[92px] flex-col items-center justify-center border border-white/10 bg-white/[0.045] px-4 py-3 text-center transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.065] ${toneClass}`}>
      <div className="flex items-center justify-center gap-2 font-mono text-[10px] uppercase tracking-[0.24em] text-white/50">
        <span className={toneClass}>{icon}</span>
        {label}
      </div>
      <div className="mt-1 text-3xl font-semibold tracking-normal text-white xl:text-4xl">{value}</div>
    </div>
  );
}

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-h-[52px] flex-col items-center justify-center border border-white/10 bg-white/[0.035] px-3 py-3 text-center transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:bg-white/[0.055]">
      <div className="text-[10px] uppercase tracking-[0.22em] text-white/40">{label}</div>
      <div className="mt-1 text-lg text-white">{value}</div>
    </div>
  );
}

function StepButton({
  icon,
  label,
  onClick,
  disabled,
  primary
}: {
  icon: ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
}) {
  return (
    <button
      className={`flex h-11 w-12 items-center justify-center border-l border-cyanOps/20 first:border-l-0 transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-25 ${
        primary
          ? "bg-cyanOps/10 hover:scale-105 hover:bg-cyanOps/22 hover:shadow-[0_0_24px_rgba(32,215,255,0.2)]"
          : "hover:bg-cyanOps/16 hover:text-white"
      }`}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
    >
      {icon}
    </button>
  );
}
