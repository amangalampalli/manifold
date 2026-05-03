import { TerminalSquare } from "lucide-react";
import type { RolloutPayload, TelemetryLine } from "../types";

interface TerminalProps {
  lines: TelemetryLine[];
  rollout: RolloutPayload | null;
  step: number;
}

export function Terminal({ lines, rollout, step }: TerminalProps) {
  const visible = lines.slice(0, Math.max(2, step + 1));
  const probes = rollout?.probeNodes[Math.max(0, step - 1)] ?? [];
  const observed = rollout?.observedNodes[Math.max(0, step - 1)] ?? [];
  const control = rollout?.controlEnergy[Math.max(0, step - 1)];
  const residual = rollout?.residualEnergy?.[Math.max(0, step - 1)] ?? rollout?.meanError[step];

  return (
    <aside className="grid h-full grid-rows-[auto_1fr_auto] border-l border-white/10 bg-carbon-950/74 backdrop-blur-xl">
      <div className="border-b border-white/10 p-4">
        <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.28em] text-cyanOps/80">
          <TerminalSquare className="h-4 w-4" />
          isr terminal
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <TerminalMetric label="observed" value={String(observed.length || "--")} />
          <TerminalMetric label="control" value={control?.toFixed(3) ?? "--"} />
          <TerminalMetric label="probes" value={String(probes.length || "--")} />
          <TerminalMetric label="residual" value={residual?.toFixed(3) ?? "--"} />
        </div>
      </div>

      <div className="overflow-hidden p-4 font-mono text-xs leading-6 text-white/70">
        {visible.map((line) => (
          <div key={`${line.step}-${line.message}`} className="border-b border-white/[0.045] py-1 transition-all duration-300 hover:translate-x-1 hover:border-cyanOps/20 hover:text-white">
            <span className="mr-2 text-cyanOps">[{String(line.step).padStart(2, "0")}]</span>
            {line.message}
          </div>
        ))}
        <div className="mt-3 text-amberOps/85">
          probe_nodes: {probes.slice(0, 12).join(", ") || "--"}
        </div>
      </div>

      <div className="border-t border-white/10 p-4 font-mono text-[11px] uppercase tracking-[0.2em] text-white/38">
        belief correction stream synced to graph horizon
      </div>
    </aside>
  );
}

function TerminalMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-white/10 bg-white/[0.035] px-3 py-2 transition-all duration-300 hover:-translate-y-1 hover:border-cyanOps/20 hover:bg-cyanOps/[0.025]">
      <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/35">{label}</div>
      <div className="mt-1 font-mono text-base text-white">{value}</div>
    </div>
  );
}
