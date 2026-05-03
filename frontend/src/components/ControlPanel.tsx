import { Gauge, Network, Waves } from "lucide-react";
import type { ReactNode } from "react";
import { Slider } from "./ui/slider";

interface ControlPanelProps {
  lambdaSheaf: number;
  pIn: number;
  pOut: number;
  setLambdaSheaf: (value: number) => void;
  setPIn: (value: number) => void;
  setPOut: (value: number) => void;
}

export function ControlPanel({
  lambdaSheaf,
  pIn,
  pOut,
  setLambdaSheaf,
  setPIn,
  setPOut
}: ControlPanelProps) {
  return (
    <section className="border-t border-white/10 bg-carbon-950/72 p-4 backdrop-blur-xl">
      <div className="mb-3 flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.28em] text-white/50">
        <Gauge className="h-4 w-4 text-cyanOps" />
        topology controls
      </div>
      <ControlSlider
        icon={<Waves className="h-4 w-4" />}
        label="lambda_sheaf"
        hint="fold intensity + edge tension"
        value={lambdaSheaf}
        min={0.02}
        max={0.6}
        step={0.01}
        onValueChange={setLambdaSheaf}
      />
      <ControlSlider
        icon={<Network className="h-4 w-4" />}
        label="p_in"
        hint="community cohesion"
        value={pIn}
        min={0.01}
        max={0.35}
        step={0.005}
        onValueChange={setPIn}
      />
      <ControlSlider
        icon={<Network className="h-4 w-4 text-amberOps" />}
        label="p_out"
        hint="cross-community bridges"
        value={pOut}
        min={0.001}
        max={0.06}
        step={0.001}
        onValueChange={setPOut}
      />
    </section>
  );
}

function ControlSlider({
  icon,
  label,
  hint,
  value,
  min,
  max,
  step,
  onValueChange
}: {
  icon: ReactNode;
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onValueChange: (value: number) => void;
}) {
  return (
    <div className="mb-4 rounded-sm border border-transparent p-2 transition-all duration-300 hover:-translate-y-1 hover:border-cyanOps/20 hover:bg-cyanOps/[0.025] hover:shadow-[0_12px_34px_rgba(32,215,255,0.06)] last:mb-0">
      <div className="mb-2 flex items-center justify-between font-mono text-xs">
        <div className="flex items-center gap-2 text-white/66">
          {icon}
          <span>{label}</span>
        </div>
        <span className="text-cyanOps">{value.toFixed(3)}</span>
      </div>
      <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.18em] text-white/35">
        {hint}
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([next]) => onValueChange(next)}
      />
    </div>
  );
}
