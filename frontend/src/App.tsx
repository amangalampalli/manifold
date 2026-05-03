import { Loader2 } from "lucide-react";
import { useEffect } from "react";
import { ControlPanel } from "./components/ControlPanel";
import { PlanarGraphScene } from "./components/PlanarGraphScene";
import { Receipts } from "./components/Receipts";
import { Scoreboard } from "./components/Scoreboard";
import { Terminal } from "./components/Terminal";
import { selectMaxStep, useShowcaseStore } from "./store";

export default function App() {
  const {
    load,
    loading,
    error,
    runs,
    run,
    runIdx,
    graph,
    neural,
    greedy,
    chatgpt,
    telemetry,
    step,
    playing,
    selectedPolicy,
    loadRun,
    setStep,
    setPlaying,
    setSelectedPolicy,
    lambdaSheaf,
    pIn,
    pOut,
    setLambdaSheaf,
    setPIn,
    setPOut
  } = useShowcaseStore();
  const maxStep = useShowcaseStore(selectMaxStep);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!neural || !playing) return;
    const timer = window.setInterval(() => {
      setStep((useShowcaseStore.getState().step + 1) % (maxStep + 1));
    }, 1150);
    return () => window.clearInterval(timer);
  }, [maxStep, neural, playing, setStep]);

  if (loading) {
    return (
      <main className="flex h-screen items-center justify-center bg-carbon-950 text-white">
        <div className="flex items-center gap-3 font-mono uppercase tracking-[0.24em] text-cyanOps">
          <Loader2 className="h-5 w-5 animate-spin" />
          loading closed-loop evidence
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="flex h-screen items-center justify-center bg-carbon-950 p-8 text-white">
        <div className="max-w-xl border border-amberOps/40 bg-amberOps/10 p-6">
          <div className="font-mono text-sm uppercase tracking-[0.22em] text-amberOps">artifact link down</div>
          <p className="mt-3 text-white/75">{error}</p>
        </div>
      </main>
    );
  }

  return (
    <main className="flex h-screen flex-col overflow-hidden bg-carbon-950 text-white">
      <Scoreboard
        run={run}
        neural={neural}
        greedy={greedy}
        chatgpt={chatgpt}
        step={step}
        maxStep={maxStep}
        playing={playing}
        setStep={setStep}
        setPlaying={setPlaying}
      />
      <section className="grid min-h-0 flex-1 grid-cols-1 overflow-auto lg:grid-cols-[320px_minmax(0,1fr)_360px] lg:overflow-hidden">
        <div className="order-2 min-h-[620px] overflow-y-auto overflow-x-hidden lg:order-none lg:min-h-0">
          <Receipts
            run={run}
            neural={neural}
            greedy={greedy}
            chatgpt={chatgpt}
            selectedPolicy={selectedPolicy}
            step={step}
            setStep={setStep}
            setPlaying={setPlaying}
          />
          <ControlPanel
            lambdaSheaf={lambdaSheaf}
            pIn={pIn}
            pOut={pOut}
            setLambdaSheaf={setLambdaSheaf}
            setPIn={setPIn}
            setPOut={setPOut}
          />
        </div>
        <div className="order-1 relative min-h-[520px] lg:order-none lg:min-h-0">
          <PlanarGraphScene
            graph={graph}
            rollout={selectedPolicy === "neural" ? neural : selectedPolicy === "chatgpt" ? chatgpt : greedy}
            reference={neural}
            neural={neural}
            greedy={greedy}
            runs={runs}
            run={run}
            runIdx={runIdx}
            loadRun={loadRun}
            selectedPolicy={selectedPolicy}
            setSelectedPolicy={setSelectedPolicy}
            step={step}
            lambdaSheaf={lambdaSheaf}
            pIn={pIn}
            pOut={pOut}
          />
        </div>
        <div className="order-3 min-h-[520px] lg:order-none lg:min-h-0">
          <Terminal lines={telemetry} rollout={neural} step={step} />
        </div>
      </section>
    </main>
  );
}
