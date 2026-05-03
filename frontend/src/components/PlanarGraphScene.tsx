import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, ReactNode, WheelEvent as ReactWheelEvent } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { GraphEdge, GraphNode, GraphPayload, Policy, RolloutPayload, RunSummary } from "../types";

interface PlanarGraphSceneProps {
  graph: GraphPayload | null;
  rollout: RolloutPayload | null;
  reference: RolloutPayload | null;
  neural: RolloutPayload | null;
  greedy: RolloutPayload | null;
  runs: RunSummary[];
  run: RunSummary | null;
  runIdx: number;
  loadRun: (runId: string, runIdx?: number) => Promise<void>;
  selectedPolicy: Exclude<Policy, "random">;
  setSelectedPolicy: (policy: Exclude<Policy, "random">) => void;
  step: number;
  lambdaSheaf: number;
  pIn: number;
  pOut: number;
}

interface ScreenNode extends GraphNode {
  sx: number;
  sy: number;
}

const WIDTH = 1000;
const HEIGHT = 720;
const PAD = 70;
type ViewMode = "controller" | "mesh";
type ViewTransform = { x: number; y: number; k: number };
type NodeOffsets = Record<number, { x: number; y: number }>;

export function PlanarGraphScene({
  graph,
  rollout,
  reference,
  neural,
  greedy,
  runs,
  run,
  runIdx,
  loadRun,
  selectedPolicy,
  setSelectedPolicy,
  step,
  lambdaSheaf,
  pIn,
  pOut
}: PlanarGraphSceneProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("controller");
  const [controlsOpen, setControlsOpen] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [transform, setTransform] = useState<ViewTransform>({ x: 0, y: 0, k: 1 });
  const [dragStart, setDragStart] = useState<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const [grabbedNodeId, setGrabbedNodeId] = useState<number | null>(null);
  const [nodeOffsets, setNodeOffsets] = useState<NodeOffsets>({});
  const [clock, setClock] = useState(0);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const projected = useMemo(
    () => (graph ? projectNodes(graph.nodes, pIn, pOut, lambdaSheaf) : []),
    [graph, lambdaSheaf, pIn, pOut]
  );
  const animatedNodes = useMemo(
    () => projected.map((node) => animateNode(node, clock, viewMode, nodeOffsets[node.id], lambdaSheaf, pIn, graph?.numNodes ?? projected.length)),
    [clock, graph?.numNodes, lambdaSheaf, nodeOffsets, pIn, projected, viewMode]
  );
  const observed = useMemo(
    () => new Set(reference?.observedNodes[Math.max(0, step - 1)] ?? []),
    [reference, step]
  );
  const probes = useMemo(
    () => new Set(reference?.probeNodes[Math.max(0, step - 1)] ?? []),
    [reference, step]
  );
  const previousProbes = useMemo(
    () => new Set(reference?.probeNodes[Math.max(0, step - 2)] ?? []),
    [reference, step]
  );
  const nextProbes = useMemo(
    () => new Set(reference?.probeNodes[Math.max(0, step)] ?? []),
    [reference, step]
  );
  const nodeById = useMemo(() => new Map(animatedNodes.map((node) => [node.id, node])), [animatedNodes]);
  const communities = useMemo(() => communitySummaries(animatedNodes), [animatedNodes]);
  const contextEdges = useMemo(
    () => selectContextEdges(graph?.edges ?? [], pOut, graph?.numNodes ?? 0),
    [graph?.edges, graph?.numNodes, pOut]
  );
  const crackEdges = useMemo(
    () => selectCrackEdges(graph?.edges ?? [], graph?.numNodes ?? 0, lambdaSheaf),
    [graph?.edges, graph?.numNodes, lambdaSheaf]
  );
  const activeEdges = useMemo(
    () => selectActiveEdges(graph?.edges ?? [], observed, probes, crackEdges),
    [graph?.edges, observed, probes, crackEdges]
  );
  const accent = selectedPolicy === "neural" ? "#20d7ff" : selectedPolicy === "chatgpt" ? "#c084fc" : "#ff9d2e";
  const selectedNode = selectedNodeId === null ? null : animatedNodes.find((node) => node.id === selectedNodeId) ?? null;
  const selectedStats = selectedNode
    ? nodeStats(selectedNode.id, rollout, reference, neural, greedy, step, observed, probes)
    : null;
  const controllerLabel = selectedPolicy === "neural" ? "sheaf ode" : selectedPolicy === "chatgpt" ? "chatgpt" : "greedy";

  useEffect(() => {
    const timer = window.setInterval(() => {
      setClock(performance.now() / 1000);
      setNodeOffsets((current) => relaxNodeOffsets(current, grabbedNodeId));
    }, 33);
    return () => window.clearInterval(timer);
  }, [grabbedNodeId]);

  const handleWheel = (event: ReactWheelEvent<SVGSVGElement>) => {
    event.preventDefault();
    const nextK = clamp(transform.k * (event.deltaY > 0 ? 0.9 : 1.1), 0.65, 3.2);
    setTransform((current) => ({ ...current, k: nextK }));
  };

  const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (grabbedNodeId !== null) return;
    if (event.button !== 0) return;
    svgRef.current?.setPointerCapture(event.pointerId);
    setDragStart({ x: event.clientX, y: event.clientY, tx: transform.x, ty: transform.y });
  };

  const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (grabbedNodeId !== null) {
      const baseNode = projected.find((node) => node.id === grabbedNodeId);
      if (!baseNode) return;
      const point = clientToMap(event, svgRef.current, transform);
      setNodeOffsets((current) => ({
        ...current,
        [grabbedNodeId]: {
          x: point.x - baseNode.sx,
          y: point.y - baseNode.sy
        }
      }));
      return;
    }
    if (!dragStart) return;
    setTransform((current) => ({
      ...current,
      x: dragStart.tx + (event.clientX - dragStart.x) / current.k,
      y: dragStart.ty + (event.clientY - dragStart.y) / current.k
    }));
  };

  const handlePointerUp = (event: ReactPointerEvent<SVGSVGElement>) => {
    svgRef.current?.releasePointerCapture(event.pointerId);
    setDragStart(null);
    setGrabbedNodeId(null);
  };

  const handleNodePointerDown = (nodeId: number, event: ReactPointerEvent<SVGGElement>) => {
    event.stopPropagation();
    svgRef.current?.setPointerCapture(event.pointerId);
    setSelectedNodeId(nodeId);
    setGrabbedNodeId(nodeId);
    setDragStart(null);
  };

  return (
    <div className="relative h-full overflow-hidden border-x border-white/10 bg-black/20">
      <div className="absolute left-1/2 top-5 z-20 w-[min(920px,calc(100%-40px))] -translate-x-1/2 font-mono uppercase">
        <div
          className={`flex origin-top items-center gap-3 overflow-x-auto overflow-y-hidden border border-cyanOps/20 bg-carbon-950/82 px-3 font-mono text-[11px] tracking-[0.18em] shadow-[0_18px_60px_rgba(0,0,0,0.42),0_0_40px_rgba(32,215,255,0.08)] backdrop-blur-xl transition-all duration-500 ease-out ${
            controlsOpen
              ? "max-h-[76px] scale-100 py-3 opacity-100"
              : "max-h-[50px] scale-[0.98] py-2 opacity-95"
          }`}
        >
          <button
            className="group flex shrink-0 items-center gap-3 border border-cyanOps/25 bg-cyanOps/10 px-3 py-2 text-[10px] tracking-[0.2em] text-cyanOps/85 transition-all duration-300 hover:-translate-y-1 hover:scale-[1.02] hover:border-cyanOps/60 hover:bg-cyanOps/18 hover:text-cyanOps hover:shadow-[0_0_24px_rgba(32,215,255,0.18)]"
            onClick={() => setControlsOpen((open) => !open)}
          >
            <span className={`h-2 w-2 bg-cyanOps shadow-[0_0_16px_rgba(32,215,255,0.75)] transition-all duration-500 ${controlsOpen ? "scale-100 opacity-100" : "scale-75 opacity-55"}`} />
            map controls
          </button>
          <div
            className={`flex min-w-0 flex-1 items-center gap-4 transition-all duration-500 ${
              controlsOpen ? "translate-x-0 opacity-100" : "pointer-events-none -translate-x-4 opacity-0"
            }`}
          >
            <ControlRow label="size">
              {runs.map((candidate) => (
                <PolicyButton
                  key={candidate.id}
                  active={run?.id === candidate.id}
              label={candidate.id === "graph-48" ? "48" : candidate.id === "graph-512" ? "512" : candidate.label}
                  onClick={() => {
                    setSelectedNodeId(null);
                    setTransform({ x: 0, y: 0, k: 1 });
                    void loadRun(candidate.id, candidate.defaultRunIdx);
                  }}
                />
              ))}
            </ControlRow>
            <ControlRow label="ctrl">
              <PolicyButton
                active={selectedPolicy === "greedy"}
                label="greedy"
                onClick={() => setSelectedPolicy("greedy")}
              />
              <PolicyButton
                active={selectedPolicy === "chatgpt"}
                label="gpt"
                onClick={() => setSelectedPolicy("chatgpt")}
              />
              <PolicyButton
                active={selectedPolicy === "neural"}
              label="ode"
                onClick={() => setSelectedPolicy("neural")}
              />
            </ControlRow>
            <ControlRow label="view">
              <PolicyButton
                active={viewMode === "controller"}
                label="probe"
                onClick={() => setViewMode("controller")}
              />
              <PolicyButton
                active={viewMode === "mesh"}
                label="mesh"
                onClick={() => setViewMode("mesh")}
              />
            </ControlRow>
            <button className="ml-auto whitespace-nowrap border border-white/10 bg-white/[0.035] px-3 py-2 text-cyanOps/75 transition duration-200 hover:-translate-y-0.5 hover:border-cyanOps/35 hover:text-cyanOps" onClick={() => setTransform({ x: 0, y: 0, k: 1 })}>
              reset
            </button>
          </div>
        </div>
      </div>

      {viewMode === "mesh" ? (
        <HairballMesh
          graph={graph}
          rollout={rollout}
          neural={neural}
          greedy={greedy}
          step={step}
          selectedPolicy={selectedPolicy}
          lambdaSheaf={lambdaSheaf}
          pIn={pIn}
          pOut={pOut}
          onSelectNode={setSelectedNodeId}
        />
      ) : (
        <svg
          ref={svgRef}
          className={`h-full w-full ${dragStart ? "cursor-grabbing" : "cursor-grab"}`}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          onWheel={handleWheel}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          onClick={() => setSelectedNodeId(null)}
        >
          <style>
            {`
              .explore-flow {
                animation: explore-dash 1.2s linear infinite;
              }
              @keyframes explore-dash {
                to { stroke-dashoffset: -32; }
              }
            `}
          </style>
          <rect width={WIDTH} height={HEIGHT} fill="#030507" />
          <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.k})`}>
            <ControllerMap
              title={
                `${controllerLabel} probe sweep`
              }
              viewMode={viewMode}
              selectedPolicy={selectedPolicy}
              nodes={animatedNodes}
              contextEdges={contextEdges}
              crackEdges={crackEdges}
              activeEdges={activeEdges}
              nodeById={nodeById}
              communities={communities}
              rollout={rollout}
              reference={reference}
              neural={neural}
              greedy={greedy}
              step={step}
              observed={observed}
              probes={probes}
              previousProbes={previousProbes}
              nextProbes={nextProbes}
              accent={accent}
              clock={clock}
              selectedNodeId={selectedNodeId}
              setSelectedNodeId={setSelectedNodeId}
              onNodePointerDown={handleNodePointerDown}
            />
          </g>
        </svg>
      )}

      <div className="group absolute right-5 top-32 z-10 font-mono uppercase">
        <div className="ml-auto flex w-fit items-center gap-2 border border-cyanOps/25 bg-carbon-950/82 px-3 py-2 text-[10px] tracking-[0.18em] text-cyanOps/80 backdrop-blur transition-all duration-300 group-hover:-translate-y-0.5 group-hover:border-cyanOps/55 group-hover:bg-carbon-900/90 group-hover:shadow-[0_0_26px_rgba(32,215,255,0.12)]">
          <span className="h-2 w-2 bg-cyanOps shadow-[0_0_14px_rgba(32,215,255,0.65)]" />
          map key
        </div>
        <div className="pointer-events-none mt-2 w-64 origin-top-right translate-x-4 scale-95 border border-white/10 bg-carbon-950/84 p-3 text-[10px] tracking-[0.15em] text-white/55 opacity-0 shadow-[0_18px_50px_rgba(0,0,0,0.36)] backdrop-blur-xl transition-all duration-300 group-hover:pointer-events-auto group-hover:translate-x-0 group-hover:scale-100 group-hover:opacity-100">
          <LegendItem color="#475569" label="graph context" />
          {viewMode === "mesh" ? (
            <LegendItem color="#64748b" label="all graph edges" />
          ) : viewMode === "controller" ? (
            <LegendItem color="#20d7ff" label="probe target" />
          ) : (
            <LegendItem color="#20d7ff" label="sensing ring" />
          )}
          <LegendItem color="#ff4d5e" label="perturbation" />
          <LegendItem color="#20d7ff" label="dominance halo" />
          <LegendItem color="#57f287" label="sheaf cracks" />
          <LegendItem color="#7c8cff" label="control effort" />
          <LegendItem color="#ffffff" label="probe node" />
        </div>
      </div>
      <div className="pointer-events-none absolute bottom-5 left-5 max-w-xl border border-cyanOps/25 bg-carbon-950/75 px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-cyanOps/80 backdrop-blur">
        {viewMode === "controller"
          ? "grab nodes to interrogate the probe sweep; release and they snap back to formation"
          : viewMode === "mesh"
            ? "full graph complexity mesh / every edge rendered / pull nodes to feel the topology"
          : `${run?.label ?? "selected graph"} / rollout ${runIdx} / drag map to pan, scroll to zoom, or pull a node`}
      </div>
      {selectedNode && selectedStats ? (
        <NodeInspector node={selectedNode} stats={selectedStats} accent={accent} onClose={() => setSelectedNodeId(null)} />
      ) : null}
    </div>
  );
}

function ControlRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex shrink-0 items-center gap-2" aria-label={label}>
      <div className="inline-grid grid-flow-col auto-cols-max border border-white/10 bg-black/30 p-1 transition-all duration-300 hover:-translate-y-0.5 hover:border-white/20 hover:bg-white/[0.04]">
        {children}
      </div>
    </div>
  );
}

function PolicyButton({
  active,
  label,
  onClick
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      className={`whitespace-nowrap px-3 py-2 transition-all duration-300 ${
        active
          ? "scale-[1.02] bg-cyanOps text-carbon-950 shadow-[0_0_18px_rgba(32,215,255,0.28)]"
          : "text-white/55 hover:scale-[1.02] hover:bg-white/10 hover:text-white hover:shadow-[0_0_12px_rgba(255,255,255,0.08)]"
      }`}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

function ControllerMap({
  title,
  viewMode,
  selectedPolicy,
  nodes,
  contextEdges,
  crackEdges,
  activeEdges,
  nodeById,
  communities,
  rollout,
  reference,
  neural,
  greedy,
  step,
  observed,
  probes,
  previousProbes,
  nextProbes,
  accent,
  clock,
  selectedNodeId,
  setSelectedNodeId,
  onNodePointerDown
}: {
  title: string;
  viewMode: ViewMode;
  selectedPolicy: Exclude<Policy, "random">;
  nodes: ScreenNode[];
  contextEdges: GraphEdge[];
  crackEdges: GraphEdge[];
  activeEdges: GraphEdge[];
  nodeById: Map<number, ScreenNode>;
  communities: ReturnType<typeof communitySummaries>;
  rollout: RolloutPayload | null;
  reference: RolloutPayload | null;
  neural: RolloutPayload | null;
  greedy: RolloutPayload | null;
  step: number;
  observed: Set<number>;
  probes: Set<number>;
  previousProbes: Set<number>;
  nextProbes: Set<number>;
  accent: string;
  clock: number;
  selectedNodeId: number | null;
  setSelectedNodeId: (nodeId: number) => void;
  onNodePointerDown: (nodeId: number, event: ReactPointerEvent<SVGGElement>) => void;
}) {
	  const currentError = rollout?.nodeError[step] ?? rollout?.nodeError[0] ?? [];
	  const initialNeuralError = neural?.nodeError[0] ?? [];
  const control = rollout?.controlMagnitude?.[Math.max(0, step - 1)] ?? [];
  const neuralError = neural?.nodeError[step] ?? [];
  const greedyError = greedy?.nodeError[step] ?? [];
  const dominance = nodes.map((node) => Math.max(0, (greedyError[node.id] ?? 0) - (neuralError[node.id] ?? 0)));
  const dominanceCutoff = quantile(dominance.filter((value) => value > 1e-6), 0.7);
  const maxDominance = Math.max(...dominance, 1e-6);
  const perturbation = reference?.initialPerturbation ?? rollout?.initialPerturbation ?? [];
  const maxPerturbation = Math.max(...perturbation, 1e-6);
  const perturbationCutoff = quantile(perturbation.filter((value) => value > 1e-6), 0.75);
  const maxControl = Math.max(...control, 1e-6);
  const effortCutoff = quantile(control.filter((value) => value > 1e-6), 0.88);
  const maxError = Math.max(...currentError, 1e-6);
  const highEffortNodes = nodes.filter((node) => (control[node.id] ?? 0) >= effortCutoff);

  return (
    <g>
      <text x="40" y="118" fill={accent} fontFamily="monospace" fontSize="14" letterSpacing="4">
        {title.toUpperCase()}
      </text>
      <g opacity="0.14">
        {communities.map((community) => (
          <circle
            key={`c-${community.id}`}
            cx={community.cx}
            cy={community.cy}
            r={community.radius}
            fill="none"
            stroke="#20d7ff"
            strokeWidth="1"
            strokeDasharray="5 8"
          />
        ))}
      </g>
      <ExplorationPulses
        nodeById={nodeById}
        probes={probes}
        previousProbes={previousProbes}
        accent={accent}
      />
      <g>
        {contextEdges.map((edge, idx) => {
          const source = nodeById.get(edge.source);
          const target = nodeById.get(edge.target);
          if (!source || !target) return null;
          return (
            <line
              key={`context-${edge.source}-${edge.target}-${idx}`}
              x1={source.sx}
              y1={source.sy}
              x2={target.sx}
              y2={target.sy}
              stroke="#64748b"
              strokeWidth={viewMode === "mesh" ? 0.55 : 0.7}
              opacity={viewMode === "mesh" ? 0.16 : 0.05}
            />
          );
        })}
      </g>
      <g>
        {activeEdges.map((edge, idx) => {
          const source = nodeById.get(edge.source);
          const target = nodeById.get(edge.target);
          if (!source || !target) return null;
          const currentProbeEdge = probes.has(edge.source) || probes.has(edge.target);
          const previousProbeEdge = previousProbes.has(edge.source) || previousProbes.has(edge.target);
          const newlyCommitted = currentProbeEdge && !previousProbeEdge;
          const fullyObserved = observed.has(edge.source) && observed.has(edge.target);
          return (
            <line
              key={`active-${edge.source}-${edge.target}-${idx}`}
              x1={source.sx}
              y1={source.sy}
              x2={target.sx}
              y2={target.sy}
              stroke="#20d7ff"
              strokeWidth={newlyCommitted ? 2.2 : currentProbeEdge ? 1.55 : 0.9}
              strokeDasharray={newlyCommitted ? "none" : currentProbeEdge ? "none" : "4 7"}
              opacity={newlyCommitted ? 0.72 : currentProbeEdge ? 0.48 : fullyObserved ? 0.24 : 0.14}
            />
          );
        })}
      </g>
      <g>
        {crackEdges.map((edge, idx) => {
          const source = nodeById.get(edge.source);
          const target = nodeById.get(edge.target);
          if (!source || !target) return null;
          const visible =
            observed.has(edge.source) ||
            observed.has(edge.target) ||
            probes.has(edge.source) ||
            probes.has(edge.target);
          if (!visible) return null;
          const discovered = observed.has(edge.source) && observed.has(edge.target);
          return (
            <line
              key={`crack-${edge.source}-${edge.target}-${idx}`}
              x1={source.sx}
              y1={source.sy}
              x2={target.sx}
              y2={target.sy}
              stroke="#57f287"
              strokeWidth={discovered ? 2.4 : 1.75}
              strokeDasharray={discovered ? "none" : "5 7"}
              opacity={discovered ? 0.9 : 0.55}
            />
          );
        })}
      </g>
      <g>
        {nodes.map((node) => {
          const perturbed = perturbation[node.id] ?? 0;
          const error = currentError[node.id] ?? 0;
          const effort = control[node.id] ?? 0;
          const isProbe = probes.has(node.id);
          const isObserved = observed.has(node.id);
          const isSelected = selectedNodeId === node.id;
          const isControllerView = viewMode === "controller";
	          const nodeDominance = dominance[node.id] ?? 0;
	          const showDominance = nodeDominance >= dominanceCutoff;
	          const healing = healingProgress(node.id, neural, reference, initialNeuralError, step);
	          const fixedByController = perturbed > 0 && selectedPolicy === "neural" && healing.normalized;
	          const redFade = selectedPolicy === "neural" ? 1 - healing.progress : 1;
	          const showPerturbation = perturbed > 0 && perturbed >= perturbationCutoff;
	          const backgroundPerturbationRadius =
	            perturbed > 0 && !showPerturbation && !fixedByController ? (2.2 + (perturbed / perturbationCutoff) * 3.2) * (0.65 + redFade * 0.35) : 0;
	          const perturbationRadius = showPerturbation && !fixedByController ? (3.4 + (perturbed / maxPerturbation) * 8.5) * (0.68 + redFade * 0.32) : 0;
          const errorRadius = 2.2 + (error / maxError) * 3.8;
          const showEffort = effort > 0 && effort >= effortCutoff;
          const effortRadius = showEffort ? 7 + (effort / maxControl) * 12 : 0;
          const baseRadius = fixedByController ? 2.8 : isProbe ? 5.2 : isObserved && !isControllerView ? 3.2 : errorRadius;
          const baseFill = fixedByController ? "#94a3b8" : isObserved && !isControllerView ? "#20d7ff" : "#94a3b8";
          const baseOpacity = fixedByController
            ? 0.28
            : isProbe
              ? 0.38
              : isObserved && !isControllerView
                ? 0.54
                : isControllerView
                  ? 0.13
                  : 0.22;
          return (
            <g
              key={`n-${node.id}`}
              className="cursor-pointer"
              onClick={(event) => {
                event.stopPropagation();
                setSelectedNodeId(node.id);
              }}
              onPointerDown={(event) => onNodePointerDown(node.id, event)}
            >
              {isObserved && !isProbe && !isControllerView && !fixedByController ? (
                <circle cx={node.sx} cy={node.sy} r="6.2" fill="none" stroke="#20d7ff" strokeWidth="0.9" opacity="0.28" />
              ) : null}
              <circle
                cx={node.sx}
                cy={node.sy}
                r={baseRadius}
                fill={baseFill}
                opacity={baseOpacity}
                stroke={error > 0.7 && !fixedByController ? "#ff4d5e" : "none"}
                strokeWidth={error > 0.7 && !fixedByController ? 1.1 : 0}
              />
              {effortRadius > 0 ? (
                <>
                  <SvgPulse cx={node.sx} cy={node.sy} r={effortRadius + 2} stroke="#7c8cff" opacity={0.24} />
                  <circle
                    cx={node.sx}
                    cy={node.sy}
                    r={effortRadius}
                    fill="none"
                    stroke="#7c8cff"
                    strokeWidth="2.2"
                    opacity="0.9"
                  />
                </>
              ) : null}
	              {showDominance && !fixedByController ? (
                <>
                  <circle
                    className="animate-pulse"
                    cx={node.sx}
                    cy={node.sy}
                    r={10 + (nodeDominance / maxDominance) * 16}
                    fill="#20d7ff"
                    opacity={selectedPolicy === "neural" ? 0.12 : 0.07}
                  />
                  <circle
                    cx={node.sx}
                    cy={node.sy}
                    r={8 + (nodeDominance / maxDominance) * 12}
                    fill="none"
                    stroke="#20d7ff"
                    strokeWidth={selectedPolicy === "neural" ? 1.6 : 1}
                    strokeDasharray={selectedPolicy === "neural" ? "none" : "3 5"}
                    opacity={selectedPolicy === "neural" ? 0.72 : 0.38}
                  />
                </>
	              ) : null}
              {backgroundPerturbationRadius > 0 ? (
                <>
                  <circle
                    cx={node.sx}
                    cy={node.sy}
                    r={backgroundPerturbationRadius + 2.6}
                    fill="#ff4d5e"
                    opacity={(isObserved ? 0.18 : 0.1) * (0.22 + redFade * 0.78)}
                  />
                  <circle
                    cx={node.sx}
                    cy={node.sy}
                    r={backgroundPerturbationRadius}
                    fill="#ff4d5e"
                    opacity={(isObserved ? 0.58 : 0.42) * (0.18 + redFade * 0.82)}
                  />
                </>
              ) : null}
              {perturbationRadius > 0.4 ? (
                <>
                  <circle
                    cx={node.sx}
                    cy={node.sy}
                    r={perturbationRadius + (isObserved ? 3.5 : 1.5)}
                    fill="#ff4d5e"
                    opacity={(isObserved ? 0.22 : 0.1) * (0.2 + redFade * 0.8)}
                  />
                  <circle
                    cx={node.sx}
                    cy={node.sy}
                    r={perturbationRadius}
                    fill="#ff4d5e"
                    opacity={0.16 + redFade * 0.76}
                  />
                </>
              ) : null}
              {isProbe ? (
                <>
                  <SvgPulse cx={node.sx} cy={node.sy} r={8.2} stroke="#ffffff" opacity={0.3} />
                  <circle
                    cx={node.sx}
                    cy={node.sy}
                    r="5.2"
                    fill="#ffffff"
                    opacity="1"
                    stroke="#030507"
                    strokeWidth="1"
                  />
                </>
              ) : null}
              {isSelected ? (
                <circle
                  cx={node.sx}
                  cy={node.sy}
                  r={Math.max(14, effortRadius + 4, perturbationRadius + 4)}
                  fill="none"
                  stroke={accent}
                  strokeWidth="2"
                  opacity="0.95"
                />
              ) : null}
            </g>
          );
        })}
      </g>
      <g>
        {communities.map((community) => (
          <text
            key={`label-${community.id}`}
            x={community.cx}
            y={community.cy}
            fill="rgba(255,255,255,0.38)"
            fontFamily="monospace"
            fontSize="12"
            textAnchor="middle"
          >
            C{community.id}
          </text>
        ))}
      </g>
      {viewMode === "controller" ? (
        <DecisionLayer
          nodeById={nodeById}
          probes={probes}
          previousProbes={previousProbes}
          nextProbes={nextProbes}
          highEffortNodes={highEffortNodes}
          accent={accent}
          step={step}
          clock={clock}
        />
      ) : null}
    </g>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="mb-1 flex items-center gap-2 last:mb-0">
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      <span>{label}</span>
    </div>
  );
}

function SvgPulse({
  cx,
  cy,
  r,
  stroke,
  opacity
}: {
  cx: number;
  cy: number;
  r: number;
  stroke: string;
  opacity: number;
}) {
  return (
    <circle cx={cx} cy={cy} r={r} fill="none" stroke={stroke} strokeWidth="1.3" opacity={opacity}>
      <animate attributeName="r" values={`${r};${r * 1.75};${r}`} dur="1.6s" repeatCount="indefinite" />
      <animate attributeName="opacity" values={`${opacity};0.03;${opacity}`} dur="1.6s" repeatCount="indefinite" />
    </circle>
  );
}

function HairballMesh({
  graph,
  rollout,
  neural,
  greedy,
  step,
  selectedPolicy,
  lambdaSheaf,
  pIn,
  pOut,
  onSelectNode
}: {
  graph: GraphPayload | null;
  rollout: RolloutPayload | null;
  neural: RolloutPayload | null;
  greedy: RolloutPayload | null;
  step: number;
  selectedPolicy: Exclude<Policy, "random">;
  lambdaSheaf: number;
  pIn: number;
  pOut: number;
  onSelectNode: (nodeId: number) => void;
}) {
  const [meshHeld, setMeshHeld] = useState(false);
  return (
    <div
      className="relative h-full w-full bg-[#030507]"
      onPointerDown={() => setMeshHeld(true)}
      onPointerUp={() => setMeshHeld(false)}
      onPointerCancel={() => setMeshHeld(false)}
      onPointerLeave={() => setMeshHeld(false)}
    >
      <Canvas camera={{ position: [0, 0, 18], fov: 48 }} dpr={[1, 1.8]} gl={{ antialias: true }}>
        <color attach="background" args={["#030507"]} />
        <fog attach="fog" args={["#030507", 10, 28]} />
        <ambientLight intensity={0.85} />
        <pointLight position={[6, 8, 10]} intensity={28} color="#20d7ff" />
        <pointLight position={[-8, -4, 7]} intensity={16} color="#ff4d5e" />
        {graph && rollout ? (
          <HairballCloud
            graph={graph}
            rollout={rollout}
            neural={neural}
            greedy={greedy}
            step={step}
            selectedPolicy={selectedPolicy}
            lambdaSheaf={lambdaSheaf}
            pIn={pIn}
            pOut={pOut}
            held={meshHeld}
            onSelectNode={onSelectNode}
          />
        ) : null}
        <MeshOrbitControls held={meshHeld} />
      </Canvas>
      <div className="pointer-events-none absolute bottom-24 right-8 max-w-xs border border-white/10 bg-carbon-950/72 p-3 font-mono text-[10px] uppercase tracking-[0.16em] text-white/48 backdrop-blur">
        probe view condenses this topology into the controller's actionable sensing field
      </div>
    </div>
  );
}

function HairballCloud({
  graph,
  rollout,
  neural,
  greedy,
  step,
  selectedPolicy,
  lambdaSheaf,
  pIn,
  pOut,
  held,
  onSelectNode
}: {
  graph: GraphPayload;
  rollout: RolloutPayload;
  neural: RolloutPayload | null;
  greedy: RolloutPayload | null;
  step: number;
  selectedPolicy: Exclude<Policy, "random">;
  lambdaSheaf: number;
  pIn: number;
  pOut: number;
  held: boolean;
  onSelectNode: (nodeId: number) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const nodesRef = useRef<THREE.InstancedMesh>(null);
  const edgesRef = useRef<THREE.LineSegments>(null);
  const time = useRef(0);
  const matrix = useMemo(() => new THREE.Matrix4(), []);
  const positions = useMemo(() => meshBasePositions(graph, pIn), [graph, pIn]);
  const edgeGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(graph.edges.length * 2 * 3), 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(graph.edges.length * 2 * 3), 3));
    return geometry;
  }, [graph.edges.length]);

  useFrame((_, delta) => {
    time.current += delta;
    const phase = time.current;
    if (groupRef.current) {
      if (!held) {
        groupRef.current.rotation.y += delta * (0.07 + lambdaSheaf * 0.12);
        groupRef.current.rotation.x = -0.45 + Math.sin(phase * 0.22) * (0.04 + lambdaSheaf * 0.18);
        groupRef.current.rotation.z = Math.sin(phase * 0.18) * (0.02 + pOut * 1.2);
      }
    }
    const edgePositions = edgeGeometry.getAttribute("position") as THREE.BufferAttribute;
    const edgeColors = edgeGeometry.getAttribute("color") as THREE.BufferAttribute;
	    const nodeError = rollout.nodeError[step] ?? rollout.nodeError[0] ?? [];
	    const initialNeuralError = neural?.nodeError[0] ?? [];
    const neuralError = neural?.nodeError[step] ?? [];
    const greedyError = greedy?.nodeError[step] ?? [];
    const perturbation = rollout.initialPerturbation;
    const maxPerturbation = Math.max(...perturbation, 1e-6);

    for (const node of graph.nodes) {
      const base = positions[node.id] ?? new THREE.Vector3();
	      const error = nodeError[node.id] ?? 0;
	      const dominance = Math.max(0, (greedyError[node.id] ?? 0) - (neuralError[node.id] ?? 0));
	      const healing = healingProgress(node.id, neural, rollout, initialNeuralError, step);
	      const fixedByController =
	        (perturbation[node.id] ?? 0) > 0 &&
	        selectedPolicy === "neural" &&
	        healing.normalized;
	      const redFade = selectedPolicy === "neural" ? 1 - healing.progress : 1;
      const pulse = Math.sin(phase * 1.7 + node.id * 0.33) * (0.08 + lambdaSheaf * 0.45);
      const x = base.x + Math.sin(phase * 0.7 + node.community) * (0.04 + pOut * 1.8);
      const y = base.y + Math.cos(phase * 0.6 + node.id * 0.11) * (0.04 + pOut * 1.8);
      const z = base.z + pulse + Math.sin(base.x * 0.8 + phase * 0.45) * (0.1 + lambdaSheaf * 0.8);
	      const nodeColor =
	        fixedByController
	          ? new THREE.Color("#94a3b8")
	          : dominance > 0.05 && selectedPolicy === "neural"
	          ? new THREE.Color("#20d7ff")
          : (perturbation[node.id] ?? 0) / maxPerturbation > 0.65
            ? new THREE.Color("#ff4d5e")
            : new THREE.Color("#94a3b8");
      if (!fixedByController && selectedPolicy === "neural" && (perturbation[node.id] ?? 0) > 0) {
        nodeColor.lerp(new THREE.Color("#94a3b8"), healing.progress * 0.72);
        nodeColor.multiplyScalar(0.78 + redFade * 0.22);
      }
      if (error > 0.75 && !fixedByController) nodeColor.lerp(new THREE.Color("#ff9d2e"), 0.45);
      nodeColor.multiplyScalar(1.7);
      const size = graph.numNodes > 128 ? 0.07 : 0.13;
      matrix.compose(
        new THREE.Vector3(x, y, z),
        new THREE.Quaternion(),
        new THREE.Vector3(size, size, size)
      );
      nodesRef.current?.setMatrixAt(node.id, matrix);
      nodesRef.current?.setColorAt(node.id, nodeColor);
    }

    graph.edges.forEach((edge, edgeIdx) => {
      const source = positions[edge.source] ?? new THREE.Vector3();
      const target = positions[edge.target] ?? new THREE.Vector3();
      const wobble = (!edge.sameCommunity || edge.distortion > 0.08) ? Math.sin(phase * 2.4 + edgeIdx * 0.07) * (0.08 + lambdaSheaf * 0.55 + pOut * 1.4) : 0;
      const offset = edgeIdx * 2;
      edgePositions.setXYZ(offset, source.x, source.y, source.z + wobble);
      edgePositions.setXYZ(offset + 1, target.x, target.y, target.z - wobble);
      const edgeColor = !edge.sameCommunity || edge.distortion > 0.08
        ? new THREE.Color("#57f287")
        : new THREE.Color("#20d7ff");
      edgeColor.multiplyScalar(edge.sameCommunity ? 0.55 + pIn * 1.2 : 0.95 + pOut * 4.5);
      edgeColors.setXYZ(offset, edgeColor.r, edgeColor.g, edgeColor.b);
      edgeColors.setXYZ(offset + 1, edgeColor.r, edgeColor.g, edgeColor.b);
    });

    if (nodesRef.current) {
      nodesRef.current.instanceMatrix.needsUpdate = true;
      if (nodesRef.current.instanceColor) nodesRef.current.instanceColor.needsUpdate = true;
    }
    edgePositions.needsUpdate = true;
    edgeColors.needsUpdate = true;
  });

  return (
    <group ref={groupRef}>
      <lineSegments ref={edgesRef} geometry={edgeGeometry}>
        <lineBasicMaterial vertexColors transparent opacity={0.82} blending={THREE.AdditiveBlending} depthWrite={false} />
      </lineSegments>
      <instancedMesh
        ref={nodesRef}
        args={[undefined, undefined, graph.nodes.length]}
        onClick={(event) => {
          event.stopPropagation();
          if (event.instanceId !== undefined) onSelectNode(event.instanceId);
        }}
        onPointerMove={() => {
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "";
        }}
      >
        <sphereGeometry args={[1, 12, 12]} />
        <meshBasicMaterial vertexColors transparent opacity={1} blending={THREE.AdditiveBlending} depthWrite={false} />
      </instancedMesh>
    </group>
  );
}

function MeshOrbitControls({ held }: { held: boolean }) {
  const { camera, gl } = useThree();
  const controlsRef = useRef<OrbitControls | null>(null);
  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.55;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.minDistance = 6;
    controls.maxDistance = 34;
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    };
    controlsRef.current = controls;
    return () => {
      controls.dispose();
      controlsRef.current = null;
    };
  }, [camera, gl.domElement]);

  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = !held;
    }
  }, [held]);

  useFrame(() => {
    controlsRef.current?.update();
  });

  return null;
}

function meshBasePositions(graph: GraphPayload, pIn: number) {
  const xs = graph.nodes.map((node) => node.x);
  const ys = graph.nodes.map((node) => node.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scale = (8.4 + pIn * 12) / Math.max(maxX - minX, maxY - minY, 1e-6);
  return graph.nodes.map((node) => {
    const x = (node.x - (minX + maxX) / 2) * scale;
    const y = (node.y - (minY + maxY) / 2) * scale;
    const radial = Math.hypot(x, y);
    const community = node.community / Math.max(1, graph.numCommunities - 1);
    const z =
      Math.sin(x * 1.6 + community * Math.PI * 2) * (1.0 + pIn * 2.8) +
      Math.cos(y * 1.2 - community * Math.PI) * (0.85 + pIn * 2.1) +
      Math.sin(radial * 1.7) * 0.75;
    return new THREE.Vector3(x, y, z);
  });
}

function ExplorationPulses({
  nodeById,
  probes,
  previousProbes,
  accent
}: {
  nodeById: Map<number, ScreenNode>;
  probes: Set<number>;
  previousProbes: Set<number>;
  accent: string;
}) {
  const current = [...probes].map((id) => nodeById.get(id)).filter((node): node is ScreenNode => Boolean(node));
  if (current.length === 0) return null;
  const previous = [...previousProbes].map((id) => nodeById.get(id)).filter((node): node is ScreenNode => Boolean(node));
  const from = previous.length ? centroid(previous) : { sx: WIDTH / 2, sy: HEIGHT / 2 };
  const to = centroid(current);
  const hasMotion = Math.hypot(to.sx - from.sx, to.sy - from.sy) > 6;
  return (
    <g className="pointer-events-none">
      <defs>
        <marker id="explore-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill={accent} opacity="0.85" />
        </marker>
      </defs>
      {hasMotion ? (
        <>
          <line
            className="explore-flow animate-pulse"
            x1={from.sx}
            y1={from.sy}
            x2={to.sx}
            y2={to.sy}
            stroke={accent}
            strokeWidth="2.1"
            strokeDasharray="7 9"
            opacity="0.65"
            markerEnd="url(#explore-arrow)"
          />
          <line
            x1={from.sx}
            y1={from.sy}
            x2={to.sx}
            y2={to.sy}
            stroke={accent}
            strokeWidth="8"
            opacity="0.06"
          />
        </>
      ) : null}
      {current.map((node) => (
        <line
          key={`probe-ray-${node.id}`}
          x1={to.sx}
          y1={to.sy}
          x2={node.sx}
          y2={node.sy}
          stroke={accent}
          strokeWidth="1"
          strokeDasharray="2 7"
          opacity="0.38"
        />
      ))}
      <SvgPulse cx={to.sx} cy={to.sy} r={10} stroke={accent} opacity={0.34} />
      <circle cx={to.sx} cy={to.sy} r="3" fill={accent} opacity="0.9" />
    </g>
  );
}

function DecisionLayer({
  nodeById,
  probes,
  previousProbes,
  nextProbes,
  highEffortNodes,
  accent,
  step,
  clock
}: {
  nodeById: Map<number, ScreenNode>;
  probes: Set<number>;
  previousProbes: Set<number>;
  nextProbes: Set<number>;
  highEffortNodes: ScreenNode[];
  accent: string;
  step: number;
  clock: number;
}) {
  const animationRef = useRef({ step, startedAt: clock });
  useEffect(() => {
    if (animationRef.current.step !== step) {
      animationRef.current = { step, startedAt: clock };
    }
  }, [clock, step]);
  const currentProbeNodes = [...probes].map((id) => nodeById.get(id)).filter((node): node is ScreenNode => Boolean(node));
  const previousProbeNodes = [...previousProbes].map((id) => nodeById.get(id)).filter((node): node is ScreenNode => Boolean(node));
  const nextProbeNodes = [...nextProbes].map((id) => nodeById.get(id)).filter((node): node is ScreenNode => Boolean(node));
  const currentCenter = currentProbeNodes.length ? centroid(currentProbeNodes) : null;
  const previousCenter = previousProbeNodes.length ? centroid(previousProbeNodes) : currentCenter;
  const decisionCenter = currentCenter
    ? movingDecisionCenter(previousCenter ?? currentCenter, currentCenter, clock - animationRef.current.startedAt)
    : null;
  const commitAngle =
    previousCenter && currentCenter
      ? Math.atan2(currentCenter.sy - previousCenter.sy, currentCenter.sx - previousCenter.sx)
      : 0;

  return (
    <g className="pointer-events-none">
      {previousProbeNodes.map((node) => (
        <circle
          key={`prev-probe-${node.id}`}
          cx={node.sx}
          cy={node.sy}
          r="9"
          fill="none"
          stroke="#ff9d2e"
          strokeWidth="1"
          strokeDasharray="2 5"
          opacity="0.34"
        />
      ))}
      {nextProbeNodes.map((node) => (
        <circle
          key={`next-probe-${node.id}`}
          className="animate-pulse"
          cx={node.sx}
          cy={node.sy}
          r="12"
          fill="none"
          stroke={accent}
          strokeWidth="1.3"
          strokeDasharray="4 6"
          opacity="0.5"
        />
      ))}
      {decisionCenter
        ? highEffortNodes
            .filter((node) => Math.hypot(node.sx - decisionCenter.sx, node.sy - decisionCenter.sy) < 175)
            .slice(0, 4)
            .map((node) => (
            <line
              key={`suppress-${node.id}`}
              x1={decisionCenter.sx}
              y1={decisionCenter.sy}
              x2={node.sx}
              y2={node.sy}
              stroke="#7c8cff"
              strokeWidth="1"
              strokeDasharray="5 8"
              opacity="0.22"
            />
          ))
        : null}
      {decisionCenter ? (
        <>
          <circle
            cx={decisionCenter.sx}
            cy={decisionCenter.sy}
            r="35"
            fill="#030507"
            opacity="0.72"
          />
          <circle
            cx={decisionCenter.sx}
            cy={decisionCenter.sy}
            r="30"
            fill={accent}
            opacity="0.13"
          />
          <SvgPulse cx={decisionCenter.sx} cy={decisionCenter.sy} r={28} stroke={accent} opacity={0.48} />
          <circle
            cx={decisionCenter.sx}
            cy={decisionCenter.sy}
            r="12"
            fill="#030507"
            opacity="0.92"
          />
          <circle
            cx={decisionCenter.sx}
            cy={decisionCenter.sy}
            r="12"
            fill={accent}
            opacity="0.2"
          />
          <circle
            cx={decisionCenter.sx}
            cy={decisionCenter.sy}
            r="13"
            fill="none"
            stroke={accent}
            strokeWidth="3"
            opacity="0.95"
          />
          <path
            d="M 0 -11 L 16 0 L 0 11 L 4 0 Z"
            transform={`translate(${decisionCenter.sx} ${decisionCenter.sy}) rotate(${(commitAngle * 180) / Math.PI})`}
            fill={accent}
            opacity="0.95"
          />
          <text
            x={decisionCenter.sx + 26}
            y={decisionCenter.sy - 28}
            fill="#030507"
            stroke="#030507"
            strokeWidth="5"
            paintOrder="stroke"
            fontFamily="monospace"
            fontSize="16"
            fontWeight="800"
            letterSpacing="3"
          >
            DECISION
          </text>
          <text
            x={decisionCenter.sx + 26}
            y={decisionCenter.sy - 28}
            fill={accent}
            fontFamily="monospace"
            fontSize="16"
            fontWeight="800"
            letterSpacing="3"
          >
            DECISION
          </text>
        </>
      ) : null}
    </g>
  );
}

function NodeInspector({
  node,
  stats,
  accent,
  onClose
}: {
  node: ScreenNode;
  stats: ReturnType<typeof nodeStats>;
  accent: string;
  onClose: () => void;
}) {
  return (
    <div className="absolute bottom-5 right-5 z-10 w-72 border border-white/10 bg-carbon-950/88 p-4 font-mono text-[11px] uppercase tracking-[0.16em] text-white/58 backdrop-blur">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div style={{ color: accent }}>node {node.id}</div>
          <div className="mt-1 text-white/38">community C{node.community}</div>
        </div>
        <button className="text-white/35 hover:text-white" onClick={onClose}>close</button>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2">
        <InspectorMetric label="perturb" value={stats.perturbation.toFixed(3)} color="#ff4d5e" />
        <InspectorMetric label="policy err" value={stats.error.toFixed(3)} color="#ffffff" />
        <InspectorMetric label="effort" value={stats.effort.toFixed(3)} color="#7c8cff" />
        <InspectorMetric label="state" value={stats.state} color={stats.isProbe ? "#ffffff" : stats.isObserved ? "#20d7ff" : "#64748b"} />
        <InspectorMetric label="neural err" value={stats.neuralError.toFixed(3)} color="#20d7ff" />
        <InspectorMetric label="greedy err" value={stats.greedyError.toFixed(3)} color="#ff9d2e" />
        <InspectorMetric label="dominance" value={stats.dominance.toFixed(3)} color={stats.dominance > 0 ? "#57f287" : "#64748b"} />
        <InspectorMetric label="community" value={`C${node.community}`} color="#ffffff" />
      </div>
    </div>
  );
}

function InspectorMetric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="border border-white/10 bg-white/[0.035] px-3 py-2">
      <div className="text-[9px] tracking-[0.18em] text-white/34">{label}</div>
      <div className="mt-1 text-sm text-white" style={{ color }}>{value}</div>
    </div>
  );
}

function nodeStats(
  nodeId: number,
  rollout: RolloutPayload | null,
  reference: RolloutPayload | null,
  neural: RolloutPayload | null,
  greedy: RolloutPayload | null,
  step: number,
  observed: Set<number>,
  probes: Set<number>
) {
  const control = rollout?.controlMagnitude?.[Math.max(0, step - 1)] ?? [];
  const error = rollout?.nodeError[step] ?? rollout?.nodeError[0] ?? [];
  const neuralError = neural?.nodeError[step] ?? [];
  const greedyError = greedy?.nodeError[step] ?? [];
  const perturbation = reference?.initialPerturbation ?? rollout?.initialPerturbation ?? [];
  const isProbe = probes.has(nodeId);
  const isObserved = observed.has(nodeId);
  const neuralValue = neuralError[nodeId] ?? 0;
  const greedyValue = greedyError[nodeId] ?? 0;
  return {
    perturbation: perturbation[nodeId] ?? 0,
    error: error[nodeId] ?? 0,
    neuralError: neuralValue,
    greedyError: greedyValue,
    dominance: Math.max(0, greedyValue - neuralValue),
    effort: control[nodeId] ?? 0,
    isProbe,
    isObserved,
    state: isProbe ? "probe" : isObserved ? "seen" : "fog"
  };
}

function healingProgress(
  nodeId: number,
  neural: RolloutPayload | null,
  sensing: RolloutPayload | null,
  initialNeuralError: number[],
  step: number
) {
  const initial = initialNeuralError[nodeId] ?? 0;
  if (!neural?.nodeError.length || initial <= 1e-9) {
    return { progress: 0, normalized: false };
  }
  const latest = Math.min(step, neural.nodeError.length - 1);
  let bestProgress = 0;
  let normalized = false;
  for (let idx = 0; idx <= latest; idx += 1) {
    const current = neural.nodeError[idx]?.[nodeId] ?? initial;
    const ratio = current / initial;
    bestProgress = Math.max(bestProgress, clamp((1 - ratio) / 0.62, 0, 1));
    normalized = normalized || ratio <= 0.38;
  }
  const contact = sensingContactProgress(nodeId, sensing, step);
  const progress = bestProgress * contact.progress;
  return { progress, normalized: normalized && contact.progress >= 0.86 };
}

function sensingContactProgress(nodeId: number, sensing: RolloutPayload | null, step: number) {
  if (!sensing) return { progress: 0, direct: false };
  const latest = Math.max(0, step - 1);
  let firstObserved: number | null = null;
  let firstProbe: number | null = null;
  for (let idx = 0; idx <= latest; idx += 1) {
    if (firstObserved === null && (sensing.observedNodes[idx] ?? []).includes(nodeId)) {
      firstObserved = idx;
    }
    if (firstProbe === null && (sensing.probeNodes[idx] ?? []).includes(nodeId)) {
      firstProbe = idx;
    }
  }
  const observedRamp =
    firstObserved === null ? 0 : Math.min(0.86, 0.42 + Math.max(0, latest - firstObserved) * 0.16);
  const probeRamp =
    firstProbe === null ? 0 : Math.min(1, 0.7 + Math.max(0, latest - firstProbe) * 0.18);
  const progress = Math.max(observedRamp, probeRamp);
  return { progress, direct: probeRamp >= 1 };
}

function animateNode(
  node: ScreenNode,
  clock: number,
  viewMode: ViewMode,
  offset: { x: number; y: number } | undefined,
  lambdaSheaf: number,
  pIn: number,
  numNodes: number
): ScreenNode {
  const smallGraphDamping = numNodes > 0 && numNodes <= 64 ? 0.38 : 1;
  const intensity = (viewMode === "controller" ? 2.7 : 1.35) * (0.75 + lambdaSheaf * 2.5) * smallGraphDamping;
  const cohesion = 1 + pIn * 1.8;
  const phase = node.id * 1.731 + node.community * 0.417;
  return {
    ...node,
    sx:
      node.sx +
      (offset?.x ?? 0) +
      Math.sin(clock * 1.4 + phase) * intensity +
      Math.sin(clock * 0.37 + phase * 2) * 0.7 * cohesion,
    sy:
      node.sy +
      (offset?.y ?? 0) +
      Math.cos(clock * 1.15 + phase * 0.8) * intensity +
      Math.cos(clock * 0.29 + phase) * 0.7 * cohesion
  };
}

function relaxNodeOffsets(offsets: NodeOffsets, grabbedNodeId: number | null) {
  let changed = false;
  const next: NodeOffsets = {};
  for (const [idText, offset] of Object.entries(offsets)) {
    const id = Number(idText);
    if (id === grabbedNodeId) {
      next[id] = offset;
      continue;
    }
    const relaxed = { x: offset.x * 0.88, y: offset.y * 0.88 };
    if (Math.hypot(relaxed.x, relaxed.y) > 0.35) {
      next[id] = relaxed;
    }
    changed = true;
  }
  return changed ? next : offsets;
}

function clientToMap(
  event: ReactPointerEvent<SVGSVGElement>,
  svg: SVGSVGElement | null,
  transform: ViewTransform
) {
  const rect = svg?.getBoundingClientRect();
  if (!rect) return { x: WIDTH / 2, y: HEIGHT / 2 };
  const viewX = ((event.clientX - rect.left) / rect.width) * WIDTH;
  const viewY = ((event.clientY - rect.top) / rect.height) * HEIGHT;
  return {
    x: (viewX - transform.x) / transform.k,
    y: (viewY - transform.y) / transform.k
  };
}

function centroid(nodes: ScreenNode[]) {
  return {
    sx: nodes.reduce((sum, node) => sum + node.sx, 0) / nodes.length,
    sy: nodes.reduce((sum, node) => sum + node.sy, 0) / nodes.length
  };
}

function movingDecisionCenter(
  from: { sx: number; sy: number },
  to: { sx: number; sy: number },
  elapsed: number
) {
  const progress = clamp(elapsed / 0.78, 0, 1);
  const smooth = 1 - Math.pow(1 - progress, 3);
  const dx = to.sx - from.sx;
  const dy = to.sy - from.sy;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const arc = Math.sin(progress * Math.PI) * Math.min(52, Math.max(16, distance * 0.2));
  return {
    sx: from.sx + dx * smooth + (-dy / distance) * arc,
    sy: from.sy + dy * smooth + (dx / distance) * arc
  };
}

function projectNodes(nodes: GraphNode[], pIn: number, pOut: number, lambdaSheaf: number): ScreenNode[] {
  if (nodes.length > 0 && nodes.length <= 64) {
    return projectSmallGraphNodes(nodes, pIn, pOut, lambdaSheaf);
  }
  const xs = nodes.map((node) => node.x);
  const ys = nodes.map((node) => node.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scale = Math.min((WIDTH - PAD * 2) / (maxX - minX), (HEIGHT - PAD * 2) / (maxY - minY));
  const xOffset = (WIDTH - (maxX - minX) * scale) / 2;
  const yOffset = (HEIGHT - (maxY - minY) * scale) / 2;
  const centerX = WIDTH / 2;
  const centerY = HEIGHT / 2;
  const mess = Math.min(1, pOut / 0.06);
  const cohesion = Math.max(0, Math.min(1, pIn / 0.35));
  const fold = Math.min(1, lambdaSheaf / 0.6);
  return nodes.map((node) => {
    const baseX = xOffset + (node.x - minX) * scale;
    const baseY = yOffset + (node.y - minY) * scale;
    const dx = baseX - centerX;
    const dy = baseY - centerY;
    const angle = Math.atan2(dy, dx);
    const radius = Math.hypot(dx, dy);
    const communityPhase = node.community * 1.73;
    const chaosX =
      Math.sin(node.id * 1.91 + communityPhase) * mess * 72 +
      Math.cos(node.id * 0.37 + fold * 4) * fold * 36;
    const chaosY =
      Math.cos(node.id * 1.47 - communityPhase) * mess * 72 +
      Math.sin(node.id * 0.53 + fold * 3) * fold * 36;
    const cohesionPull = 1 - cohesion * 0.22;
    const bridgeSpiral = mess * 0.42 + fold * 0.18;
    return {
      ...node,
      sx: centerX + Math.cos(angle + bridgeSpiral * Math.sin(node.community)) * radius * cohesionPull + chaosX,
      sy: centerY + Math.sin(angle + bridgeSpiral * Math.cos(node.community)) * radius * cohesionPull + chaosY
    };
  });
}

function projectSmallGraphNodes(nodes: GraphNode[], pIn: number, pOut: number, lambdaSheaf: number): ScreenNode[] {
  const groups = new Map<number, GraphNode[]>();
  for (const node of nodes) {
    groups.set(node.community, [...(groups.get(node.community) ?? []), node]);
  }
  const communityIds = [...groups.keys()].sort((left, right) => left - right);
  const centerX = WIDTH / 2;
  const centerY = HEIGHT / 2 + 18;
  const mess = Math.min(1, pOut / 0.06);
  const fold = Math.min(1, lambdaSheaf / 0.6);
  const cohesion = Math.max(0, Math.min(1, pIn / 0.35));
  const orbitRadius = 202 + mess * 26 - cohesion * 16;
  const result = new Map<number, ScreenNode>();

  communityIds.forEach((communityId, communityIndex) => {
    const group = [...(groups.get(communityId) ?? [])].sort((left, right) => left.id - right.id);
    const communityAngle = -Math.PI / 2 + (communityIndex / Math.max(1, communityIds.length)) * Math.PI * 2;
    const cx = centerX + Math.cos(communityAngle) * orbitRadius;
    const cy = centerY + Math.sin(communityAngle) * orbitRadius * 0.74;
    const rows = Math.ceil(Math.sqrt(group.length));
    const spacing = 31 - Math.min(8, group.length * 0.32);

    group.forEach((node, index) => {
      const col = index % rows;
      const row = Math.floor(index / rows);
      const localX = (col - (rows - 1) / 2) * spacing;
      const localY = (row - (Math.ceil(group.length / rows) - 1) / 2) * spacing;
      const rotate = communityAngle + Math.PI / 6;
      const structuredX = localX * Math.cos(rotate) - localY * Math.sin(rotate);
      const structuredY = localX * Math.sin(rotate) + localY * Math.cos(rotate);
      const foldX = Math.sin(node.id * 1.7 + communityId) * (8 + mess * 12 + fold * 8);
      const foldY = Math.cos(node.id * 1.3 - communityId) * (6 + mess * 10 + fold * 8);
      result.set(node.id, {
        ...node,
        sx: cx + structuredX + foldX,
        sy: cy + structuredY + foldY
      });
    });
  });

  return nodes.map((node) => result.get(node.id) ?? { ...node, sx: centerX, sy: centerY });
}

function communitySummaries(nodes: ScreenNode[]) {
  const groups = new Map<number, ScreenNode[]>();
  for (const node of nodes) {
    groups.set(node.community, [...(groups.get(node.community) ?? []), node]);
  }
  return [...groups.entries()].map(([id, group]) => {
    const cx = group.reduce((sum, node) => sum + node.sx, 0) / group.length;
    const cy = group.reduce((sum, node) => sum + node.sy, 0) / group.length;
    const radius =
      Math.max(...group.map((node) => Math.hypot(node.sx - cx, node.sy - cy))) + 18;
    return { id, cx, cy, radius };
  });
}

function selectActiveEdges(
  edges: GraphEdge[],
  observed: Set<number>,
  probes: Set<number>,
  crackEdges: GraphEdge[]
) {
  const crackKeys = new Set(crackEdges.map((edge) => edgeKey(edge)));
  const probeEdges = edges
    .filter((edge) => !crackKeys.has(edgeKey(edge)) && (probes.has(edge.source) || probes.has(edge.target)))
    .sort((left, right) => edgeScore(right) - edgeScore(left))
    .slice(0, 70);
  const observedEdges = edges
    .filter((edge) => !crackKeys.has(edgeKey(edge)) && observed.has(edge.source) && observed.has(edge.target))
    .sort((left, right) => edgeScore(right) - edgeScore(left))
    .slice(0, 120);
  const seen = new Set<string>();
  return [...probeEdges, ...observedEdges].filter((edge) => {
    const key = edgeKey(edge);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function selectContextEdges(edges: GraphEdge[], pOut: number, numNodes: number) {
  if (numNodes > 0 && numNodes <= 64) {
    const sameCommunity = edges
      .filter((edge) => edge.sameCommunity)
      .sort((left, right) => edgeScore(right) - edgeScore(left))
      .slice(0, 72);
    const bridgeLimit = Math.round(8 + Math.min(1, pOut / 0.06) * 18);
    const bridges = edges
      .filter((edge) => !edge.sameCommunity)
      .sort((left, right) => edgeScore(right) - edgeScore(left))
      .slice(0, bridgeLimit);
    return [...sameCommunity, ...bridges];
  }
  const sameCommunity = edges
    .filter((edge) => edge.sameCommunity)
    .sort((left, right) => edgeScore(right) - edgeScore(left))
    .slice(0, 90);
  const bridgeLimit = Math.round(35 + Math.min(1, pOut / 0.06) * 65);
  const bridges = edges
    .filter((edge) => !edge.sameCommunity)
    .sort((left, right) => edgeScore(right) - edgeScore(left))
    .slice(0, bridgeLimit);
  return [...sameCommunity, ...bridges];
}

function selectCrackEdges(edges: GraphEdge[], numNodes: number, lambdaSheaf: number) {
  const candidates = edges.filter((edge) => !edge.sameCommunity || edge.distortion > 0.08);
  if (numNodes > 0 && numNodes <= 64) {
    const limit = Math.max(10, Math.ceil(candidates.length * (0.42 + Math.min(0.2, lambdaSheaf * 0.18))));
    return candidates.sort((left, right) => tensionScore(right) - tensionScore(left)).slice(0, limit);
  }
  const limit = Math.max(1, Math.ceil(candidates.length * (0.08 + Math.min(0.22, lambdaSheaf * 0.22))));
  return candidates.sort((left, right) => tensionScore(right) - tensionScore(left)).slice(0, limit);
}

function edgeScore(edge: GraphEdge) {
  return edge.communityGap * 3 + edge.distortion * 5 + (edge.sameCommunity ? 0 : 1);
}

function tensionScore(edge: GraphEdge) {
  return edge.communityGap * 4 + edge.distortion * 8 + edge.friction * 3 + (edge.sameCommunity ? 0 : 2);
}

function edgeKey(edge: GraphEdge) {
  return `${edge.source}:${edge.target}`;
}

function quantile(values: number[], q: number) {
  if (values.length === 0) return Number.POSITIVE_INFINITY;
  const sorted = [...values].sort((left, right) => left - right);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
