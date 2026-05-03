import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { GraphPayload, RolloutPayload } from "../types";

interface ManifoldSceneProps {
  graph: GraphPayload | null;
  rollout: RolloutPayload | null;
  step: number;
  playing: boolean;
  lambdaSheaf: number;
  pIn: number;
  pOut: number;
}

interface WorkerPosition {
  id: number;
  x: number;
  y: number;
}

export function ManifoldScene({ graph, rollout, step, playing, lambdaSheaf, pIn, pOut }: ManifoldSceneProps) {
  const [workerPositions, setWorkerPositions] = useState<Map<number, WorkerPosition>>(new Map());

  useEffect(() => {
    if (!graph) return;
    const worker = new Worker(new URL("../workers/layoutWorker.ts", import.meta.url), {
      type: "module"
    });
    worker.onmessage = (event: MessageEvent<{ positions: WorkerPosition[] }>) => {
      setWorkerPositions(new Map(event.data.positions.map((position) => [position.id, position])));
    };
    worker.postMessage({ nodes: graph.nodes, edges: graph.edges, pIn, pOut });
    return () => worker.terminate();
  }, [graph, pIn, pOut]);

  return (
    <div className="relative h-full overflow-hidden border-x border-white/10 bg-black/20 scanline">
      <div className="grid-floor pointer-events-none absolute inset-x-0 bottom-0 h-1/2 opacity-50" />
      <div className="pointer-events-none absolute left-5 top-5 z-10 font-mono text-[11px] uppercase tracking-[0.24em] text-cyanOps/80">
        sheaf ode manifold / recurrent sensing envelope
      </div>
      <Canvas
        camera={{ position: [0, 0, 15], fov: 48 }}
        dpr={[1, 1.7]}
        gl={{ preserveDrawingBuffer: true, antialias: true }}
      >
        <color attach="background" args={["#030507"]} />
        <fog attach="fog" args={["#030507", 9, 25]} />
        <ambientLight intensity={0.35} />
        <pointLight position={[4, 6, 8]} intensity={18} color="#20d7ff" />
        <pointLight position={[-6, -4, 6]} intensity={10} color="#ff9d2e" />
        {graph && rollout ? (
          <GraphCloud
            graph={graph}
            rollout={rollout}
            workerPositions={workerPositions}
            step={step}
            playing={playing}
            lambdaSheaf={lambdaSheaf}
            pOut={pOut}
          />
        ) : null}
      </Canvas>
      <div className="pointer-events-none absolute right-5 top-5 z-10 w-64 border border-white/10 bg-carbon-950/70 p-3 font-mono text-[11px] uppercase tracking-[0.16em] text-white/55 backdrop-blur">
        <div className="mb-2 text-cyanOps">legend</div>
        <LegendItem color="bg-white" label="probe nodes" />
        <LegendItem color="bg-cyanOps" label="observed k-hop field" />
        <LegendItem color="bg-amberOps" label="sheaf tension bridges" />
        <LegendItem color="bg-slate-500" label="unseen nodes" />
      </div>
      <div className="pointer-events-none absolute bottom-20 left-5 z-10 max-w-md border border-white/10 bg-carbon-950/70 p-3 font-mono text-[11px] uppercase tracking-[0.15em] text-white/50 backdrop-blur">
        showing active sensed neighborhood + ranked tension bridges only
      </div>
    </div>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="mb-1 flex items-center gap-2 last:mb-0">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      <span>{label}</span>
    </div>
  );
}

function edgeScore(edge: GraphPayload["edges"][number]) {
  return edge.communityGap * 2.5 + edge.distortion * 4 + (edge.sameCommunity ? 0 : 1);
}

function GraphCloud({
  graph,
  rollout,
  workerPositions,
  step,
  playing,
  lambdaSheaf,
  pOut
}: {
  graph: GraphPayload;
  rollout: RolloutPayload;
  workerPositions: Map<number, WorkerPosition>;
  step: number;
  playing: boolean;
  lambdaSheaf: number;
  pOut: number;
}) {
  const pointsRef = useRef<THREE.Points>(null);
  const edgesRef = useRef<THREE.LineSegments>(null);
  const time = useRef(0);
  const observed = useMemo(() => new Set(rollout.observedNodes[Math.max(0, step - 1)] ?? []), [rollout, step]);
  const probes = useMemo(() => new Set(rollout.probeNodes[Math.max(0, step - 1)] ?? []), [rollout, step]);
  const displayEdges = useMemo(() => {
    const activeEdges = graph.edges
      .filter((edge) => {
        const sourceActive = observed.has(edge.source) || probes.has(edge.source);
        const targetActive = observed.has(edge.target) || probes.has(edge.target);
        return (sourceActive && targetActive) || probes.has(edge.source) || probes.has(edge.target);
      })
      .sort((left, right) => edgeScore(right) - edgeScore(left))
      .slice(0, 260);
    const activeKeys = new Set(activeEdges.map((edge) => `${edge.source}:${edge.target}`));
    const bridgeEdges = graph.edges
      .filter((edge) => !activeKeys.has(`${edge.source}:${edge.target}`) && (!edge.sameCommunity || edge.distortion > 0.1))
      .sort((left, right) => edgeScore(right) - edgeScore(left))
      .slice(0, 45);
    return [...activeEdges, ...bridgeEdges];
  }, [graph.edges, observed, probes]);

  const positions = useMemo(() => {
    const scale = 0.78;
    return graph.nodes.map((node) => {
      const moved = workerPositions.get(node.id);
      const x = (moved?.x ?? node.x) * scale;
      const y = (moved?.y ?? node.y) * scale;
      return new THREE.Vector3(x, y, 0);
    });
  }, [graph.nodes, workerPositions]);

  const pointGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(graph.nodes.length * 3), 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(graph.nodes.length * 3), 3));
    return geometry;
  }, [graph.nodes.length]);

  const edgeGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(displayEdges.length * 2 * 3), 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(displayEdges.length * 2 * 3), 3));
    return geometry;
  }, [displayEdges.length]);

  useFrame((_, delta) => {
    time.current += delta;
    const pointPositions = pointGeometry.getAttribute("position") as THREE.BufferAttribute;
    const pointColors = pointGeometry.getAttribute("color") as THREE.BufferAttribute;
    const edgePositions = edgeGeometry.getAttribute("position") as THREE.BufferAttribute;
    const edgeColors = edgeGeometry.getAttribute("color") as THREE.BufferAttribute;
    const nodeError = rollout.nodeError[step] ?? rollout.nodeError[0] ?? [];
    const phase = playing ? time.current : step * 0.35;

    for (const node of graph.nodes) {
      const base = positions[node.id] ?? new THREE.Vector3();
      const error = nodeError[node.id] ?? 0;
      const fold =
        Math.sin(base.x * 1.4 + phase * 0.18) * Math.cos(base.y * 1.1) * lambdaSheaf * 4.8 +
        error * 0.9;
      const active = observed.has(node.id);
      const probe = probes.has(node.id);
      pointPositions.setXYZ(node.id, base.x, base.y, fold);
      const color = probe
        ? new THREE.Color("#ffffff")
        : active
          ? new THREE.Color("#20d7ff")
          : new THREE.Color("#304455");
      if (!active && !probe) color.multiplyScalar(0.24);
      if (error > 0.8) color.lerp(new THREE.Color("#ff9d2e"), 0.55);
      pointColors.setXYZ(node.id, color.r, color.g, color.b);
    }

    displayEdges.forEach((edge, edgeIdx) => {
      const source = graph.nodes[edge.source];
      const target = graph.nodes[edge.target];
      const sourceBase = positions[source.id] ?? new THREE.Vector3();
      const targetBase = positions[target.id] ?? new THREE.Vector3();
      const fighting = !edge.sameCommunity || edge.distortion > 0.08;
      const pulse = fighting ? Math.sin(phase * 2.2 + edgeIdx * 0.13) * lambdaSheaf * 0.28 : 0;
      const sourceError = nodeError[source.id] ?? 0;
      const targetError = nodeError[target.id] ?? 0;
      const sourceZ = Math.sin(sourceBase.x * 1.4 + phase * 0.18) * lambdaSheaf * 4.8 + sourceError * 0.9 + pulse;
      const targetZ = Math.sin(targetBase.x * 1.4 + phase * 0.18) * lambdaSheaf * 4.8 + targetError * 0.9 - pulse;
      const offset = edgeIdx * 2;
      edgePositions.setXYZ(offset, sourceBase.x, sourceBase.y, sourceZ);
      edgePositions.setXYZ(offset + 1, targetBase.x, targetBase.y, targetZ);
      const visible =
        (observed.has(source.id) && observed.has(target.id)) ||
        probes.has(source.id) ||
        probes.has(target.id);
      const color = fighting ? new THREE.Color("#ff9d2e") : new THREE.Color("#20d7ff");
      const bridgePressure = Math.min(1, Math.max(0.04, pOut / 0.06));
      const hiddenScale = fighting ? 0.045 + bridgePressure * 0.12 : 0.035;
      color.multiplyScalar(visible ? 0.95 : hiddenScale);
      edgeColors.setXYZ(offset, color.r, color.g, color.b);
      edgeColors.setXYZ(offset + 1, color.r, color.g, color.b);
    });

    pointPositions.needsUpdate = true;
    pointColors.needsUpdate = true;
    edgePositions.needsUpdate = true;
    edgeColors.needsUpdate = true;
  });

  return (
    <group rotation={[-0.72, 0.08, 0]}>
      <lineSegments ref={edgesRef} geometry={edgeGeometry}>
        <lineBasicMaterial vertexColors transparent opacity={0.42} blending={THREE.AdditiveBlending} />
      </lineSegments>
      <points ref={pointsRef} geometry={pointGeometry}>
        <pointsMaterial size={0.06 + lambdaSheaf * 0.1} vertexColors transparent opacity={0.98} blending={THREE.AdditiveBlending} depthWrite={false} />
      </points>
    </group>
  );
}
