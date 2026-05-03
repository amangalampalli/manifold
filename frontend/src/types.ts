export type Policy = "random" | "greedy" | "neural" | "chatgpt";

export interface HeadlineMetric {
  label: string;
  value: number;
  unit: string;
  description: string;
  polarity: "lower_is_better";
}

export interface PolicySummary {
  runs: number;
  final_mean_error: number;
  mean_error_auc: number;
}

export interface RunSummary {
  id: string;
  label: string;
  defaultRunIdx: number;
  artifactPath: string;
  policies: Partial<Record<Policy, PolicySummary>>;
  deltas: Record<string, number>;
  headlineMetrics: HeadlineMetric[];
  sensing: {
    budget: number | null;
    kHop: number | null;
    noiseStd: number | null;
  };
  model: {
    sheafLambda: number | null;
    latentDim: number | null;
  };
  graph: {
    numNodes: number | null;
    numCommunities: number | null;
    pIn: number | null;
    pOut: number | null;
  };
}

export interface GraphNode {
  id: number;
  community: number;
  x: number;
  y: number;
}

export interface GraphEdge {
  source: number;
  target: number;
  sameCommunity: boolean;
  friction: number;
  distortion: number;
  communityGap: number;
}

export interface GraphPayload {
  runId: string;
  numNodes: number;
  numCommunities: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface RolloutPayload {
  runId: string;
  policy: Policy;
  runIdx: number;
  times: number[];
  meanError: number[];
  nodeError: number[][];
  initialPerturbation: number[];
  stateProjection: number[][][];
  controlEnergy: number[];
  controlMagnitude: number[][];
  correctionEnergy: number[];
  residualEnergy: number[];
  probeNodes: number[][];
  observedNodes: number[][];
  finalMeanError: number;
  meanErrorAuc: number;
  modelLabel?: string;
  validDecisions?: number;
  chatgptDecisions?: Array<{
    selected_nodes: number[];
    damping_gain: number;
    laplacian_gain: number;
    center_pull_gain: number;
    confidence: number;
    rationale: string;
    valid: boolean;
    step_idx: number;
  }>;
}

export interface TelemetryLine {
  step: number;
  message: string;
}
