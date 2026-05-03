import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY } from "d3-force";
import type { GraphEdge, GraphNode } from "../types";

interface WorkerRequest {
  nodes: GraphNode[];
  edges: GraphEdge[];
  pIn: number;
  pOut: number;
}

interface SimNode {
  id: number;
  x: number;
  y: number;
  community: number;
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const { nodes, edges, pIn, pOut } = event.data;
  const communityPull = Math.max(0.02, Math.min(0.32, pIn * 3.5));
  const bridgeCharge = Math.max(-120, -22 - pOut * 9000);
  const simNodes: SimNode[] = nodes.map((node) => ({
    id: node.id,
    x: node.x * 46,
    y: node.y * 46,
    community: node.community
  }));
  const simEdges = edges.map((edge) => ({
    source: edge.source,
    target: edge.target,
    strength: edge.sameCommunity ? communityPull : Math.max(0.004, pOut * 7),
    distance: edge.sameCommunity ? 16 : 42 + edge.communityGap * 34
  }));

  const simulation = forceSimulation(simNodes)
    .force(
      "link",
      forceLink<SimNode, (typeof simEdges)[number]>(simEdges)
        .id((node) => node.id)
        .strength((edge) => edge.strength)
        .distance((edge) => edge.distance)
    )
    .force("charge", forceManyBody().strength(bridgeCharge))
    .force("collide", forceCollide(6))
    .force("x", forceX(0).strength(0.018))
    .force("y", forceY(0).strength(0.018))
    .stop();

  for (let idx = 0; idx < 180; idx += 1) {
    simulation.tick();
  }

  self.postMessage({
    positions: simNodes.map((node) => ({
      id: node.id,
      x: Number((node.x / 46).toFixed(5)),
      y: Number((node.y / 46).toFixed(5))
    }))
  });
};

