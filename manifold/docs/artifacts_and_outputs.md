# Artifacts and Outputs

Training and evaluation artifacts are saved under `outputs/`. The current primary demo directory is:

```text
outputs/demo-512-c12-final
```

## Run Directory Layout

```text
outputs/demo-512-c12-final/
  best.json
  config.json
  startup.json
  graph.pt
  graph_layout.json
  eval_set.pt
  logs/
    metrics.jsonl
    eval.jsonl
  checkpoints/
    best.pt
    last.pt
  rollouts/
  closed-loop-eval-20-t8/
  closed-loop-eval-48-t8/
  chatgpt55-high-comparison/
```

## Core Files

| File | Meaning |
|---|---|
| `config.json` | Full config used for the saved run. |
| `startup.json` | Startup report: selected device, MPS availability, graph size, sparse backend status, seed, RK4 steps. |
| `best.json` | Best early-stopping score and epoch. |
| `graph.pt` | Saved graph tensors: `edge_index`, `edge_attr`, labels, community centers, graph metadata. |
| `graph_layout.json` | Node positions used by graph visualization plots. |
| `eval_set.pt` | Saved paired evaluation cases for reproducible comparisons. |
| `checkpoints/best.pt` | Best model checkpoint for reuse. |
| `checkpoints/last.pt` | Last checkpoint from training. |

## Logs

### `logs/metrics.jsonl`

One JSON object per epoch. Common fields:

- `controller_loss`
- `source_loss`
- `post_control_perturbation_auc`
- `trajectory_mse`
- `final_belief_mse`
- `control_energy`
- `control_target_mse`
- `observed_fraction`
- `source_auc`
- `perturbation_norm`
- `boundary_nodes_used`
- `source_frozen`

### `logs/eval.jsonl`

Periodic paired comparison rows. Each row includes per-policy metrics plus `neural_minus_greedy` deltas.

Important fields:

- `post_control_perturbation_auc`
- `trajectory_mse`
- `neural_vs_greedy_auc_improvement_pct`
- `neural_vs_greedy_mse_improvement_pct`
- `neural_vs_greedy_post_control_auc_improvement_pct`

## Closed-loop Rollouts

Closed-loop exports are saved under folders such as:

```text
closed-loop-eval-20-t8/
closed-loop-eval-48-t8/
```

Each policy rollout `.pt` contains:

| Key | Meaning |
|---|---|
| `policy` | Policy name. |
| `mode` | Closed-loop mode marker. |
| `run_idx` | Rollout index. |
| `pred_trajectory` | Predicted/recovered state trajectory. |
| `target_trajectory` | Clean target trajectory used for metric computation. |
| `initial_clean_state` | Clean initial latent state. |
| `initial_belief` | Perturbed/corrupted initial belief state. |
| `controls` | Control tensor per closed-loop step. |
| `corrected_states` | Belief-corrected states after sensing. |
| `probe_nodes` | Probe nodes selected at each step. |
| `observed_masks` | k-hop visibility masks. |
| `labels` | Community labels. |
| `times` | Rollout times. |
| `node_error` | Per-node error over time. |
| `mean_error` | Mean node error over time. |

## ChatGPT Baseline Artifacts

Saved under:

```text
outputs/demo-512-c12-final/chatgpt55-high-comparison/
```

Important files:

| Path | Meaning |
|---|---|
| `prompts/` | Exact prompt sent for each ChatGPT control step. |
| `raw_responses/` | Raw `codex exec` transcripts. |
| `chatgpt_rollouts/` | Plot-compatible ChatGPT rollout tensors. |
| `chatgpt_rollouts_manifest.json` | Summary of exported ChatGPT rollouts. |
| `comparison_summary.json` | Metrics against the selected saved baselines. |
| `timing.json` | Wall-clock timing per rollout and step. |

## Why `.pt` Files?

The `.pt` files store tensors with exact dtype/shape preservation. This avoids lossy JSON/CSV conversion for trajectories, controls, masks, and graph tensors. The visualization module loads these directly with `torch.load(..., map_location="cpu")`.

