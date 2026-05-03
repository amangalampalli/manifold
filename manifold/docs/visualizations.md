# Visualizations

The plotting module is:

```text
manifold/viz/plots.py
```

Run it with:

```bash
conda run -n manifold python -m manifold.viz \
  --run-dir outputs/demo-512-c12-final \
  --fig-dir manifold/docs/figs
```

It writes PNG files and a manifest:

```text
manifold/docs/figs/plot_manifest.json
```

## Main Plot Groups

### Selected Demo Rollouts

These use human-readable names:

```text
48_node_graph_rollout_014_*.png
512_node_graph_rollout_015_*.png
```

| Plot | Meaning |
|---|---|
| `*_auc_curves.png` | Instantaneous mean error and cumulative AUC over time. |
| `*_real_error_values.png` | AUC and trajectory MSE bars for each policy. |
| `*_node_advantage_histograms.png` | Distribution of node-step error reduction by neural vs baselines. |
| `*_error_histograms.png` | Final/all-step node-error distributions. |
| `*_token_control_efficiency.png` | Active node-step coverage and mean control density. |
| `*_layout_error.png` | Graph layout colored by neural final node error. |
| `*_control_sensing.png` | Control norm and observed fraction over time. |
| `*_policy_scores.png` | Compact policy score bars. |
| `*_mean_error.png` | Mean error curve over rollout steps. |

### Cross-run Summaries

| Plot | Meaning |
|---|---|
| `neural_vs_greedy_real_error_values.png` | Clean neural vs greedy AUC/MSE values across selected graph sizes. |
| `selected_runs_control_coverage_values.png` | Active node-step coverage for neural and ChatGPT across selected graph sizes. |
| `selected_runs_neural_vs_greedy.png` | Percent improvement over greedy from selected runs. |

### Training and Eval Diagnostics

These are useful for debugging and internal review:

| Plot | Meaning |
|---|---|
| `training_losses.png` | Controller and source losses. |
| `training_recovery_metrics.png` | Recovery/AUC/MSE during training. |
| `training_control_metrics.png` | Control energy, target control MSE, observed fraction. |
| `training_source_metrics.png` | Source AUC, perturbation norm, source freeze behavior. |
| `eval_neural_vs_greedy_improvement.png` | Eval improvement percentages during training. |
| `eval_policy_post_control_auc.png` | Eval AUC by policy. |
| `eval_policy_trajectory_mse.png` | Eval MSE by policy. |

### Run Selection / Graph Diagnostics

| Plot | Meaning |
|---|---|
| `closed_loop_auc_mse_tradeoff.png` | Scatter of run-level AUC vs MSE improvement. |
| `closed-loop-eval-*_top_run_improvements.png` | Top rollout candidates by improvement. |
| `graph_degree_and_communities.png` | Degree distribution and community sizes. |
| `artifact_storage_sizes.png` | Storage footprint of output folders. |
| `chatgpt_wallclock.png` | ChatGPT wall-clock timing from `timing.json`. |

## Recommended Editing Pattern

If a plot is too crowded for slides:

1. Prefer per-case plots over combined plots.
2. Show real values on the figure.
3. Put multiplier claims in slide text, not inside the chart.
4. Use the 512-node figures as the primary demo evidence.
5. Use 48-node figures only as a small proof-of-concept or UI-friendly example.

## Adding New Plots

Add new plot functions to `manifold/viz/plots.py`, call them from `plot_all`, and update this file with the output filename and interpretation.

Run:

```bash
conda run -n manifold python -m manifold.viz \
  --run-dir outputs/demo-512-c12-final \
  --fig-dir manifold/docs/figs

conda run -n manifold pytest -q
```

