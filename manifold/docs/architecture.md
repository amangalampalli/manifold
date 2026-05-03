# Architecture

The project is a runnable research prototype for graph signal stabilization under stochastic/interfering perturbations. It uses synthetic community graphs, neural sheaf operators, continuous-time rollout, active sensing, and minimax source/controller training.

## High-level Flow

```text
synthetic graph + latent trajectory
        |
        v
boundary / learned perturbation source
        |
        v
initial belief state
        |
        v
controller policy -> control tensor
        |
        v
NeuralSheafODE RK4 rollout
        |
        v
active sensing + belief correction
        |
        v
metrics, checkpoints, rollouts, plots
```

## Main Components

| Component | Files | Purpose |
|---|---|---|
| Config | `manifold/utils/config.py`, `configs/default.yaml` | Dataclass config and YAML defaults. |
| Device handling | `manifold/utils/devices.py` | Selects `auto`, `mps`, `cuda`, or `cpu`; `auto` prefers Apple MPS when available. |
| Synthetic data | `manifold/data/generation.py` | Creates community graphs, latent centers, labels, and trajectories. |
| Sheaf math | `manifold/math/sheaf.py` | Constructs edge restriction maps, sparse coboundary behavior, and sheaf Laplacian application. |
| ODE dynamics | `manifold/math/dynamics.py` | Implements continuous-time state update with RK4 rollout. |
| Sensing | `manifold/trainer/game/sensing.py` | Selects probe nodes, builds k-hop observation masks, and corrects belief states. |
| Policies | `manifold/trainer/support/policies.py`, `manifold/trainer/core/control.py` | Random, weak greedy, neural control, and source disturbance helpers. |
| Training/eval | `manifold/trainer/core/*.py` | Minimax training, checkpointing, eval, closed-loop export, ChatGPT baseline export. |
| CLI | `manifold/cli/*.py` | Commands for training, comparison, exports, and baselines. |
| Visualization | `manifold/viz/plots.py` | Reads saved artifacts and generates figures under `manifold/docs/figs`. |

## Neural Sheaf ODE

The state is node-local latent state `h` with default latent dimension `16`.

The dynamics are:

```text
dh/dt = f_theta(h, t, edge_index) - lambda_sheaf * L_s h + G(u)
```

Where:

- `f_theta` is the graph message passing block.
- `L_s h` is the sheaf consistency term.
- `G(u)` is a learned control projection.
- Integration uses fixed-step RK4 through `torchdiffeq`.

## Sheaf Operator

Each node has a stalk, and each edge has learned restriction maps that model boundary distortion/friction. The sheaf Laplacian is applied as a sparse operator equivalent to:

```text
L_s = delta.T @ delta
```

The implementation keeps sparse operations sparse and preserves selected device placement where supported. If a sparse backend fails, the code has fallback behavior for CPU-compatible sparse products.

## Active Sensing

At each closed-loop step:

1. The sensing policy chooses probe nodes under a fixed budget.
2. A k-hop observation mask is built around those probes.
3. The controller acts under partial visibility.
4. The model rolls one ODE window.
5. A learned gain corrects the belief:

```text
h_corr = h_pred + K(y - h_y)
```

## Policies

### Neural

The neural controller uses sheaf-aware features:

```text
[h, L_s h, ||h||, ||L_s h||]
```

It outputs structured control:

```text
control = -gate * h + residual
```

This makes the policy graph-aware and lets it act densely across node states.

### Greedy

The greedy baseline is intentionally weak. It acts on one high-energy node per step and is intended to expose the weakness of one-node repair under multi-node boundary perturbations.

### Random

The random baseline creates non-targeted control. It can touch many nodes, but it does not exploit graph/sheaf/community structure.

### ChatGPT 5.5 High

ChatGPT is evaluated as a fair partial-visibility controller through `codex exec`. It receives only the same visible runtime summary available to a controller and returns JSON decisions that are sanitized before being converted to control.

