# Running Experiments

All commands assume the repository root:

```bash
cd /Users/aditya/Developer/manifold
```

Use the conda environment:

```bash
conda run -n manifold <command>
```

## Test Suite

```bash
conda run -n manifold pytest -q
```

## Basic Training

Run the default config:

```bash
conda run -n manifold python -m manifold.cli compare \
  --config configs/default.yaml \
  --device auto \
  --epochs 5 \
  --policies random greedy neural \
  --progress
```

The installed script entrypoint is also available:

```bash
conda run -n manifold manifold-train compare \
  --config configs/default.yaml \
  --device auto
```

## Demo-scale Training

The current demo run is built around a 512-node, 12-community graph:

```bash
conda run -n manifold python -m manifold.cli compare \
  --config configs/default.yaml \
  --device auto \
  --epochs 100 \
  --num-nodes 512 \
  --num-communities 12 \
  --p-in 0.04 \
  --p-out 0.0025 \
  --trajectory-steps 8 \
  --sensing-budget 24 \
  --eval-runs 2 \
  --policies random greedy neural \
  --early-stopping \
  --run-name demo-512-c12-final \
  --progress
```

Use early stopping for longer runs. The checkpoint with the best validation gain over greedy is saved under:

```text
outputs/<run-name>/checkpoints/best.pt
```

## Closed-loop Export

Closed-loop export reuses a saved model/checkpoint and writes rollout tensors for plotting and demos.

```bash
conda run -n manifold python -m manifold.cli export-closed-loop \
  --config configs/default.yaml \
  --device auto \
  --num-nodes 512 \
  --num-communities 12 \
  --p-in 0.04 \
  --p-out 0.0025 \
  --trajectory-steps 8 \
  --sensing-budget 24 \
  --eval-runs 20 \
  --policies random greedy neural \
  --source-run-dir outputs/demo-512-c12-final \
  --closed-loop-output-name closed-loop-eval-20-t8 \
  --progress
```

The 48-node proof-of-concept export uses the same checkpoint but a smaller graph:

```bash
conda run -n manifold python -m manifold.cli export-closed-loop \
  --config configs/default.yaml \
  --device auto \
  --num-nodes 48 \
  --num-communities 12 \
  --trajectory-steps 8 \
  --sensing-budget 6 \
  --eval-runs 20 \
  --policies random greedy neural \
  --source-run-dir outputs/demo-512-c12-final \
  --closed-loop-output-name closed-loop-eval-48-t8 \
  --fresh-eval-set \
  --progress
```

## ChatGPT 5.5 High Baseline

Run the fair ChatGPT baseline on selected rollout cases:

```bash
conda run -n manifold python -m manifold.cli export-chatgpt-baseline \
  --config configs/default.yaml \
  --device auto \
  --source-run-dir outputs/demo-512-c12-final \
  --rollout-spec 48:closed-loop-eval-48-t8:14 \
  --rollout-spec 512:closed-loop-eval-20-t8:15 \
  --chatgpt-model gpt-5.5 \
  --chatgpt-reasoning high \
  --chatgpt-output-name chatgpt55-high-comparison \
  --parallel \
  --progress
```

If raw ChatGPT responses already exist and only the parser/control conversion needs to be rerun:

```bash
conda run -n manifold python -m manifold.cli export-chatgpt-baseline \
  --config configs/default.yaml \
  --device auto \
  --source-run-dir outputs/demo-512-c12-final \
  --rollout-spec 48:closed-loop-eval-48-t8:14 \
  --rollout-spec 512:closed-loop-eval-20-t8:15 \
  --chatgpt-output-name chatgpt55-high-comparison \
  --chatgpt-replay-raw-responses \
  --parallel \
  --progress
```

## Regenerate Plots

```bash
conda run -n manifold python -m manifold.viz \
  --run-dir outputs/demo-512-c12-final \
  --fig-dir manifold/docs/figs
```

Use explicit selected runs if needed:

```bash
conda run -n manifold python -m manifold.viz \
  --run-dir outputs/demo-512-c12-final \
  --fig-dir manifold/docs/figs \
  --selected-run closed-loop-eval-48-t8:14 \
  --selected-run closed-loop-eval-20-t8:15
```

