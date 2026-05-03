# ChatGPT 5.5 High Baseline

The ChatGPT baseline evaluates direct LLM control under the same closed-loop regime as random, greedy, and neural policies.

## Fairness Constraints

ChatGPT receives:

- current visible belief summary
- visible node ids
- visible top-energy node summaries
- community labels for visible nodes
- probe node ids
- visible edge/cross-community counts
- sensing budget and allowed response schema
- previous control mean norm

ChatGPT does not receive:

- clean target trajectory
- future states
- unobserved node errors
- neural/greedy/random decisions
- full hidden state outside the observation mask
- privileged rankings from the selected rollout

## Decision Schema

ChatGPT must return JSON:

```json
{
  "selected_nodes": [0],
  "damping_gain": 0.08,
  "laplacian_gain": 0.1,
  "center_pull_gain": 0.03,
  "confidence": 0.7,
  "rationale": "short explanation"
}
```

The adapter enforces:

- selected nodes must be visible
- selected nodes are capped to sensing/control budget
- gains are clamped to `[0, 0.15]`
- invalid output falls back to zero control

## Parser and Replay

`codex exec` returns a transcript, not only the final JSON. The parser scans the transcript and uses the last decision-shaped JSON object.

If raw responses already exist, replay them without calling ChatGPT again:

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

## Current Result

After parser replay, ChatGPT produces valid decisions and nonzero controls on all selected steps:

| Case | Valid decisions | Nonzero control steps |
|---|---:|---:|
| 48-node rollout 14 | 8 / 8 | 8 / 8 |
| 512-node rollout 15 | 8 / 8 | 8 / 8 |

However, its controls are still too sparse/weak to materially change the 512-node trajectory compared with random/greedy.

For 512-node rollout 15:

| Policy | Active node-steps | Mean control norm | AUC | MSE |
|---|---:|---:|---:|---:|
| Greedy | 8 / 4096 | 0.00009 | 7.108 | 0.1034 |
| ChatGPT 5.5 High | 192 / 4096 | 0.040 | 7.105 | 0.1030 |
| Neural | 4096 / 4096 | 5.046 | 5.099 | 0.0604 |

Interpretation: ChatGPT is a valid fair baseline, but direct text-to-control decisions do not exploit graph-scale state-space structure like the trained neural controller.

