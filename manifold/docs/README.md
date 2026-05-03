# Manifold Signal Stabilization Docs

This directory documents the research prototype for continuous-time graph signal recovery with neural sheaf ODEs. It covers the model components, training/evaluation workflows, saved artifacts, and slide-ready figures.

## Quick Links

- [Architecture](architecture.md): model, sheaf operator, dynamics, sensing, and policies.
- [Running Experiments](running_experiments.md): training, closed-loop export, ChatGPT baseline export, and plotting commands.
- [Artifacts and Outputs](artifacts_and_outputs.md): what is saved under `outputs/` and how to reuse it.
- [Visualizations](visualizations.md): how to regenerate figures and what each plot means.
- [Results and Pitch Figures](results_and_pitch.md): the small set of figures to use in slides.
- [ChatGPT Baseline](chatgpt_baseline.md): fair baseline protocol, parser/replay behavior, and interpretation.

## Current Demo Run

Most generated documentation and figures assume this run directory:

```bash
outputs/demo-512-c12-final
```

The slide-ready figures are in:

```bash
manifold/docs/figs
```

Regenerate all figures with:

```bash
conda run -n manifold python -m manifold.viz \
  --run-dir outputs/demo-512-c12-final \
  --fig-dir manifold/docs/figs
```

## Environment

The expected local environment is the conda environment named `manifold`, with the repository's Poetry dependencies already installed.

```bash
conda run -n manifold pytest -q
```

