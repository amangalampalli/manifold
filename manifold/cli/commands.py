"""Command implementations for the manifold CLI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import logging
from datetime import datetime
from dataclasses import replace
from pathlib import Path
import time

from tqdm.auto import tqdm

from manifold.cli.parsing import build_parser, with_cli_overrides
from manifold.utils.config import ExperimentConfig, GraphConfig, SensingConfig, TrainingConfig, load_config, with_overrides
from manifold.trainer.core.chatgpt_baseline import (
    CodexExecChatGPTRunner,
    FakeChatGPTRunner,
    RawResponseReplayRunner,
    export_chatgpt_baseline_rollout,
)
from manifold.trainer.core.trainer import MinimaxTrainer


def run_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_and_override_config(args)

    trainer = MinimaxTrainer(config)
    if args.mode == "export-closed-loop":
        return _export_closed_loop_main(trainer, args)
    if args.mode == "export-chatgpt-baseline":
        return _export_chatgpt_baseline_main(config, args)

    run_dir = _prepare_run_dir(args) if args.save_run else None
    startup = trainer.startup_report()
    if run_dir is not None:
        _write_json(run_dir / "config.json", asdict(config))
        _write_json(run_dir / "startup.json", startup)
    logging.info("startup=%s", json.dumps(startup, sort_keys=True))

    eval_cases = None
    if args.mode == "compare":
        if args.eval_set_path is not None:
            eval_cases = trainer.load_eval_cases(args.eval_set_path)
        else:
            eval_cases = trainer.make_eval_cases(args.eval_runs)
        if run_dir is not None:
            trainer.save_eval_cases(run_dir / "eval_set.pt", eval_cases)
            trainer.save_graph_artifacts(run_dir)

    history, comparison = _run_training(
        trainer, args=args, run_dir=run_dir, eval_cases=eval_cases
    )
    if history.metrics:
        print(json.dumps({"final": history.metrics[-1]}, sort_keys=True))
    if args.mode == "compare":
        if comparison is None:
            policies = tuple(args.policies)
            comparison = trainer.compare_policies(
                policies,
                runs=args.eval_runs,
                eval_cases=eval_cases,
                progress=args.progress,
            )
        logging.info(
            "comparison_summary=%s",
            json.dumps(_comparison_summary(comparison), sort_keys=True),
        )
        if run_dir is not None and args.export_rollouts:
            manifest = trainer.export_rollouts(
                run_dir,
                policies=tuple(args.policies),
                eval_cases=eval_cases,
                progress=args.progress,
            )
            logging.info(
                "rollouts=%s", json.dumps({"count": len(manifest)}, sort_keys=True)
            )
        print(json.dumps({"comparison": comparison}, sort_keys=True))
    return 0


def _export_chatgpt_baseline_main(config: ExperimentConfig, args: argparse.Namespace) -> int:
    source_run_dir = Path(args.source_run_dir or "outputs/demo-512-c12-final")
    specs = args.rollout_spec or [
        "48:closed-loop-eval-48-t8:14",
        "512:closed-loop-eval-20-t8:15",
    ]
    output_dir = source_run_dir / args.chatgpt_output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def run_one(spec: str) -> dict[str, object]:
        label, folder, run_idx_text = _parse_rollout_spec(spec)
        run_idx = int(run_idx_text)
        folder_dir = source_run_dir / folder
        source_rollout_path = folder_dir / "closed_loop_rollouts" / f"neural_closed_loop_run_{run_idx:03d}.pt"
        graph_path = folder_dir / "graph.pt"
        source_rollout = __import__("torch").load(source_rollout_path, map_location="cpu", weights_only=False)
        graph_payload = __import__("torch").load(graph_path, map_location="cpu", weights_only=False)
        local_config = _config_for_rollout_artifacts(config, graph_payload, source_rollout)
        trainer = MinimaxTrainer(local_config)
        _install_graph_payload(trainer, graph_payload)
        trainer.load_checkpoint(source_run_dir / "checkpoints" / "best.pt")
        run_label = f"{label}_run_{run_idx:03d}"
        if args.chatgpt_replay_raw_responses:
            runner = RawResponseReplayRunner(raw_dir=output_dir / "raw_responses", run_label=run_label)
        elif args.chatgpt_fake_runner:
            runner = FakeChatGPTRunner()
        else:
            runner = CodexExecChatGPTRunner(model=args.chatgpt_model, reasoning=args.chatgpt_reasoning, cwd=Path.cwd())
        return export_chatgpt_baseline_rollout(
            trainer,
            output_dir=output_dir,
            source_rollout_path=source_rollout_path,
            run_label=run_label,
            runner=runner,
            progress=args.progress,
        )

    if args.parallel and len(specs) > 1:
        with ThreadPoolExecutor(max_workers=len(specs)) as executor:
            manifest = list(
                tqdm(
                    executor.map(run_one, specs),
                    total=len(specs),
                    desc="chatgpt-rollouts",
                    unit="rollout",
                    disable=not args.progress,
                )
            )
    else:
        manifest = [
            run_one(spec)
            for spec in tqdm(
                specs,
                desc="chatgpt-rollouts",
                unit="rollout",
                disable=not args.progress,
            )
        ]

    _write_json(output_dir / "chatgpt_rollouts_manifest.json", manifest)
    _write_json(output_dir / "comparison_summary.json", _chatgpt_comparison_summary(source_run_dir, output_dir, specs, manifest))
    timing = {
        "display_name": "ChatGPT 5.5 High",
        "transport": "codex exec",
        "model": args.chatgpt_model,
        "reasoning": args.chatgpt_reasoning,
        "total_wall_time_sec": time.perf_counter() - start,
        "rollouts": [
            {
                "run_label": item["run_label"],
                "chatgpt_rollout_wall_time_sec": item["chatgpt_rollout_wall_time_sec"],
                "chatgpt_step_wall_time_sec": item["chatgpt_step_wall_time_sec"],
            }
            for item in manifest
        ],
    }
    _write_json(output_dir / "timing.json", timing)
    logging.info("chatgpt_baseline_export=%s", json.dumps({"output_dir": str(output_dir), "rollouts": len(manifest)}, sort_keys=True))
    print(json.dumps({"chatgpt_output_dir": str(output_dir), "rollouts": len(manifest)}, sort_keys=True))
    return 0


def load_and_override_config(args: argparse.Namespace) -> ExperimentConfig:
    config = load_config(args.config)
    config = with_overrides(config, device=args.device)
    return with_cli_overrides(config, args)


def _export_closed_loop_main(trainer: MinimaxTrainer, args: argparse.Namespace) -> int:
    source_run_dir = args.source_run_dir or (
        args.output_dir / args.run_name if args.run_name else None
    )
    if source_run_dir is None:
        raise ValueError(
            "export-closed-loop requires --source-run-dir or --run-name with --output-dir"
        )
    source_run_dir = Path(source_run_dir)
    checkpoint_path = args.checkpoint_path or source_run_dir / "checkpoints" / "best.pt"
    eval_set_path = args.eval_set_path or source_run_dir / "eval_set.pt"
    output_dir = source_run_dir / args.closed_loop_output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = trainer.load_checkpoint(checkpoint_path)
    if args.fresh_eval_set:
        eval_cases = trainer.make_eval_cases(args.eval_runs)
        eval_set_source = "fresh"
    else:
        eval_cases = trainer.load_eval_cases(eval_set_path)[: args.eval_runs]
        eval_set_source = str(eval_set_path)
    trainer.save_graph_artifacts(output_dir)
    trainer.save_eval_cases(output_dir / "eval_set.pt", eval_cases)
    _write_json(output_dir / "startup.json", trainer.startup_report())
    _write_json(
        output_dir / "source.json",
        {
            "source_run_dir": str(source_run_dir),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "eval_set_path": eval_set_source,
            "eval_runs": len(eval_cases),
            "trajectory_steps": trainer.config.training.trajectory_steps,
        },
    )
    manifest = trainer.export_closed_loop_rollouts(
        output_dir,
        policies=tuple(args.policies),
        eval_cases=eval_cases,
        progress=args.progress,
    )
    logging.info(
        "closed_loop_export=%s",
        json.dumps(
            {"output_dir": str(output_dir), "rollouts": len(manifest)}, sort_keys=True
        ),
    )
    print(
        json.dumps(
            {"closed_loop_output_dir": str(output_dir), "rollouts": len(manifest)},
            sort_keys=True,
        )
    )
    return 0


def _run_training(
    trainer: MinimaxTrainer,
    *,
    args: argparse.Namespace,
    run_dir: Path | None,
    eval_cases: object | None,
):
    history = trainer.history
    comparison = None
    best_metric = float("-inf")
    stale_epochs = 0
    epochs = range(1, trainer.config.training.epochs + 1)
    iterator = tqdm(epochs, desc="training", unit="epoch", disable=not args.progress)
    for epoch in iterator:
        metrics = trainer.train_epoch(epoch)
        history.metrics.append(metrics)
        _log_epoch(run_dir, metrics)
        log_every = max(1, trainer.config.training.log_every)
        if int(metrics["epoch"]) % log_every == 0 and args.mode != "compare":
            logging.info(
                "train_summary=%s", json.dumps(_train_summary(metrics), sort_keys=True)
            )

        should_eval = epoch == trainer.config.training.epochs or (
            args.mode == "compare"
            and args.early_stopping
            and epoch % max(1, args.eval_every) == 0
        )
        if args.mode == "compare" and should_eval:
            comparison = trainer.compare_policies(
                tuple(args.policies),
                runs=args.eval_runs,
                eval_cases=eval_cases,
                progress=False,
            )
            eval_summary = {"epoch": float(epoch), "comparison": comparison}
            _log_eval(run_dir, eval_summary)
            score = _early_stopping_score(comparison, args.early_stopping_metric)
            metrics["early_stopping_score"] = score
            logging.info(
                "eval_summary=%s",
                json.dumps(
                    {"epoch": epoch, **_comparison_summary(comparison)}, sort_keys=True
                ),
            )
            if score > best_metric + args.early_stopping_min_delta:
                best_metric = score
                stale_epochs = 0
                if run_dir is not None:
                    trainer.save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        epoch=epoch,
                        metrics=metrics,
                    )
                    _write_json(
                        run_dir / "best.json",
                        {
                            "epoch": epoch,
                            "metric": args.early_stopping_metric,
                            "score": score,
                            "metrics": metrics,
                        },
                    )
            else:
                stale_epochs += 1
            if args.early_stopping and stale_epochs >= args.early_stopping_patience:
                metrics["early_stopped"] = 1.0
                break

        if run_dir is not None:
            trainer.save_checkpoint(
                run_dir / "checkpoints" / "last.pt", epoch=epoch, metrics=metrics
            )
        if args.progress:
            iterator.set_postfix(
                auc=f"{metrics['perturbation_auc']:.4f}",
                post_auc=f"{metrics['post_control_perturbation_auc']:.4f}",
                mse=f"{metrics['trajectory_mse']:.4f}",
            )
    return history, comparison


def _early_stopping_score(comparison: list[dict[str, float]], metric: str) -> float:
    for item in comparison:
        if item.get("policy") == "neural_minus_greedy":
            return float(item[metric])
    raise ValueError("Early stopping requires neural and greedy policies in comparison")


def _parse_rollout_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --rollout-spec '{spec}', expected label:folder:run_idx")
    return parts[0], parts[1], parts[2]


def _config_for_rollout_artifacts(
    config: ExperimentConfig, graph_payload: dict[str, object], source_rollout: dict[str, object]
) -> ExperimentConfig:
    probe_nodes = source_rollout.get("probe_nodes")
    times = source_rollout["times"]
    graph = replace(
        config.graph,
        num_nodes=int(graph_payload["num_nodes"]),
        num_communities=int(graph_payload["num_communities"]),
    )
    sensing = replace(
        config.sensing,
        budget=int(probe_nodes.shape[1]) if hasattr(probe_nodes, "shape") and len(probe_nodes.shape) == 2 else config.sensing.budget,
    )
    training = replace(
        config.training,
        trajectory_steps=int(times.numel() - 1),
    )
    return ExperimentConfig(
        device=config.device,
        seed=config.seed,
        graph=graph,
        model=config.model,
        sensing=sensing,
        training=training,
    )


def _install_graph_payload(trainer: MinimaxTrainer, graph_payload: dict[str, object]) -> None:
    trainer.dataset.edge_index = graph_payload["edge_index"].to(trainer.device)
    trainer.dataset.edge_attr = graph_payload["edge_attr"].to(trainer.device)
    trainer.dataset.labels = graph_payload["labels"].to(trainer.device)
    trainer.dataset.community_centers = graph_payload["community_centers"].to(trainer.device)
    trainer.dataset.graph_config = replace(
        trainer.dataset.graph_config,
        num_nodes=int(graph_payload["num_nodes"]),
        num_communities=int(graph_payload["num_communities"]),
    )
    trainer._cached_boundary_nodes = None


def _chatgpt_comparison_summary(
    source_run_dir: Path,
    output_dir: Path,
    specs: list[str],
    manifest: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_label = {item["run_label"]: item for item in manifest}
    rows = []
    torch = __import__("torch")
    for spec in specs:
        label, folder, run_idx_text = _parse_rollout_spec(spec)
        run_idx = int(run_idx_text)
        run_label = f"{label}_run_{run_idx:03d}"
        chatgpt = by_label[run_label]
        folder_dir = source_run_dir / folder / "closed_loop_rollouts"
        row: dict[str, object] = {
            "run_label": run_label,
            "chatgpt_mean_error_auc": chatgpt["mean_error_auc"],
            "chatgpt_trajectory_mse": chatgpt["trajectory_mse"],
            "chatgpt_final_mean_error": chatgpt["final_mean_error"],
            "chatgpt_path": chatgpt["path"],
        }
        for policy in ("random", "greedy", "neural"):
            path = folder_dir / f"{policy}_closed_loop_run_{run_idx:03d}.pt"
            if not path.exists():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            auc = float(torch.trapz(payload["mean_error"], payload["times"]))
            mse = float(torch.mean((payload["pred_trajectory"] - payload["target_trajectory"]) ** 2))
            row[f"{policy}_mean_error_auc"] = auc
            row[f"{policy}_trajectory_mse"] = mse
            row[f"chatgpt_vs_{policy}_auc_improvement_pct"] = 100.0 * (auc - float(chatgpt["mean_error_auc"])) / max(abs(auc), 1e-12)
            row[f"chatgpt_vs_{policy}_mse_improvement_pct"] = 100.0 * (mse - float(chatgpt["trajectory_mse"])) / max(abs(mse), 1e-12)
        rows.append(row)
    return rows


def _train_summary(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "epoch": metrics["epoch"],
        "post_control_perturbation_auc": metrics["post_control_perturbation_auc"],
        "trajectory_mse": metrics["trajectory_mse"],
        "control_energy": metrics["control_energy"],
        "boundary_nodes_used": metrics["boundary_nodes_used"],
    }


def _comparison_summary(comparison: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for item in comparison:
        policy = item.get("policy")
        if policy == "neural_minus_greedy":
            summary["neural_vs_greedy_post_control_auc_improvement_pct"] = item[
                "neural_vs_greedy_post_control_auc_improvement_pct"
            ]
            summary["neural_vs_greedy_mse_improvement_pct"] = item[
                "neural_vs_greedy_mse_improvement_pct"
            ]
            summary["neural_vs_greedy_auc_improvement_pct"] = item[
                "neural_vs_greedy_auc_improvement_pct"
            ]
        elif policy in {"random", "greedy", "neural"}:
            summary[f"{policy}_post_control_auc"] = item[
                "post_control_perturbation_auc"
            ]
            summary[f"{policy}_trajectory_mse"] = item["trajectory_mse"]
            summary[f"{policy}_boundary_nodes_used"] = item.get(
                "boundary_nodes_used", 0.0
            )
    return summary


def _prepare_run_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.run_name or f"{args.mode}-{stamp}"
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _log_epoch(run_dir: Path | None, metrics: dict[str, float]) -> None:
    if run_dir is not None:
        _append_jsonl(run_dir / "logs" / "metrics.jsonl", metrics)


def _log_eval(run_dir: Path | None, payload: object) -> None:
    if run_dir is not None:
        _append_jsonl(run_dir / "logs" / "eval.jsonl", payload)
