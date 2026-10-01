"""Token-budgeted training, checkpointing and runtime diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

from flystudy.language_support import atomic_json, build_model, interface_fingerprint, utcnow
from flystudy.language_metrics import evaluate, paired_relations, family_subset, familiar_subset, text_lm_score
from flystudy.language_runtime import (
    DenseLanguageModel, TokenMixer, file_hash, load_records, processed_tokens,
    tiny_model, update,
)


SOURCE_FILES = (
    "language_training.py", "language_runtime.py", "language_metrics.py", "language_corpus.py",
    "language_audit.py",
    "language_support.py", "language_model.py", "language_lexicon.py", "language_anatomy.py",
    "language_sparse.py", "sparse_ops.py", "graph.py", "language_data.py",
)


def source_hashes():
    root = Path(__file__).parent
    return {name: file_hash(root / name) for name in SOURCE_FILES}


def device_for(args):
    if args.tiny:
        return torch.device("cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("language training requires one process per GPU")
    if visible is not None:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if len(devices) != 1:
            raise RuntimeError("exactly one CUDA device must be visible to each process")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA is unavailable or GPU isolation failed")
    torch.cuda.set_device(0)
    expected = os.environ.get("FLY_EXPECTED_GPU_UUID")
    if expected:
        actual = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
        normalize = lambda text: text.lower().removeprefix("gpu-").replace("-", "")
        if normalize(actual) != normalize(expected):
            raise RuntimeError("CUDA device UUID does not match the permitted nvidia-smi index")
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda:0")


def build(args, device):
    if args.tiny:
        model = tiny_model(seed=args.seed, dense=args.condition == "dense",
            token_steps=args.token_steps, token_drive_mode=args.token_drive_mode,
            leak=args.leak, token_gain=args.token_gain, sensor_gain=args.sensor_gain,
            temperature=args.temperature).to(device)
        manifest = {"graph_sha256": "synthetic-local-test", "neurons": model.n_nodes,
                    "edges": model.theta.numel(), "synthetic": True}
    else:
        if args.graph_dir is None:
            raise ValueError("--graph-dir is required")
        manifest = json.loads((args.graph_dir / "manifest.json").read_text())
        if file_hash(args.graph_dir / "graph.npz") != manifest["graph_sha256"]:
            raise ValueError("graph checksum mismatch")
        rewired = manifest.get("control") == "directed-degree-preserving-rewire"
        if rewired != (args.condition == "rewired"):
            raise ValueError("condition and graph topology disagree")
        if args.condition == "dense":
            width = round(math.sqrt(manifest["edges"]))
            model = DenseLanguageModel(width, seed=args.seed, interface_seed=args.interface_seed,
                embedding=args.embedding, dimension=args.dimension, token_group=args.token_group,
                output_group=args.output_group, token_steps=args.token_steps, leak=args.leak,
                token_gain=args.token_gain, sensor_gain=args.sensor_gain, temperature=args.temperature,
                backend=args.backend, chunk_size=args.chunk_size,
                token_drive_mode=args.token_drive_mode).to(device)
        else:
            model, _, _, _ = build_model(args, device)
    if [name for name, _ in model.named_parameters()] != ["theta"]:
        raise ValueError("only recurrent theta may be trained")
    if args.condition != "dense" and model.theta.numel() != manifest["edges"]:
        raise ValueError("connectome parameter count mismatch")
    return model, manifest


def config_for(args, model, manifest):
    # Runtime location, device assignment and interruptions can change on resume.
    excluded = {"run_dir", "graph_dir", "corpus", "phase", "resume", "deadline", "stop_after_updates",
                "benchmark_updates", "microbatches"}
    return {"schema": 7, "revision": 2,
            "arguments": {k: v for k, v in vars(args).items() if k not in excluded},
            "graphSha256": manifest["graph_sha256"],
            "corpusSha256": file_hash(args.corpus / "manifest.json"),
            "sourceHashes": source_hashes(), "interfaceSha256": interface_fingerprint(model),
            "trainableParameters": model.theta.numel(), "neurons": model.n_nodes,
            "objective": "pure next-token CE normalized by global supervised tokens",
            "textLoss": "entire causal document", "groundedLoss": "answer and EOS only",
            "mixture": "60/20/20 processed tokens after report warmup",
            "primaryCheckpoint": "final common token budget; validation-selected checkpoint secondary",
            "precision": "FP32, TF32 disabled", "syntheticTest": args.tiny,
            "runtimeVersions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                "numpy": np.__version__, "torchCUDA": torch.version.cuda}}


def save_state(path, model, optimizer, config, mixer, updates, best_score, history):
    temporary = path.with_suffix(".tmp")
    torch.save({"config": config, "theta": model.theta.detach().cpu(),
                "optimizer": optimizer.state_dict(), "mixer": mixer.state_dict(),
                "updates": updates, "bestScore": best_score, "history": history,
                "torchRng": torch.get_rng_state()}, temporary)
    os.replace(temporary, path)


def initialize_run_directory(args, config):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.run_dir / "config.json", config)
    snapshot = args.run_dir / "source-snapshot"
    snapshot.mkdir()
    for name in SOURCE_FILES:
        (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())


def save_diagnosis_state(path, model, optimizer, config, updates, history):
    temporary = path.with_suffix(".tmp")
    torch.save({"config": config, "theta": model.theta.detach().cpu(),
                "optimizer": optimizer.state_dict(), "updates": updates,
                "history": history, "torchRng": torch.get_rng_state()}, temporary)
    os.replace(temporary, path)


def restore_diagnosis_state(path, model, optimizer, config):
    saved = torch.load(path, map_location=model.theta.device, weights_only=False)
    if saved["config"] != config:
        raise ValueError("diagnosis source/data/graph/interface/config mismatch")
    with torch.no_grad():
        model.theta.copy_(saved["theta"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torchRng"].cpu())
    return saved


def restore(path, model, optimizer, config, mixer):
    saved = torch.load(path, map_location=model.theta.device, weights_only=False)
    if saved["config"] != config:
        raise ValueError("resume source/data/graph/interface/config mismatch")
    with torch.no_grad():
        model.theta.copy_(saved["theta"])
    if optimizer is not None:
        optimizer.load_state_dict(saved["optimizer"])
    if mixer is not None:
        mixer.load_state_dict(saved["mixer"])
    torch.set_rng_state(saved["torchRng"].cpu())
    return saved


def optimizer_for(model, args):
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def run_train(args, model, manifest, device):
    config = config_for(args, model, manifest)
    rows = load_records(args.corpus, "train")
    mixer = TokenMixer(rows, warmup_tokens=args.warmup_tokens)
    optimizer = optimizer_for(model, args)
    checkpoint = args.run_dir / "checkpoint-latest.pt"
    updates, history, best_score = 0, [], -1.0
    if args.resume:
        saved = restore(checkpoint, model, optimizer, config, mixer)
        updates, history, best_score = saved["updates"], saved["history"], saved["bestScore"]
        if (args.run_dir / "evaluation.json").exists():
            raise ValueError("cannot resume training after final test evaluation")
    else:
        initialize_run_directory(args, config)
    validation = family_subset(load_records(args.corpus, "validationComposition"), args.validation_per_family, reports_only=True)
    validation_id = familiar_subset(load_records(args.corpus, "validationId"), args.id_examples)
    stopped = False
    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True
    old_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    initial_tokens = mixer.total
    started = time.monotonic()
    last_save = mixer.total
    milestones = [x for x in (1_000_000, 2_000_000, 4_000_000, args.train_tokens) if x > mixer.total]
    completed = False
    try:
        while mixer.total < args.train_tokens:
            if stopped or (args.deadline and time.time() >= args.deadline):
                break
            if args.stop_after_updates and updates >= args.stop_after_updates:
                break
            batch_rows = mixer.take(args.global_batch, args.train_tokens - mixer.total)
            if not batch_rows:
                completed = True  # final unused budget is shorter than one complete record
                break
            try:
                metric = update(model, optimizer, batch_rows, microbatch=args.microbatch,
                    encode_steps=args.encode_steps, device=device, no_sensory=args.condition == "no_sensory",
                    gradient_clip=args.gradient_clip)
            except Exception:
                # The previous checkpoint remains the last atomic model/data cursor pair.
                atomic_json(args.run_dir / "failure.json", {"at": utcnow(), "updates": updates,
                                                          "message": "update failed; resume last checkpoint"})
                raise
            updates += 1
            elapsed = time.monotonic() - started
            live = {"status": "training", "at": utcnow(), "updates": updates,
                    "processedTokens": mixer.total, "budgetTokens": args.train_tokens,
                    "tokensPerSecond": (mixer.total - initial_tokens) / max(elapsed, 1e-9),
                    "mixtureTokens": mixer.exposure, **{k:v for k,v in metric.items() if k != "processedTokens"}}
            if updates == 1 or updates % args.log_every == 0:
                atomic_json(args.run_dir / "live.json", live)
                with (args.run_dir / "training.jsonl").open("a") as stream:
                    stream.write(json.dumps(live) + "\n")
                print(json.dumps(live), flush=True)
            due = any(mixer.total >= m for m in milestones)
            if due and (not args.deadline or time.time() < args.deadline):
                try:
                    composition, _ = evaluate(model, validation, device=device, batch_size=args.eval_batch,
                        encode_steps=args.encode_steps, no_sensory=args.condition == "no_sensory", deadline=args.deadline)
                    familiar, _ = evaluate(model, validation_id, device=device, batch_size=args.eval_batch,
                        encode_steps=args.encode_steps, no_sensory=args.condition == "no_sensory", deadline=args.deadline)
                except TimeoutError:
                    break
                entry = {"tokens": mixer.total, "validationComposition": composition, "validationId": familiar}
                history.append(entry)
                milestones = [m for m in milestones if m > mixer.total]
                if composition["familyMacroSemantic"] > best_score:
                    best_score = composition["familyMacroSemantic"]
                    save_state(args.run_dir / "checkpoint-validation-best.pt", model, optimizer,
                               config, mixer, updates, best_score, history)
            if due or mixer.total - last_save >= args.checkpoint_every_tokens:
                save_state(checkpoint, model, optimizer, config, mixer, updates, best_score, history)
                last_save = mixer.total
        completed = completed or mixer.total >= args.train_tokens
        save_state(checkpoint, model, optimizer, config, mixer, updates, best_score, history)
        summary = {"status": "trained" if completed else "paused", "at": utcnow(),
                   "processedTokens": mixer.total, "budgetTokens": args.train_tokens,
                   "unusedTokens": args.train_tokens - mixer.total, "updates": updates,
                   "maxRecordTokens": max(map(processed_tokens, rows)),
                   "mixtureTokens": mixer.exposure,
                   "trainingSecondsThisSession": time.monotonic() - started,
                   "history": history, "testEvaluated": False, "syntheticTest": args.tiny}
        atomic_json(args.run_dir / "training-state.json", summary)
        atomic_json(args.run_dir / "live.json", summary)
        return summary
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def run_evaluation(args, model, manifest, device):
    config = config_for(args, model, manifest)
    state = json.loads((args.run_dir / "training-state.json").read_text())
    if state["status"] != "trained":
        raise ValueError("complete the common training budget before final test evaluation")
    destination = args.run_dir / "evaluation.json"
    if destination.exists():
        raise FileExistsError("final evaluation already exists")
    saved = restore(args.run_dir / "checkpoint-latest.pt", model, None, config, None)
    rows = load_records(args.corpus, "testComposition")
    primary = family_subset(rows, args.test_per_family, reports_only=True)
    pairs = family_subset(rows, args.pairs_per_family, reports_only=True, order_invariant=True)
    if args.intervention_examples % 24 or args.intervention_examples < 24:
        raise ValueError("intervention count must be a positive multiple of 24 families")
    interventions = family_subset(rows, args.intervention_examples // 24, reports_only=True)
    familiar_rows = load_records(args.corpus, "validationId")
    id_rows = familiar_subset(familiar_rows, args.id_examples)
    results, predictions = {}, {}
    for name, subset, intervention in (("composition", primary, "normal"), ("familiar", id_rows, "normal"),
                                      ("sensoryZero", interventions, "zero"),
                                      ("sensoryShuffle", interventions, "shuffle"),
                                      ("orderSwap", interventions, "swap"),
                                      ("strengthChange", interventions, "flip_strength")):
        if args.deadline and time.time() >= args.deadline:
            raise TimeoutError("evaluation deadline reached before all required metrics")
        results[name], predictions[name] = evaluate(model, subset, device=device,
            batch_size=args.eval_batch, encode_steps=args.encode_steps, intervention=intervention,
            no_sensory=args.condition == "no_sensory", deadline=args.deadline)
    results["pairedRelations"], predictions["pairedRelations"] = paired_relations(model, pairs,
        device=device, batch_size=max(1, args.eval_batch // 4), encode_steps=args.encode_steps,
        no_sensory=args.condition == "no_sensory", deadline=args.deadline)
    results["validationTextLM"] = text_lm_score(model, [r for r in familiar_rows if r.mode == "text"][:args.id_examples],
        device=device, batch_size=args.eval_batch, encode_steps=args.encode_steps, deadline=args.deadline)
    atomic_json(args.run_dir / "predictions.json", predictions)
    result = {"at": utcnow(), "config": config, "processedTokens": saved["mixer"]["total"],
              "checkpoint": "checkpoint-latest.pt", "results": results}
    atomic_json(destination, result)
    return result


def run_benchmark(args, model, manifest, device):
    parity = verify_kernel(device)
    rows = load_records(args.corpus, "train")
    initial = model.theta.detach().cpu().clone()
    results = []
    for microbatch in args.microbatches:
        with torch.no_grad():
            model.theta.copy_(initial)
        optimizer = optimizer_for(model, args)
        mixer = TokenMixer(rows, warmup_tokens=0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations, tokens = [], []
        try:
            for iteration in range(args.benchmark_updates + 1):
                block = mixer.take(args.global_batch, 10**9)
                if device.type == "cuda": torch.cuda.synchronize()
                begin = time.monotonic()
                metric = update(model, optimizer, block, microbatch=microbatch,
                    encode_steps=args.encode_steps, device=device, no_sensory=args.condition == "no_sensory")
                if device.type == "cuda": torch.cuda.synchronize()
                if iteration:
                    durations.append(time.monotonic() - begin)
                    tokens.append(metric["processedTokens"])
            results.append({"microbatch": microbatch, "tokensPerSecond": sum(tokens)/sum(durations),
                            "peakBytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})
        except torch.cuda.OutOfMemoryError:
            results.append({"microbatch": microbatch, "error": "CUDA out of memory"})
        finally:
            model.zero_grad(set_to_none=True)
            del optimizer
            if device.type == "cuda": torch.cuda.empty_cache()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = {"at": utcnow(), "syntheticTest": args.tiny, "condition": args.condition,
              "physicalGPU": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "interfaceSha256": interface_fingerprint(model), "measurements": results,
              "kernelParity": parity}
    atomic_json(args.run_dir / "benchmark.json", output)
    return output


def verify_kernel(device):
    from flystudy.language_corpus import make_record
    from flystudy.language_runtime import make_batch
    a, b = tiny_model().to(device), tiny_model().to(device)
    b.backend = "edge"
    batch = make_batch([make_record(0), make_record(8)], device=device)
    x, y = a(*batch.inputs()), b(*batch.inputs())
    torch.testing.assert_close(x, y, atol=3e-5, rtol=3e-5)
    x.square().mean().backward()
    y.square().mean().backward()
    torch.testing.assert_close(a.theta.grad, b.theta.grad, atol=5e-6, rtol=3e-4)
    return {"passed": True, "device": str(device), "scope": "synthetic sparse kernel forward/backward parity",
            "maxGradientDifference": float((a.theta.grad-b.theta.grad).abs().max())}


def run_diagnosis(args, model, manifest, device):
    """Train-only acquisition gate. No validation or test selects this subset."""
    config = config_for(args, model, manifest)
    rows = [r for r in load_records(args.corpus, "train") if r.mode == "report"]
    # Equal single/two-event diagnostic exposure avoids hiding second-event failure.
    subset = [r for r in rows if len(r.scene.events) == 1][:args.diagnostic_examples // 2]
    subset += [r for r in rows if len(r.scene.events) == 2][:args.diagnostic_examples // 2]
    if len(subset) != args.diagnostic_examples:
        raise ValueError("insufficient diagnostic rows")
    optimizer = optimizer_for(model, args)
    checkpoint = args.run_dir / "diagnosis-checkpoint.pt"
    if args.resume:
        saved = restore_diagnosis_state(checkpoint, model, optimizer, config)
        completed_updates, metrics = saved["updates"], saved["history"]
    else:
        initialize_run_directory(args, config)
        completed_updates, metrics = 0, []
    stopped = False
    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True
    old_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for step in range(completed_updates, args.diagnostic_updates):
            if stopped or (args.deadline and time.time() >= args.deadline - 300):
                break
            offset = (step * args.global_batch) % len(subset)
            block = [subset[(offset + i) % len(subset)] for i in range(min(args.global_batch, len(subset)))]
            metric = update(model, optimizer, block, microbatch=args.microbatch, encode_steps=args.encode_steps,
                            device=device, no_sensory=args.condition == "no_sensory")
            completed_updates = step + 1
            if step == 0 or completed_updates % 10 == 0:
                metrics.append({"update": completed_updates, **metric})
                print(json.dumps(metrics[-1]), flush=True)
            if completed_updates % args.diagnostic_checkpoint_every == 0:
                save_diagnosis_state(checkpoint, model, optimizer, config, completed_updates, metrics)
        save_diagnosis_state(checkpoint, model, optimizer, config, completed_updates, metrics)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if completed_updates < args.diagnostic_updates:
        output = {"at": utcnow(), "status": "paused", "trainingSubsetOnly": True,
                  "updates": completed_updates, "targetUpdates": args.diagnostic_updates,
                  "gatePassed": False, "history": metrics,
                  "checkpoint": checkpoint.name, "checkpointTransferredToMainTraining": False}
        atomic_json(args.run_dir / "diagnosis.json", output)
        atomic_json(args.run_dir / "diagnosis-samples.json", [])
        return output
    result, samples = evaluate(model, subset, device=device, batch_size=args.eval_batch,
                               encode_steps=args.encode_steps, no_sensory=args.condition == "no_sensory",
                               deadline=args.deadline)
    two = [s for s in samples if s["family"] != "single"]
    two_accuracy = sum(s["semanticCorrect"] for s in two) / len(two)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = {"at": utcnow(), "status": "complete", "trainingSubsetOnly": True,
              "config": config, "updates": completed_updates, "overall": result,
              "twoEventSemanticExact": two_accuracy, "gatePassed": two_accuracy >= .9,
              "history": metrics, "checkpoint": checkpoint.name,
              "checkpointTransferredToMainTraining": False}
    atomic_json(args.run_dir / "diagnosis.json", output)
    atomic_json(args.run_dir / "diagnosis-samples.json", samples)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("train", "evaluate", "benchmark", "diagnose"), default="train")
    p.add_argument("--condition", choices=("real", "rewired", "no_sensory", "dense"), default="real")
    p.add_argument("--corpus", type=Path, default=Path("data/language-corpus"))
    p.add_argument("--graph-dir", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--interface-seed", type=int, default=20260914)
    p.add_argument("--embedding", choices=("orthogonal", "random"), default="orthogonal")
    for name, value in (("dimension",128), ("token-group",8), ("output-group",8), ("token-steps",4),
                        ("encode-steps",8), ("chunk-size",262144), ("global-batch",192), ("microbatch",24),
                        ("train-tokens",4000000), ("warmup-tokens",1000000), ("checkpoint-every-tokens",500000),
                        ("validation-per-family",2), ("test-per-family",16), ("pairs-per-family",8),
                        ("intervention-examples",192), ("id-examples",384), ("eval-batch",12),
                        ("log-every",5), ("stop-after-updates",0), ("benchmark-updates",4),
                        ("diagnostic-examples",256), ("diagnostic-updates",120),
                        ("diagnostic-checkpoint-every",60)):
        p.add_argument("--" + name, type=int, default=value)
    for name, value in (("leak",.5), ("token-gain",4.), ("sensor-gain",1.), ("temperature",.1),
                        ("learning-rate",.001), ("weight-decay",.0001), ("gradient-clip",1.), ("deadline",0.)):
        p.add_argument("--" + name, type=float, default=value)
    p.add_argument("--backend", choices=("csr", "edge"), default="csr")
    p.add_argument("--token-drive-mode", choices=("pulse", "sustained"), default="pulse")
    p.add_argument("--microbatches", type=int, nargs="+", default=[64,96,128])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tiny", action="store_true", help="synthetic CPU verification; not a fly experiment")
    return p
