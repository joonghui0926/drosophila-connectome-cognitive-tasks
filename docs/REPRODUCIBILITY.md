# Reproducibility guide

## Levels of verification

The release supports three levels of verification.

1. `python scripts/reproduce_paper_results.py` recomputes every headline aggregate from the committed run records.
2. `python -m pytest -q` checks graph preparation, fixed-topology models, losses, split invariants, campaign construction, interventions, and aggregate values on small fixtures.
3. The commands in the root README rebuild the MaleCNS graph, corpus, rewired controls, and full training campaign.

The first two levels are CPU-friendly. The complete campaign is expensive: it trains a 25,582,938-edge recurrent network for 48 language cells and 12 reported addition runs, and it materializes 20 large rewired graphs.

## Addition protocol

The four anatomical fits use optimization seeds 11, 22, 33, and 44. Three rewired graphs contribute eight fits: four on rewire 1 and two each on rewires 2 and 3. This is an unbalanced repeated-run comparison, so the release reports its 24.84-point difference descriptively.

After preparing the rewires with the README commands, the following commands reproduce the reported addition run plan. The first three ensemble slots use graph-generation seeds 20260913, 20260917, and 20260918, respectively.

```bash
python scripts/run_addition.py \
  --graph-dir data/rewires/rewire-01 \
  --output runs/addition-rewire-01 \
  --seeds 11 22 33 44 --gpus 0

python scripts/run_addition.py \
  --graph-dir data/rewires/rewire-02 \
  --output runs/addition-rewire-02 \
  --seeds 11 22 --gpus 0

python scripts/run_addition.py \
  --graph-dir data/rewires/rewire-03 \
  --output runs/addition-rewire-03 \
  --seeds 11 22 --gpus 0
```

Every epoch traverses all 3,584 training expressions once without replacement. Model selection uses validation loss. The 792-expression test split is read after selection. `results/addition/runs.json` excludes activity-rendering payloads but retains the run configuration, graph manifest, baselines, validation metrics, and complete held-out metrics.

New fits require an empty run directory. Resume checks the saved model and optimizer settings, data hash, neuron metadata hash, and graph identifier before restoring weights. Stopping limits may be changed to continue a run. Checkpoints without the data and metadata hashes are not accepted. The loader also verifies each graph file against its manifest before constructing the model.

## Language protocol

The primary factorial crosses the anatomical graph and four rewired graphs with four fixed interfaces. All primary cells use optimization seed 11 and the same input and output codebooks. The graph-null comparison evaluates the anatomical graph and 20 independently rewired graphs at the canonical interface. Additional cells cover dense and no-sensory baselines and optimization seeds 22 and 33 for the anatomical graph and rewires 1 and 2.

Eight canonical-interface evaluations came from completed runs at the same four-million-token endpoint. Graph-hash validation found that two directory labels did not match their graph identities. The released records assign topology by graph SHA-256 and record the duplicate-resolution rule in `results/language/correction-audit.json`.

The 50 raw evaluations yield 49 deduplicated records. Forty-eight belong to the specified design. One additional valid evaluation is retained in `results/language/runs.json` but is excluded from its primary aggregates.

The campaign launcher covers all 48 cells. GPU indices in the configuration control scheduling only. Each subprocess receives one selected device through `CUDA_VISIBLE_DEVICES` and addresses it as `cuda:0`.

## Primary language score

For a held-out two-event group, the evaluator creates an original scene and an order-reversed scene. Both candidate descriptions use the fixed relation-first form "X happened before Y". It scores the matched and mismatched equal-length descriptions under both scenes. A group is correct only if both matched scores exceed both mismatched scores; ties fail. Chance under independent continuous scores is one sixth. This is a paired teacher-forced ranking metric, not free-generation exact match.

The sensory-shuffle intervention keeps the questions and candidate descriptions fixed while replacing a group's complete sensory-amplitude tensor with the tensor from another group. The reported comparison uses the same 48 group IDs in both conditions.

## Determinism and provenance

Training on sparse CUDA kernels may vary slightly across hardware and software versions. The language launcher records graph and corpus hashes, interface seeds, optimization seeds, codebook seeds, training budgets, and evaluation counts. Each run directory stores its configuration. Language runs also retain source snapshots and an interface fingerprint.

The code that produced the saved results was tested on Python 3.11 and PyTorch 2.6.0 with CUDA 12.4. The package accepts newer compatible installations, but exact floating-point trajectories are not promised across devices.
