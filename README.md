# A Drosophila Whole-Connectome Network Can Learn Human-Designed Cognitive Tasks

This repository contains the code and compact result records for the bounded-addition and grounded relational language experiments in the paper. The recurrent graph is built from the public MaleCNS v1.0 release. Its directed edge set is fixed, and each retained edge receives one trainable scalar. Addition and language use separate fits.

## Repository layout

```text
configs/       Experiment settings
data/          The arithmetic split used for addition; MaleCNS files are downloaded separately
docs/          Data and reproduction notes
provenance/    Corpus manifests and audits
results/       Compact run-level records used to recover the paper values
scripts/       Corpus, rewire, training, evaluation, and analysis entry points
src/flystudy/  Graph preparation and task implementations
tests/         Unit, integration, and paper-result checks
```

## Installation

Python 3.11 or later is required. Rewire generation uses `g++` with C++17 support. Full training requires a CUDA installation compatible with PyTorch. The full campaign was run on Linux.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. The tests and result checks run on CPU. Rewire generation also requires `g++` on `PATH`.

## Verify the release

The following commands run the test suite and independently recover the reported aggregate values from the compact run records:

```bash
python -m pytest -q
python scripts/reproduce_paper_results.py
```

The result check reports 92.77% versus 67.93% for addition, 61.59% versus 44.17% for the four-interface language comparison, 71.35% for the anatomical graph in the locked graph-null comparison, and 60.94% versus 19.27% for the matched sensory-shuffle intervention.

## Prepare MaleCNS

The downloader retrieves the two official MaleCNS v1.0 flat-connectome files and verifies their byte sizes and SHA-256 digests. The graph builder retains rows with an assigned `superclass`, sorts bodies by `bodyId`, keeps directed connections for which both endpoints are retained, aggregates repeated body pairs, and applies no edge threshold.

```bash
flystudy download
flystudy prepare
```

The expected processed graph has 166,700 nodes, 25,582,938 directed body-pair edges, and 124,177,617 synaptic contacts. See `docs/DATA.md` for source URLs, checksums, and the distinction between total and connected neurons.

## Bounded addition

The addition subset contains 3,584 training, 775 validation, and 792 test expressions. Training visits every addition expression once per epoch. The selected checkpoint minimizes validation loss; the held-out test split is evaluated after selection.

```bash
flystudy train-addition \
  --graph-dir data/processed/malecns-v1.0-annotated \
  --dataset data/bounded-arithmetic-split-v1.json \
  --config configs/addition.json \
  --run-dir runs/addition-seed11 \
  --device cuda \
  --max-epochs 100 \
  --min-epochs 20 \
  --patience 15 \
  --seed 11
```

The following command launches the four anatomical fits used in the paper. It uses one GPU by default and accepts more GPU indices through `--gpus`.

```bash
python scripts/run_addition.py \
  --graph-dir data/processed/malecns-v1.0-annotated \
  --output runs/addition-anatomical \
  --gpus 0
```

Use `--seeds` to select optimization seeds for a rewired graph. The paper uses four fits on rewire 1 and two each on rewires 2 and 3. The exact commands are in `docs/REPRODUCIBILITY.md`.

Start each new experiment in an empty run directory. To continue an existing fit, add `--resume` to the same command. Resume preserves the training log and checks the saved settings and data hashes.

## Grounded relational language

First materialize the deterministic controlled-language corpus and create the 20 directed degree-preserving rewires:

```bash
python scripts/build_language_corpus.py \
  --output-dir data/language-corpus

python scripts/prepare_language_rewire_ensemble.py \
  --source data/processed/malecns-v1.0-annotated \
  --output-root data/rewires \
  --config configs/language_confirmatory.json \
  --workers 2 \
  --execute
```

Set `allowedPhysicalGpus` in `configs/language_confirmatory.json` to the GPU indices available on the machine. The default uses one GPU. Additional indices run cells in parallel. Each subprocess sees one device through `CUDA_VISIBLE_DEVICES` and addresses it as `cuda:0`.

```bash
python scripts/run_language.py \
  --real data/processed/malecns-v1.0-annotated \
  --rewire-root data/rewires \
  --corpus data/language-corpus \
  --output runs/language-confirmatory \
  --config configs/language_confirmatory.json \
  --execute
```

The launcher writes a campaign plan before training and records file and graph hashes. It skips completed cells, resumes interrupted fits, and runs the evaluator only after the fixed token budget is reached.

## Results and provenance

`results/addition/runs.json` and `results/language/runs.json` contain the compact records used for the paper. Aggregate language results are in `results/language/summary.json`. See `docs/REPRODUCIBILITY.md` for run selection and graph-identity checks, including the record audit in `results/language/correction-audit.json`.

## Citation

Please cite the accompanying paper. Machine-readable author and title metadata are in `CITATION.cff`.

## License

The code is released under the MIT License. MaleCNS data are not redistributed and remain under the terms given by the Janelia FlyEM release.
