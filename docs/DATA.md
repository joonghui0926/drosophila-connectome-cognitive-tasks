# Data sources and generated artifacts

## MaleCNS v1.0

The code uses the official Janelia FlyEM MaleCNS v1.0 release. The downloader in `src/flystudy/acquire.py` streams each file to a temporary path, verifies its byte size and SHA-256 digest, and then performs an atomic rename.

| Purpose | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Neuron annotations | `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| Directed connections | `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

Source page: <https://male-cns.janelia.org/download/>

The processed graph selects annotation rows with a non-null `superclass`, sorts retained `bodyId` values, keeps a connection only when both endpoints are retained, preserves direction from `body_pre` to `body_post`, and aggregates repeated directed body pairs. It does not add or threshold edges.

The expected graph manifest is:

| Field | Value |
| --- | ---: |
| Retained nodes | 166,700 |
| Directed body-pair edges | 25,582,938 |
| Synaptic contacts | 124,177,617 |
| Self-edges | 101 |
| Graph SHA-256 | `e906f5f47da85b625c3941a411882d507ff7383c98097ac07819a83ee998266a` |

The accompanying [Cell article](https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6) reports 166,700 neurons in total and 166,483 connected neurons after excluding 217 neurons without synaptic connections. The processed graph retains 166,700 unique body IDs under the stated selection rule. Historical run manifests preserve an earlier `published_neuron_count` value of 166,691. That field is source metadata, not the number of nodes used for training.

## Bounded addition

`data/bounded-arithmetic-split-v1.json` contains a deterministic arithmetic split from which the reported addition runs select the `+` rows. The selected subset contains 3,584 training, 775 validation, and 792 test expressions. Unordered operand pairs do not overlap across splits. The other operation rows are retained because they share the same family-level split and fixed 47-channel encoder, but they are not read by the reported addition trainer.

## Controlled-language corpus

The corpus is generated deterministically by `scripts/build_language_corpus.py`. The default materialization contains 200,000 training rows and 4,096 rows in each validation and test split. Training rows use 228 unordered channel-pair families; validation and test reserve 24 families each. Both event orders and paraphrases for a pair remain in the same partition.

The generated corpus is not committed because its five data files can be reproduced byte for byte. The reference hashes and corpus audits are in `provenance/language_corpus_manifest.json`, `provenance/language_corpus_audit.json`, and `provenance/language_corpus_materialized_audit.json`. A fresh manifest records the current builder and source snapshot. Those source hashes can differ from the historical manifest without changing the generated data.

## Rewired graphs

Rewired controls are generated locally from the processed graph. A directed double-edge swap preserves each node's unweighted in-degree and out-degree, retains the original self-loop set, and rejects duplicate edges. The released campaign configuration fixes 20 graph seeds and requires five accepted swaps per original edge. The graphs are not committed because each compressed adjacency is large and can be rebuilt from the public source graph.
