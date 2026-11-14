# Qwen3 latent relational computation — reproduction bundle

This is a **local-CPU, executed experiment bundle**, not a claim of universal self-rewriting. The report `RESEARCH_REPORT_2026-11-13.md` separates the proven benchmarks, architectural priors, and unsolved goals.

## Reproducing

1. Place the archive contents in `/mnt/data/qwen_next` (or update `P=Path('/mnt/data/qwen_next')` in the scripts). Python 3.13, PyTorch 2.10 CPU, NumPy, SciPy, and Transformers 4.57.6 were used. Most training scripts need only PyTorch/NumPy and the supplied pretrained feature caches, not the full original weights.
2. Fetch the **official** bAbI Task 19 en-10k train/test files and Task 2 en-10k train/test files into that directory: `qa19_10k_train.txt`, `qa19_10k_test.txt`, `qa2_10k_train.txt`, `qa2_10k_test.txt`. The connected GitHub repo branch `research/qwen3-streaming-memory-babi-20261113` contains the original data-fetching GitHub Actions workflows.
3. With the Qwen feature caches already in this bundle, run the following commands:

```bash
python qa19_anonymous_graph.py --epochs 100 --seed 11819 --feature qwen
python qa19_anonymous_graph.py --epochs 100 --seed 11819 --feature whiten
python qa19_anonymous_graph_ood.py --epochs 100 --seed 11819 --feature whiten --nodes 6
python qa19_graph_learn_inverse.py --epochs 120 --seed 11819 --feature whiten --nodes 8 --reverse learned
python qa19_graph_learn_inverse.py --epochs 100 --seed 11819 --feature whiten --nodes 8 --reverse none
python qa19_exact_reference.py
python stats_qa19.py
QWEN_FEATURE_MODE=raw python qa2_gru_geometry_transfer.py
QWEN_FEATURE_MODE=whiten python qa2_gru_geometry_transfer.py
```

These experiments select checkpoints on **story-disjoint validation**, then evaluate the official test. `--feature whiten` for the OOD study fits the transform exclusively on the **training vocabulary** after excluding the twelve held-out sentence types. The original Task 19 train/test vocabularies share all 150 sentence strings; the withheld-types experiment does not.

## Execution record

- The `qa19_anonymous_graph_seed11819.*` files are from original frozen Qwen hidden embeddings.
- The `qa19_anonymous_graph_whiten_seed11819.*` files are from whitened Qwen embeddings using an unchanged, six-address graph controller.
- `qa19_anonymous_graph_ood12_whiten_n6_seed11819.*` and `..._n8_...` keep the different node-count versions separate. During experimentation a filename collision was detected and corrected; the original six-address version was deterministically re-executed and recovered at **885/1000** and the eight-address version preserved at **869/1000**.
- `qa19_anonymous_graph_ood12_whiten_n8_reverselearned_seed11819.*` is the eight-address model that inferred the reversible-direction mapping from final answers.
- `qa19_anonymous_graph_ood12_whiten_n8_reversenone_seed11819.*` is its eight-address no-reverse control.
- `qa19_paired_statistics.json` includes exact McNemar and 10,000-draw paired bootstrap comparisons.
- `qa19_anonymous_slot_permutation_results.json` confirms an arbitrary address relabeling leaves the eight-address model's output numerically unchanged.

The original 1.5-GB Qwen3-0.6B model checkpoint is deliberately excluded. Its download, SHA-verification, segmentation and offline package-transfer procedures are preserved in `AlterProgramming/Forge`, branch `research/qwen3-context-rewrite-cpu-20261113`. This archive includes only trained small adapters, source, pretrained frozen sentence feature caches, and result records. No new training work is needed just to inspect the results.

### Source-of-truth limitation

The learned relation composer is an **external trained module** using frozen Qwen features and a preselected two-edge matrix-product computation. It is not a modification of Qwen3's own transformer parameters, nor does it establish a universally transferable self-rewrite mechanism. On QA2, the identical whitening algorithm performed worse with the prior recurrent network.