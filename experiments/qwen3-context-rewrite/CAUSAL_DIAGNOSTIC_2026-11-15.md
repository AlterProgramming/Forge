# Executed causal rewrite diagnostic — 2026-11-15

This is a **small CPU interface test, NOT Qwen3-0.6B training**. The model checkpoint was not loaded. The result cannot be counted as progress on Qwen's learned self-rewriting capability. Prior research branches and their results are untouched.

## Executed protocol
Code: `state_rewrite_causal_diagnostic.py`; 9,156 trainable parameters per model; PyTorch CPU, 2 threads; 500 AdamW steps at 0.003, gradient clipping 1.0, BCE over all intermediate outputs; 256 independently generated length-six training sequences, seeds 100–102. Three weight initializations seeds 20–22. Separate deterministic 512-sequence test sets for lengths 6, 12, 24 (seeds 1006–1008, 1012–1014, 1024–1026). No hyperparameter search on test sets. Each event specifies one of four keys and a binary value. Independent transformations: overwrite keyed value, toggle keyed bit. At each step, model must output all four current values.

Rewrite: `s <- s + f(s, e)`, then read `s`. Matched recurrent replacement: `s <- tanh(f(s,e))`, then read `s`. Matched nonmemory: `s <- f(0,e)`. All share identical encoder, update network, readout, 9,156 parameters, 500 optimization steps, batch, and data. Ablation rolls hidden state by one example across batch after every update.

## Executed results (bit accuracy, three-seed means, rounded)

| Task | Model | length 6 | length 12 | length 24 |
|---|---|---:|---:|---:|
| Overwrite | additive rewrite | 1.000 | 0.986 | 0.879 |
| Overwrite | recurrent replacement | 1.000 | 0.999 | **0.998** |
| Overwrite | event-only | 0.835 | 0.746 | 0.687 |
| Toggle | additive rewrite | 0.999 | 0.967 | 0.807 |
| Toggle | recurrent replacement | 0.998 | 0.985 | **0.943** |
| Toggle | event-only | 0.774 | 0.661 | 0.579 |

Length-24 shuffled-state accuracy fell to ~0.526–0.531 in both recurrent conditions, demonstrating reliance on example-specific state. Exact seed-level results are in the adjacent CSV. Local execution of final 18-condition experiment took about 20 seconds; hardware-independent FLOPs and Qwen token costs are **not measured**. The GitHub Actions job reruns the published source and retains a complete JSON artifact.

## Interpretation and obstruction
The state is trainable, updated repeatedly, and causally used by the same network at every step. But this does **not** show Qwen learned to rewrite its own hidden representation: the test uses a separate small neural network, supervised intermediate states, and narrow synthetic transformations. Additive rewrite generalizes **worse** than an equally sized recurrent replacement model on both length-24 tests. This is a negative control result, not a breakthrough. Neither method demonstrates task-independent state semantics or held-out domain transfer.

The next actual Qwen test must attach a learned rewrite/read interface to the original checkpoint, train from outcomes without intermediate state labels, and compare a compute-matched recurrent replacement and a frozen/nonupdated-state interface. A complete comparison also needs repeated latent edits, state swaps, independent tasks, multiple seeds, Qwen token/latency accounting, and official held-out data.
