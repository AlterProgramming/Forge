# Qwen3-0.6B cross-domain performance intervention (2026-11-14)

## Objective

Improve useful task performance on **unrelated benchmark families**. The method is
deliberately not restricted to representation learning or bAbI graph traversal.

This is **inference-time augmentation** of *frozen, official*
\`Qwen/Qwen3-0.6B\` weights, not a new checkpoint. The entire method is in
\`boost.py\`. It uses the same underlying model for all three families.

## Independent public test sets

- **GSM8K** (\`openai/gsm8k\`, \`main\`, \`test\`): math word problems. The
  model produces numerical answers. Unseen ground truth is used for scoring only.
- **ARC-Challenge** (\`allenai/ai2_arc\`, \`ARC-Challenge\`, \`test\`):
  multiple-choice science questions. Gold choices are used for scoring only.
- **MBPP** (\`google-research-datasets/mbpp\`, \`full\`, \`test\`):
  program synthesis. The first supplied unit test is a *development* verifier
  for candidate selection; the remaining tests are held out from selection
  and used for evaluation. Test case inputs or outputs are never shown in the
  prompt; only the first test's function name is extracted.

No benchmark examples are used for training. The initial smoke trial samples
4 examples per family with a fixed RNG seed. This is exploratory, far too
small for a reliable generalization claim. No reported score should be
described as an improvement unless the actual run demonstrates it.

## Protocol and fairness

For each problem, sample \`k=3\` outputs from *the identical model* at
temperature 0.7, top-p 0.8, top-k 20, non-thinking mode. Baseline is **the
first** of the generated candidates; it is a valid single-sample baseline.
For GSM8K and ARC use plurality of parsed final answers, resolving ties by
first occurrence. For MBPP use the first candidate passing the development
test, falling back to the first candidate if all fail.

Score both baseline and boosted answers on identical hidden labels or code
tests. Log every candidate, input/generated token counts, selected index,
task IDs, prompt, correctness, and elapsed time as a JSON workflow artifact.

**Important limitations:** Multi-sampling consumes roughly 3 times the
generation budget. "First candidate" is **not an equal-compute baseline**.
Plurality selection is not guaranteed to improve accuracy; if all candidates
are wrong, no selection method fixes this. For MBPP, selection on one
development test can overfit that test and fail held-out tests. Code execution
is intrinsically risky and the subprocess timeout is NOT a secure sandbox:
the Github Actions runner is a disposable workflow with read-only repo
permissions. Do not execute the code-verification mode against untrusted
outputs on your personal computer without real OS/container isolation.

## Running

Test parser/selection logic without downloading the model:

\`\`\`sh
python boost.py --self-test
\`\`\`

Run the official Qwen3 model and the public test datasets (Python 3.11):

\`\`\`sh
python -m pip install "torch>=2.4" "transformers>=4.51,<5" accelerate datasets
python boost.py --model Qwen/Qwen3-0.6B --n 4 --k 3 \
  --seed 20261114 --output results.json
\`\`\`

The repository workflow \`.github/workflows/qwen3-cross-domain.yml\` runs the
same study on GitHub Actions CPU and archives full outputs. For a more
credible analysis, expand to multiple fixed seeds and at least 100
examples/family with confidence intervals; include an equal-compute
comparison and a larger unmodified-Qwen baseline before publishing
performance claims.

## Decision gate

Only claim cross-domain improvement if held-out scores actually improve on
**multiple** independent families, the gain survives seed variation, and
total inference cost is disclosed. Otherwise retain the full negative
outcome without converting it into a "representation breakthrough".
