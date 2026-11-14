# Independent-task replication: minimal relational composition adapter for frozen Qwen3-0.6B

**Executed locally on CPU on November 13, 2026; exploratory result, not a preregistered test.**

## Research question

Does the QA2 intermediate-answer objective transfer directly to the independent Facebook bAbI Task 19 Path Finding benchmark? If not, does one small task-specific computational operation permit effective learned state manipulation without changing pretrained Qwen3?

## Conditions and data

- Original bAbI Task 19 en-10k, official train 10,000 (9,000 train, 1,000 validation, story-disjoint), official test 1,000 stories and questions.
- Frozen Qwen3-0.6B sentence/question representations. Unsupervised whitening on 150 sentence/question types; these types appear in training (thus this is *not* new-sentence OOD generalization).
- Eight *anonymous learned* node addresses. Neural binding of sentences and queries to graph nodes/directions. The graph operator and 2-edge path answer format are provided as **architectural priors**. Opposite compass directions were fixed in this comparison.
- 294,768 external trainable controller parameters, same input features, training/validation/test split, optimizer, initialization seed, and model parameters between the direct-only and two-hop composed variants.
- Minimal change: perform a **second differentiable graph traversal**, `h2 = einsum('bdj,bcjk->bdck', h1,G)`, where `h1` is the first relation traversal. This step adds **zero learned parameters**, but additional operations and a task-specific compositional prior.
- Auxiliary variant: apply 10% final-answer cross-entropy to the intermediate direct readout and 90% to the two-hop readout. No intermediate path answers, support-fact labels, or transition targets were provided. The intermediate readout itself need not represent the correct full path; the extra loss can act as regularization, so don't equate this with verified intermediate reasoning.

## Original generic-recurrence transfer (same Qwen input features)

The original repeated-rewrite model with eight recurrent updates scored **107/1,000** with final-step supervision and **112/1,000** when the same final labels supervised steps 1,2,4,8. This simple objective change did **not** clearly replicate the QA2 improvement.

## Results: task-specific composition adaptation

| Initialization seed | Direct-only (one hop) | Two-hop composition | Two-hop + 10% auxiliary loss |
|---|---:|---:|---:|
| 11819 | 283/1,000 | 938/1,000 | 987/1,000 |
| 11820 | Not rerun | 995/1,000 | 1,000/1,000 |
| 11821 | Not rerun | 990/1,000 | 1,000/1,000 |

- For seed 11819, composition vs direct control: 677 questions fixed and 22 worsened; paired exact McNemar/binomial p=1.90×10^-169, improvement +65.5 percentage points (question bootstrap 95% CI 62.3 to 68.7 pp).
- For seed 11819, auxiliary loss vs composed final-only: 57 fixed and 8 worsened, p=3.16×10^-10, +4.9 pp (95% CI 3.4 to 6.5 pp). Seed 11820: +5 questions, p=0.0625; seed 11821: +10 questions, p=0.00195. All tests used identical 1,000 questions within seed; three seeds are not independent test datasets.
- On the **same learned two-hop checkpoints**, removing the second traversal during testing causes accuracy to collapse: composed three seeds **137,130,117/1,000**; auxiliary models **214,239,228/1,000**. This is a strong causal ablation of the specific computation.
- The exact classical graph solver scores 1,000/1,000. This particular benchmark and architecture embed the known two-edge answer structure. Earlier training in this research program showed that unique random sentence identifiers can also achieve high in-distribution scores with this graph architecture. Therefore, this is **not** evidence Qwen3 alone invented graph structure or that pretrained semantics are required for in-vocabulary path solving.
- The official bAbI test split has been inspected in earlier experiments, so the full research program is exploratory and requires a newly frozen blind benchmark for confirmatory claims.

## Interpretation

**Successful targeted transfer of the *methodological idea*, not the learned solution.** An intermediate-answer loss alone did not improve a generic recurrent controller materially. Supplying an appropriate task-specific differentiable composition operation caused a large improvement, and adding the same auxiliary-output objective improved results across three tested initializations (magnitude varied). The adaptive ingredient is the representation and computation primitive, not a larger Qwen model or expensive GPU retraining.

This still falls short of a controller *discovering* its own representation and deciding when to build or invoke new operations. That must be tested with an unbiased operator menu and independent tasks/compute controls.

## Reproduction

`qa19_minimal_composition_transfer.py --method {direct,compose,compose_anytime} --feature whiten --nodes 8 --epochs 120 --seed {11819,11820,11821}`. It derives from `qa19_anonymous_graph.py`, with a per-iteration intermediate readout and one additional graph traversal. Offline feature file `qa19_frozen_qwen_sentence_embeddings.pt` and public original benchmark train/test files are included in the attached research archive. `qa19_minimal_adapter_testtime_ablation.py` reloads the stored best-validation checkpoints and directly removes the extra traversal at inference time. Model checkpoint weights, exact per-question accuracy vectors, and code are provided. Full Qwen3 pretrained checkpoint is excluded and can be reconstructed through prior sharded GitHub Actions transfer workflows.