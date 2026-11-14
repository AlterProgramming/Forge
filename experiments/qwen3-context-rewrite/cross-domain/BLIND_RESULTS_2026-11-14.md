# Qwen3-0.6B blind cross-family validation — 2026-11-14

**Status: Executed and independently evaluated, no reliable improvement established.**

GitHub Actions run: https://github.com/AlterProgramming/Forge/actions/runs/43629177966

## Frozen evaluation protocol

Official `Qwen/Qwen3-0.6B`, no weight updates. Three independent published test sets: GSM8K (math), ARC-Challenge (science), MBPP (Python programming). Six new test problems sampled from each with seed 20261115, with the twelve formerly inspected prompts excluded before inference.

All conditions used the *same pretrained model*, under one fixed protocol:
1. **Plain** non-thinking, max 160 generated tokens;
2. **Thinking** Qwen3 native thinking mode, max 512 generated tokens;
3. **Thinking + forced finalization**, exactly the same thinking trace, plus a non-thinking second call (max 192 generated tokens) where the first pass did not produce a final answer.

MBPP candidate continuation may consider one development test; scoring uses remaining held-out tests. Correct answers / hidden evaluation tests were not in prompts.

## Measured results

| Official test set | Plain | Thinking | Thinking + completion |
|---|---:|---:|---:|
| GSM8K math | 1/6 | 3/6 | 2/6 |
| ARC-Challenge | 6/6 | 5/6 | 6/6 |
| MBPP Python | 1/6 | 0/6 | 1/6 |
| **Total** | **8/18** | **8/18** | **9/18** |

**Additional output-token accounting**: plain 821; thinking 7,805; completion 430; complete enhanced pipeline 8,235, roughly **10.0×** the plain approach. Input/prompt-token overhead is additional and was logged in the result artifacts.

The forced completion won one extra correct task versus cheaper plain Qwen, and did not improve on the plain condition on ARC or MBPP. For one GSM8K example the first reasoning trace contained a parsable correct number but forcing finalization changed it to an incorrect answer. Incomplete thinking traces are not necessarily user-facing complete answers; counting a number extracted from an incomplete trace as a correct "thinking" result is therefore a lenient diagnostic and should not be interpreted as actual complete-response accuracy.

## Interpretation and next research gate

The earlier **3/12→7/12** result on the twelve repeatedly inspected problems was an *exploratory, post-hoc intervention*. The frozen procedure did not reproduce a persuasive, cost-efficient gain on these independent samples. Neither model weights nor their general-purpose capability were demonstrated to have improved.

The generic arbitrary-prompt inference CLI is at [PR #6](https://github.com/AlterProgramming/Forge/pull/6). It has passed syntax and unit smoke tests but its cross-domain performance is **not proven** by this blind trial.

**Do not claim** a broadly effective Qwen performance upgrade on these results. Larger independent samples and an equal-compute comparator would be needed to support such a claim. This report and the original artifacts preserve failed as well as successful cases.

Raw artifacts attached to Actions run:
- GSM8K: 14127911459
- ARC: 14127362307
- MBPP: 14128285025
