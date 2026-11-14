# Frozen Qwen3-0.6B can use learned continuous working memory — November 13, 2026

## Results from executed local CPU experiments

Original Facebook bAbI Task 2 en-10k: 9,000 training QA pairs, story-disjoint 1,000-question validation, official 1,000-question test.

- An upstream 7-slot state editor (177,074 learned parameters) receives frozen pretrained Qwen3 sentence features, and was trained with exact intermediate state-transition labels. This is a **task-structured, supervised writer**, not autonomous ontology invention.
- A 55,506-parameter neural closure trained **from final QA answers only** learns to transform that seven-slot memory into three object-location distributions. Five independent training seeds on the official test score **972, 962, 975, 965, 962 / 1,000**. The selected original run scores 975/1,000.
- Convert the closure outputs into **three continuous vectors in Qwen3's frozen embedding space**; use inputs_embeds (no serialized state text) as Qwen3's memory prefix. The original Qwen3-0.6B weights are completely frozen.
- On the same preseeded 120 held-out official questions, greedy free generation, proper non-thinking ChatML boundaries: **continuous memory 110/120; corresponding text state 106/120; continuous state swapped between questions 29/120**.
- Paired latent versus text exclusive wins 5 vs 1, exact p=0.21875 (no demonstrated advantage over text). Latent vs swapped exclusive wins 81 vs 0, exact p=8.27e-25 (causal state dependence).
- Average total Qwen input positions: **41** continuous, **40** text. This is not an input-length or latency advantage in this implementation.
- Ablation on three withheld action verbs: state editor with Qwen pretrained hidden features **986/1,000**, with dimension-matched random bag-of-words sentence features **936/1,000**. Specific unseen 'grabbed' event exact edits: 220/234 vs 58/234. Qwen benefit is not uniform for every withheld verb.

## Interpretation

Positive demonstration of **frozen-pretrained-model reading of a changed continuous representation**. The 3 dynamic vectors are computed by a learned map, and memory corruption destroys correct generation. **Not** a discovery that pretrained Qwen learns its own memory ontology, that all possible improvements already exist in its weights, that token processing is eliminated from all inputs, or that this architecture beats a classical exact state machine (1000/1000). The semantic anchors are learned from Qwen's original lexical embedding space. There is no significant accuracy gain over compact text memory, and no context-position savings.

## Reproduction

Local experiment archive: `qwen3_latent_memory_continuous_results.zip` attached in the research ChatGPT conversation on 2026-11-13. Scripts include `qwen3_learned_slot_editor.py`, `qwen3_neural_closure_to_latent.py`, `qwen3_neural_closure_generate_nothink.py`, `qwen3_neural_closure_generate_nothink_swapped.py`, `qwen3_neural_closure_multiseed.py`, and `qwen3_pretrained_vs_random_ood.py`, with source logs, exact JSON results, and small trained checkpoints. Official Qwen3-0.6B original weights/data are not redistributed. Existing GitHub Actions export workflows in this research branch enable restoring the CPU checkpoint.

The next methodological requirement is a shared learned read–**self-rewrite**–read mechanism trained without predefined seven-slot semantics, validated against matched controls and independent tasks.
