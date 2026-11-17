# Mixed-operation state rewrite diagnostic — 2026-11-17

## Execution
Executed locally on CPU using PyTorch 2.10.0+cpu, two CPU threads; wall time 15.120 seconds. Reproduction: `python experiments/qwen3-context-rewrite/rewrite_mixed_20261117.py --steps 400 --seeds 3`. The published script writes its JSON in the working directory; the executed local script wrote it to /mnt/data; computations are otherwise identical. Three seeds (2026–2028), 256 train sequences of length 6 per seed, train data seeds 100–102; 512 held-out sequences per length/task/seed, evaluation seeds `10000+100*seed+length+task_offset` (mixed=0, overwrite=1, toggle=2). No pretrained Qwen weights loaded.

All modes share 9,156 trainable parameters: encoder 10→64→32, update 64→64→32, readout 32→4. AdamW lr .003, 400 full-batch updates, gradient clipping 1, binary cross entropy on every step. Mixed task uses keyed overwrite and XOR toggle randomly per step. Modes: additive state-dependent `s+=f(s,e)`, additive state-blind `s+=f(0,e)`, recurrent replacement `s=f(s,e)`. Same network evaluations and optimizer; state-blind has less state-dependent arithmetic, replacement has a different state accumulation operator. These are architecture-matched, not exactly FLOP-matched.

## Executed seed-level results
Columns: mixed L6/L24/L48 accuracy, mixed L24 forced-blind/shuffled-halfway/zeroed-halfway accuracy, overwrite L48, toggle L48. Each value is bit accuracy over all time steps (not final-state exact-match).

| mode | seed | train loss | L6 | L24 | L48 | blind | shuffle | zero | overwrite48 | toggle48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| additive dependent |0|.000411|.993734|.762553|.633382|.580973|.745361|.820353|.632151|.620819|
| additive dependent |1|.000587|.987142|.793681|.662181|.573181|.759460|.818359|.684489|.601725|
| additive dependent |2|.000386|.998372|.843384|.679901|.571147|.793274|.853780|.706197|.639425|
| additive blind |0|.310735|.913818|.663371|.614054|.663371|.655863|.679810|.657654|.543803|
| additive blind |1|.290151|.908529|.675029|.615306|.675029|.663859|.691854|.672597|.540334|
| additive blind |2|.326957|.906657|.676412|.622833|.676412|.670695|.692485|.680786|.540700|
| replacement |0|.000654|.992839|.938680|.917562|.641296|.839437|.848328|.951650|.722432|
| replacement |1|.000542|.995931|.962240|.942190|.636332|.853617|.860372|.985036|.780955|
| replacement |2|.000366|.998861|.975281|.959147|.635498|.857625|.866109|.983287|.841929|

## Conclusion and obstruction
Mixed L24 means: dependent additive .79987, blind additive .67160, replacement .95873. Mixed L48 means: .65849, .61740, .93963. The causal blind intervention degrades dependent additive (.79987→.57510), but its midpoint state zeroing *improves* L24 accuracy (.79987→.83083), suggesting uncontrolled drift in additive memory. Replacement dominates both and is much more stable at 8× training length. This is a negative result for additive latent rewriting as the preferred update form, not evidence of Qwen self-rewriting. This task is synthetic with fixed four-key semantics; independent semantic transfer and Qwen readback were not tested. Exact FLOP, energy, and peak memory accounting were not measured; training forward passes: 3 modes × 3 seeds × 400 steps × 256 examples × 6 steps = 5,529,600 example-steps; evaluation: 3 modes × 3 seeds × 3 tasks × 3 lengths × 512 examples × 4 interventions = 165,888 sequences, each evaluated at its task length. GPU cost zero. No claim of robust generalization or novel capability.

Next obstruction: Qwen must consume a learned state after each update. A true test requires loading the SHA-verified checkpoint into a network-enabled runner, integrating a readback pathway, and comparing matched replacement, additive and blind controls on independent held-out tasks. Preserve prior branches unchanged.
