# Avatar Forge Hair Research

This branch is the source authority for the Avatar Forge hair experiments.

## Current causal architecture

The texture of a fiber must emerge from its microscopic/material state. The current direction is:

`cross-section + cortex differential strain/orientation + hydration + heterogeneity`
`-> intrinsic rest strain field kappa1(s), kappa2(s), tau(s)`
`-> Cosserat-style natural fiber integration`
`-> mechanics/contact/gravity/product interactions`
`-> observed strand/lock/groom appearance`

`avatarforge_hair/micro_fiber.py` implements the first version of that bridge. It does **not** accept curl radius, cycle count, or a kink target. Those are outcomes to measure from the resulting centerline.

The familiar 1A-4C vocabulary may remain as an editor/readout convention, but it must not be the geometry generator. A label can select or summarize a region of microscopic state space; it does not prescribe the final curve.

## Important invalidation

The earlier v11 `generate_texture_lab.py` morph-target path directly prescribed fields such as coil radius, cycles per 10 cm, and kink. That was useful for testing topology/runtime plumbing but is **not an acceptable causal hair model** and must not be treated as the production texture generator.

Likewise, the first 1A-4C atlas is invalid visual evidence. Its high-curvature states mixed numerical undersampling with prescribed geometry.

## What remains reusable

- real `AvatarHead` scalp sampling and stable roots;
- hairstyle guide / scalp-flow work;
- guide-to-dense reconstruction architecture;
- collision and secondary-dynamics infrastructure;
- product/environment state plumbing;
- standalone Babylon viewer and tests.

These layers should consume micro-derived intrinsic fiber state rather than manufacture texture themselves.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/*.test.js
```

`tests/test_micro_fiber.py` verifies that geometry changes when microscopic differential strain/orientation changes, and that water first changes the microstate before the natural shape is recomputed.

See `SOURCE-PROVENANCE.md` for pre-repository lineage.
