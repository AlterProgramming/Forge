# Avatar Forge Hair Texture Lab v11

A source-controlled hair generation/editor prototype for the real Avatar Forge model.

## Current architecture

`real AvatarHead scalp -> canonical roots -> style guides -> continuous strand texture -> lock/follower reconstruction -> dynamics -> render`

Texture and hairstyle are separate. The familiar 1A–4C labels are editor anchors in one continuous texture space, not biological categories. One rooted topology can interpolate between anchors.

## Generate

```bash
python -m tools.generate_texture_lab out/me.glb out/me.hair-v11.medium.glb --strands 600 --length medium
python -m tools.generate_texture_lab out/me.glb out/me.hair-v11.short.glb --strands 600 --length short
```

The interactive topology is intentionally guide-density. Dense follower reconstruction is a render concern and should not multiply every morph target.

## Viewer

```bash
python -m tools.studio
```

Hair controls include:

- 1A through 4C texture anchors;
- continuous interpolation between anchors;
- water/rain, conditioner, gel, and oil response controls;
- original cards / generated strands comparison;
- secondary lock dynamics and gust stress test.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/*.test.js
```

See `SOURCE-PROVENANCE.md` before treating an older generated asset as source authority.
