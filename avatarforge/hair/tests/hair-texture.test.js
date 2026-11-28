const test = require('node:test');
const assert = require('node:assert/strict');
const { LABELS, HairTextureController, metaAt } = require('../viewer/hair-texture.js');

function fakeMesh() {
  const targets = Array.from({length: 11}, (_, i) => ({ name: LABELS[i+1], influence: 0 }));
  return {
    morphTargetManager: {
      numTargets: targets.length,
      getTarget(i) { return targets[i]; },
    },
    material: { roughness: 0.36, anisotropy: { isEnabled: false, intensity: 0.8 } },
    _targets: targets,
  };
}

test('exposes the 1A through 4C editor anchors', () => {
  assert.deepEqual(LABELS, ['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']);
});

test('interpolates between adjacent texture targets on one topology', () => {
  const mesh = fakeMesh();
  const controller = new HairTextureController([mesh], null);
  controller.setCoordinate(4.5);
  assert.equal(mesh._targets[3].influence, 0.5);
  assert.equal(mesh._targets[4].influence, 0.5);
  assert.equal(mesh._targets.reduce((s,t)=>s+t.influence,0), 1);
});

test('water causes a bounded temporary move toward a looser anchor', () => {
  const mesh = fakeMesh();
  const controller = new HairTextureController([mesh], null);
  controller.setCoordinate(11);
  const dry = controller.effectiveCoordinate();
  controller.setProduct('water', 1);
  const wet = controller.effectiveCoordinate();
  assert.ok(wet < dry);
  assert.ok(wet > 10);
});

test('texture metadata changes motion response as texture tightens', () => {
  assert.ok(metaAt(11).tipPower > metaAt(3).tipPower);
  assert.ok(metaAt(11).damping > metaAt(3).damping);
});
