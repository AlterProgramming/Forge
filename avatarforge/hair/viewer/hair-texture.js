/* global BABYLON */
(function (root) {
  "use strict";

  const LABELS = Object.freeze(["1A","1B","1C","2A","2B","2C","3A","3B","3C","4A","4B","4C"]);
  const clamp = (v, lo, hi) => Math.min(Math.max(Number(v) || 0, lo), hi);

  const textureMeta = Object.freeze({
    "1A": { curl:0.00, damping:1.00, tipPower:1.00, roughness:0.34 },
    "1B": { curl:0.07, damping:1.00, tipPower:1.02, roughness:0.35 },
    "1C": { curl:0.14, damping:1.02, tipPower:1.04, roughness:0.36 },
    "2A": { curl:0.23, damping:1.05, tipPower:1.08, roughness:0.37 },
    "2B": { curl:0.32, damping:1.08, tipPower:1.12, roughness:0.38 },
    "2C": { curl:0.41, damping:1.12, tipPower:1.17, roughness:0.39 },
    "3A": { curl:0.51, damping:1.18, tipPower:1.25, roughness:0.40 },
    "3B": { curl:0.60, damping:1.24, tipPower:1.34, roughness:0.42 },
    "3C": { curl:0.69, damping:1.31, tipPower:1.44, roughness:0.44 },
    "4A": { curl:0.78, damping:1.40, tipPower:1.57, roughness:0.46 },
    "4B": { curl:0.89, damping:1.52, tipPower:1.72, roughness:0.48 },
    "4C": { curl:1.00, damping:1.66, tipPower:1.90, roughness:0.50 },
  });

  function mix(a, b, t) { return a + (b-a)*t; }

  function metaAt(coordinate) {
    const x = clamp(coordinate, 0, LABELS.length-1);
    const i0 = Math.floor(x), i1 = Math.min(LABELS.length-1, i0+1), t = x-i0;
    const a = textureMeta[LABELS[i0]], b = textureMeta[LABELS[i1]];
    return {
      curl: mix(a.curl,b.curl,t),
      damping: mix(a.damping,b.damping,t),
      tipPower: mix(a.tipPower,b.tipPower,t),
      roughness: mix(a.roughness,b.roughness,t),
    };
  }

  class HairTextureController {
    constructor(meshes, dynamics) {
      this.meshes = (meshes || []).filter(mesh => mesh.morphTargetManager);
      this.dynamics = dynamics || null;
      this.coordinate = 4;
      this.products = { water:0, conditioner:0, gel:0, oil:0 };
      this.available = this.meshes.some(mesh => mesh.morphTargetManager?.numTargets >= 11);
      if (this.available) this.apply();
    }

    setDynamics(dynamics) { this.dynamics = dynamics || null; this.applyMaterialResponse(); }
    setCoordinate(value) { this.coordinate = clamp(value, 0, LABELS.length-1); this.apply(); }
    setLabel(label) {
      const index = LABELS.indexOf(String(label || "").toUpperCase());
      if (index >= 0) this.setCoordinate(index);
    }
    setProduct(name, value) {
      if (!(name in this.products)) return;
      this.products[name] = clamp(value, 0, 1);
      this.apply();
    }
    resetProducts() { this.products = { water:0, conditioner:0, gel:0, oil:0 }; this.apply(); }

    effectiveCoordinate() {
      const curlFraction = this.coordinate / (LABELS.length-1);
      const relaxation = this.products.water * 0.52 * Math.pow(curlFraction, 0.75);
      return clamp(this.coordinate - relaxation, 0, LABELS.length-1);
    }

    applyMorphs(coordinate) {
      for (const mesh of this.meshes) {
        const manager = mesh.morphTargetManager;
        if (!manager) continue;
        for (let i=0; i<manager.numTargets; i++) manager.getTarget(i).influence = 0;
        const x = clamp(coordinate, 0, LABELS.length-1);
        const lo = Math.floor(x), hi = Math.min(LABELS.length-1, lo+1), f = x-lo;
        if (lo > 0 && lo-1 < manager.numTargets) manager.getTarget(lo-1).influence += 1-f;
        if (hi > 0 && hi-1 < manager.numTargets) manager.getTarget(hi-1).influence += f;
      }
    }

    applyMaterialResponse() {
      const meta = metaAt(this.effectiveCoordinate());
      const { water, conditioner, gel, oil } = this.products;
      const roughness = Math.max(0.12, meta.roughness - 0.10*conditioner - 0.09*oil - 0.03*water);
      for (const mesh of this.meshes) {
        const material = mesh.material;
        if (!material) continue;
        if ("roughness" in material) material.roughness = roughness;
        if (material.anisotropy) {
          material.anisotropy.isEnabled = true;
          material.anisotropy.intensity = Math.min(1, 0.76 + 0.10*oil + 0.04*conditioner);
        }
      }
      if (this.dynamics?.setMaterialState) {
        this.dynamics.setMaterialState({
          curl: meta.curl,
          tipPower: meta.tipPower,
          dampingScale: meta.damping * (1 + 0.24*water + 0.16*conditioner + 0.42*gel + 0.07*oil),
          stiffnessScale: Math.max(0.72, 1 + 0.50*gel - 0.09*water),
          massScale: 1 + 0.36*water + 0.05*oil,
          cohesion: Math.min(1, 0.30 + 0.38*meta.curl + 0.18*conditioner + 0.38*gel),
        });
      }
    }

    apply() {
      if (!this.available) return;
      this.applyMorphs(this.effectiveCoordinate());
      this.applyMaterialResponse();
    }

    labelSummary() {
      const x = this.effectiveCoordinate();
      const lo = Math.floor(x), hi = Math.min(LABELS.length-1, lo+1), f=x-lo;
      const label = f < 0.02 ? LABELS[lo] : f > 0.98 ? LABELS[hi] : `${LABELS[lo]}→${LABELS[hi]}`;
      return { label, coordinate:x, products:{...this.products}, meta:metaAt(x) };
    }
  }

  root.AvatarForgeHairTexture = Object.freeze({ LABELS, HairTextureController, metaAt });
  if (typeof module === "object" && module.exports) module.exports = { LABELS, HairTextureController, metaAt };
})(typeof globalThis !== "undefined" ? globalThis : this);
