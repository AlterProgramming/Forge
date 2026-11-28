/* global BABYLON */
(function (root) {
  "use strict";

  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

  class Spring2D {
    constructor() { this.x = 0; this.z = 0; this.vx = 0; this.vz = 0; }
    reset() { this.x = this.z = this.vx = this.vz = 0; }
    step(dt, tx, tz, stiffness, damping) {
      const ax = stiffness * (tx - this.x) - damping * this.vx;
      const az = stiffness * (tz - this.z) - damping * this.vz;
      this.vx += ax * dt; this.vz += az * dt;
      this.x += this.vx * dt; this.z += this.vz * dt;
    }
  }

  class HairDynamics {
    constructor(meshes, runtime, options) {
      this.runtime = runtime || null;
      this.meshes = [];
      this.enabled = true;
      this.strength = 1;
      this.stiffness = 46;
      this.damping = 8.8;
      this.maxOffset = 0.028;
      this.materialState = { curl:0.32, tipPower:1.12, dampingScale:1, stiffnessScale:1, massScale:1, cohesion:0.45 };
      this.frame = 0;
      this.lastHeadPos = null;
      this.lastHeadRot = null;
      this.lastVelocity = BABYLON.Vector3.Zero();
      this.lockStates = [];
      Object.assign(this, options || {});
      this._prepare(meshes || []);
    }

    _prepare(meshes) {
      let maxLocks = 0;
      for (const mesh of meshes) {
        const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
        const uv = mesh.getVerticesData(BABYLON.VertexBuffer.UVKind);
        if (!positions || !uv || uv.length / 2 !== positions.length / 3) continue;
        const rest = new Float32Array(positions);
        const working = new Float32Array(rest);
        const count = rest.length / 3;
        const tipWeight = new Float32Array(count);
        const rootCoordinate = new Float32Array(count);
        const lockValue = new Float32Array(count);
        const unique = [];
        const lookup = new Map();
        for (let i = 0; i < count; i++) {
          const q = clamp(uv[i*2], 0, 1);
          const key = Math.round(uv[i*2+1] * 100000) / 100000;
          if (!lookup.has(key)) { lookup.set(key, unique.length); unique.push(key); }
          rootCoordinate[i] = q;
          tipWeight[i] = Math.pow(q, this.materialState.tipPower || 1.72);
          lockValue[i] = lookup.get(key);
        }
        maxLocks = Math.max(maxLocks, unique.length);
        mesh.setVerticesData(BABYLON.VertexBuffer.PositionKind, Array.from(rest), true);
        this.meshes.push({ mesh, rest, working, tipWeight, rootCoordinate, lockValue, lockCount:unique.length });
      }
      this.lockStates = Array.from({ length:maxLocks }, () => new Spring2D());
    }

    setEnabled(value) { this.enabled = Boolean(value); if (!this.enabled) this.reset(); }
    setStrength(value) { this.strength = clamp(Number(value) || 0, 0, 2.5); }
    setDamping(value) { this.damping = clamp(Number(value) || 0, 1, 24); }
    setMaterialState(state) {
      Object.assign(this.materialState, state || {});
      const power = clamp(Number(this.materialState.tipPower) || 1.72, 0.7, 2.5);
      for (const item of this.meshes) {
        for (let i=0; i<item.tipWeight.length; i++) item.tipWeight[i] = Math.pow(item.rootCoordinate[i], power);
      }
    }

    impulse(amount) {
      const strength = clamp(Number(amount) || 1, 0, 3);
      for (let i=0; i<this.lockStates.length; i++) {
        const state = this.lockStates[i];
        const phase = i * 2.399963229728653;
        state.vx += Math.cos(phase) * 0.18 * strength;
        state.vz += Math.sin(phase * 0.73 + 0.8) * 0.13 * strength;
      }
    }

    reset() {
      for (const state of this.lockStates) state.reset();
      for (const item of this.meshes) {
        item.working.set(item.rest);
        item.mesh.updateVerticesData(BABYLON.VertexBuffer.PositionKind, item.working, false, false);
        item.mesh.refreshBoundingInfo();
      }
      this.lastHeadPos = null; this.lastHeadRot = null; this.lastVelocity.set(0,0,0);
    }

    _headKinematics(dt) {
      if (!this.runtime?.getBoneWorldTransform) return { accel:BABYLON.Vector3.Zero(), omega:BABYLON.Vector3.Zero() };
      const tr = this.runtime.getBoneWorldTransform("Head");
      if (!tr) return { accel:BABYLON.Vector3.Zero(), omega:BABYLON.Vector3.Zero() };
      const pos = tr.position, rot = tr.rotation;
      if (!this.lastHeadPos || !this.lastHeadRot || dt <= 1e-5) {
        this.lastHeadPos = pos.clone(); this.lastHeadRot = rot.clone();
        return { accel:BABYLON.Vector3.Zero(), omega:BABYLON.Vector3.Zero() };
      }
      const velocity = pos.subtract(this.lastHeadPos).scale(1/dt);
      const accel = velocity.subtract(this.lastVelocity).scale(1/dt);
      this.lastVelocity.copyFrom(velocity);
      const invPrev = this.lastHeadRot.clone(); invPrev.conjugateInPlace();
      const delta = invPrev.multiply(rot); if (delta.w < 0) delta.scaleInPlace(-1);
      const omega = delta.toEulerAngles().scale(1/dt);
      this.lastHeadPos.copyFrom(pos); this.lastHeadRot.copyFrom(rot);
      return { accel, omega };
    }

    update(dt) {
      if (!this.enabled || !this.meshes.length) return;
      dt = clamp(Number(dt) || 0, 0, 1/24); if (!dt) return;
      this.frame++;
      const { accel, omega } = this._headKinematics(dt);
      const baseX = clamp((-omega.z*0.012 - accel.x*0.00013)*this.strength, -this.maxOffset, this.maxOffset);
      const baseZ = clamp(( omega.x*0.010 - accel.z*0.00013)*this.strength, -this.maxOffset, this.maxOffset);
      for (let li=0; li<this.lockStates.length; li++) {
        const phase = li*1.61803398875;
        const scale = 0.82 + 0.18*Math.sin(phase);
        const cross = 0.12*Math.cos(phase*0.71);
        const tx = clamp(baseX*scale + baseZ*cross, -this.maxOffset, this.maxOffset);
        const tz = clamp(baseZ*scale - baseX*cross, -this.maxOffset, this.maxOffset);
        const ms = this.materialState;
        const stiffness = this.stiffness * (ms.stiffnessScale || 1) / Math.max(0.65, ms.massScale || 1);
        const damping = this.damping * (ms.dampingScale || 1);
        this.lockStates[li].step(dt, tx, tz, stiffness, damping);
      }
      for (const item of this.meshes) {
        const { rest, working, tipWeight, lockValue, lockCount, mesh } = item;
        for (let i=0; i<tipWeight.length; i++) {
          const state = this.lockStates[Math.min(lockValue[i] | 0, lockCount-1)] || this.lockStates[0];
          const w = tipWeight[i], k = i*3;
          working[k] = rest[k] + state.x*w;
          working[k+1] = rest[k+1] + (Math.abs(state.x)+Math.abs(state.z))*0.055*w*w;
          working[k+2] = rest[k+2] + state.z*w;
        }
        mesh.updateVerticesData(BABYLON.VertexBuffer.PositionKind, working, false, false);
        if ((this.frame & 7) === 0) mesh.refreshBoundingInfo();
      }
    }

    dispose() { this.reset(); this.meshes.length = 0; this.lockStates.length = 0; }
  }

  root.AvatarForgeHairDynamics = HairDynamics;
  if (typeof module === "object" && module.exports) module.exports = { HairDynamics, Spring2D };
})(typeof globalThis !== "undefined" ? globalThis : this);
