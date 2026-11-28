/* global BABYLON, AvatarForgeHairDynamics, AvatarForgeHairTexture */
(function () {
  "use strict";
  const canvas = document.getElementById("canvas");
  const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer:true, stencil:true });
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.035,0.045,0.065,1);
  const camera = new BABYLON.ArcRotateCamera("camera", -Math.PI/2, Math.PI/2.25, 2.6, new BABYLON.Vector3(0,1.5,0), scene);
  camera.attachControl(canvas, true); camera.minZ=0.01; camera.wheelDeltaPercentage=0.02;
  new BABYLON.HemisphericLight("fill", new BABYLON.Vector3(0,1,0), scene).intensity=0.75;
  const key = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-0.5,-0.8,-0.5), scene); key.intensity=2.0;
  const rim = new BABYLON.DirectionalLight("rim", new BABYLON.Vector3(0.6,-0.25,0.8), scene); rim.intensity=0.8;

  let imported=[]; let dynamics=null; let texture=null; let fileUrl=null;
  const status=document.getElementById("status"), summary=document.getElementById("summary");
  const picker=document.getElementById("file");
  const labels=AvatarForgeHairTexture.LABELS;
  const select=document.getElementById("label"), coord=document.getElementById("coord");
  for (const label of labels) { const o=document.createElement("option"); o.value=label; o.textContent=label; select.append(o); }
  select.value="2B";

  function fit(meshes){
    let min=new BABYLON.Vector3(Infinity,Infinity,Infinity), max=new BABYLON.Vector3(-Infinity,-Infinity,-Infinity), ok=false;
    for(const mesh of meshes){ if(!mesh.getTotalVertices?.()) continue; mesh.computeWorldMatrix(true); const b=mesh.getBoundingInfo()?.boundingBox; if(!b) continue; min=BABYLON.Vector3.Minimize(min,b.minimumWorld); max=BABYLON.Vector3.Maximize(max,b.maximumWorld); ok=true; }
    if(!ok) return; const center=min.add(max).scale(0.5), size=max.subtract(min); camera.setTarget(center); camera.radius=Math.max(size.y*1.25,size.x*2.1,size.z*2.1,0.7);
  }
  function sync(){
    if(!texture?.available){ summary.textContent="No AvatarForgeHairTextureLab morph mesh found."; return; }
    const s=texture.labelSummary(); document.getElementById("coordOut").textContent=s.label; const nearest=Math.round(s.coordinate); select.value=labels[nearest];
    summary.textContent=`${s.label} · water ${s.products.water.toFixed(2)} · conditioner ${s.products.conditioner.toFixed(2)} · gel ${s.products.gel.toFixed(2)} · oil ${s.products.oil.toFixed(2)}`;
  }
  function product(name,id){ const el=document.getElementById(id); const apply=()=>{ document.getElementById(`${id}Out`).textContent=Number(el.value).toFixed(2); texture?.setProduct(name,el.value); sync(); }; el.oninput=apply; return (v)=>{el.value=v;apply();}; }
  const setWater=product("water","water"), setConditioner=product("conditioner","conditioner"), setGel=product("gel","gel"), setOil=product("oil","oil");

  async function load(file){
    if(fileUrl) URL.revokeObjectURL(fileUrl); fileUrl=URL.createObjectURL(file); status.textContent=`Loading ${file.name}…`;
    for(const node of imported) node.dispose?.(); imported=[]; dynamics?.dispose(); dynamics=null; texture=null;
    try {
      const result=await BABYLON.SceneLoader.ImportMeshAsync("","",fileUrl,scene); imported=[...(result.meshes||[]),...(result.transformNodes||[])];
      const meshes=result.meshes.filter(m=>m.getTotalVertices?.()>0); const hair=meshes.filter(m=>/AvatarForgeHairTextureLab|hair.?texture.?lab/i.test(m.name||""));
      dynamics=hair.length ? new AvatarForgeHairDynamics(hair,null,{strength:1,damping:8.8,enabled:true}) : null;
      texture=hair.length ? new AvatarForgeHairTexture.HairTextureController(hair,dynamics) : null;
      texture?.setCoordinate(coord.value); fit(meshes); status.textContent=`${file.name} · ${hair.length} texture-lab mesh${hair.length===1?"":"es"}`; sync();
    } catch(error){ status.textContent=`Load failed: ${error?.message||error}`; }
  }

  select.onchange=()=>{ const i=labels.indexOf(select.value); coord.value=String(i); texture?.setCoordinate(i); sync(); };
  coord.oninput=()=>{ texture?.setCoordinate(coord.value); sync(); };
  document.getElementById("open").onclick=()=>picker.click(); picker.onchange=()=>picker.files[0]&&load(picker.files[0]);
  document.body.addEventListener("dragover",e=>e.preventDefault()); document.body.addEventListener("drop",e=>{e.preventDefault(); if(e.dataTransfer.files[0]) load(e.dataTransfer.files[0]);});
  document.getElementById("soak").onclick=()=>setWater(1); document.getElementById("dry").onclick=()=>setWater(0);
  document.getElementById("clear").onclick=()=>{setWater(0);setConditioner(0);setGel(0);setOil(0);};
  const gust=document.getElementById("gust"); gust.oninput=()=>document.getElementById("gustOut").textContent=Number(gust.value).toFixed(2);
  document.getElementById("gustNow").onclick=()=>dynamics?.impulse(gust.value); document.getElementById("reset").onclick=()=>dynamics?.reset();
  engine.runRenderLoop(()=>{ const dt=engine.getDeltaTime()/1000; dynamics?.update(dt); scene.render(); }); window.addEventListener("resize",()=>engine.resize());
})();
