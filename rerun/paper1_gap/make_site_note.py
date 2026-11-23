#!/usr/bin/env python3
"""make_site_note.py <cellobs.json> <out.html> -- the CERN-site note on chamber A's two surfaces: an interactive 3-D view
of the fitted mesh and cathode with their drift lines, maps of what the geometry does to the drift field, the measured
maps, the consistency table and the tracking result.  Self-contained (inline JS/SVG, data embedded), works offline.
Inputs: results/{compare,joint,track_grid,track_test}.json of this package and cellobs.json (EOS
x17rerun/paper1_gap/caches/g_det3_wknd/cellobs.json).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def nan2none(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, (list, tuple)):
        return [nan2none(v) for v in x]
    if isinstance(x, dict):
        return {k: nan2none(v) for k, v in x.items()}
    return x


def data(cellobs_path):
    C = json.load(open(f'{HERE}/results/compare.json'))
    J = json.load(open(f'{HERE}/results/joint.json'))
    TG = json.load(open(f'{HERE}/results/track_grid.json'))
    TT = json.load(open(f'{HERE}/results/track_test.json'))['metrics']
    O = json.load(open(cellobs_path))
    # measured 4x4 maps from gapcompare (deviations, both runs combined)
    maps = {}
    for m in ('G', 'GY', 'VT', 'T', 'V'):
        maps[m] = [dict(x=r[0], y=r[1], v=r[2], e=r[3]) for r in C['grids']['4x4'][m]['comb']]
    # cellobs: mesh height (positions) and the drift-tilt offset w0, each run and view centred on its own mean
    def centred(key):
        vals = {}
        for run, rows in O.items():
            arr = np.array([r[key] if r[key] is not None else np.nan for r in rows], float)
            arr = arr - np.nanmean(arr)
            for r, a in zip(rows, arr):
                vals.setdefault((round(r['U'], 3), round(r['W'], 3)), []).append(a)
        return [dict(x=100 * u + 200, y=100 * w + 200, v=float(np.nanmean(v)) if np.isfinite(v).any() else None)
                for (u, w), v in vals.items()]
    for k in ('hp_x', 'hp_y', 'w0_x', 'w0_y'):
        maps[k] = centred(k)
    prof = {}
    for g in ('x5', 'y5'):
        prof[g] = {m: [dict(x=r[0], y=r[1], v=r[2], e=r[3]) for r in C['grids'][g][m]['comb']] for m in ('G', 'GY', 'VT', 'T')}
    arms = {}
    for a in ('all', 'tilt', 'gap', 'geo', 'no_hp', 'm3dish'):
        F = J['arms'][a]
        arms[a] = dict(params=F['params'], errors=F['errors'], chi2=F['chi2'], ndf=F['n'] - F['nfree'])
    trk = {}
    for run in ('run1', 'run2'):
        for p in 'xy':
            R = TG[run][p]
            trk[f'{p}_{run}'] = dict(pass1=R['pass1']['s68'], model=TT[run][p]['model']['s68'],
                                     grid={n: R[f'{n}x{n} w0+v']['s68'] for n in (4, 6, 8, 10)},
                                     w0only=R['8x8 w0']['s68'], vonly=R['8x8 v']['s68'])
    return nan2none(dict(maps=maps, prof=prof, arms=arms, trk=trk, floors=J['floors'], chamber=C['chamber']))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chamber A: mesh, cathode and drift field</title>
<meta name="description" content="The drift gap of MX17 chamber A across its surface: one model of the bowed readout board and the cathode fits every measurement, shows what the geometry does to the drift field, and a finer drift-direction map improves the angles by 16-21 %.">
<style>
:root{--bg:#fbfbfa;--fg:#1b2430;--mut:#5b6573;--line:#d9dde3;--card:#ffffff;--acc:#0b3d91;--acc2:#d55e00;--hi:#eef3fb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#12161c;--fg:#e4e8ee;--mut:#9aa5b3;--line:#2c333d;--card:#1a2029;--acc:#7fa8f0;--acc2:#f08a4b;--hi:#1e2836}}
:root[data-theme=dark]{--bg:#12161c;--fg:#e4e8ee;--mut:#9aa5b3;--line:#2c333d;--card:#1a2029;--acc:#7fa8f0;--acc2:#f08a4b;--hi:#1e2836}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 80px}h1{font-size:1.75rem;margin:.2em 0 .1em}h2{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.2em}
h3{margin:1.4em 0 .4em}.sub{color:var(--mut);margin:0 0 1.2em}.verdict{background:var(--hi);border-left:4px solid var(--acc);padding:.8em 1.1em;border-radius:4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:14px 0}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:.9rem;margin:6px 0}.controls label{display:flex;gap:6px;align-items:center}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit;font-size:.88rem}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
canvas{display:block;width:100%;touch-action:none;border-radius:6px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.cap{font-size:.86rem;color:var(--mut);margin-top:6px}table{border-collapse:collapse;font-size:.88rem;margin:.6em 0;width:100%}
td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}td:first-child,th:first-child{text-align:left}
.tw{overflow-x:auto}.tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);padding:4px 8px;border-radius:5px;font-size:.82rem;display:none;z-index:9;box-shadow:0 2px 8px #0003}
.mt{font-size:.9rem;font-weight:600;margin-bottom:4px}.k{font-family:ui-monospace,Menlo,monospace;font-size:.85em}
.theme{position:fixed;top:10px;right:12px}ul li{margin:.25em 0}
</style></head><body>
<button class="theme" id="thm" title="toggle light / dark">◐</button>
<main>
<h1>Chamber A: the mesh, the cathode and the drift field</h1>
<p class="sub">MX17 cosmic bench, chamber A (det3), the two long runs at the operating point (490 V mesh, 1000 V drift), final reconstruction (pass 1). 2026-11-24. Paper I, Sec. 5.1 and the appendix "The mesh and the cathode of chamber A".</p>

<div class="verdict"><b>What we found.</b> The drift gap of chamber A is 28.2 mm on average, but it is not the same everywhere. It runs from about 27 to about 29 mm over the chamber. We describe the chamber with two surfaces: the mesh, which sits on the readout board, and the cathode. One model of these two surfaces describes every measurement we have: the positions, the drift direction, the geometric gap, the drift time and the drift velocity. The readout board is dished, and its dish is measured twice, once with the reference telescope and once without it. Along y the cathode stays nearly flat above the dished board, so the gap is longest in the middle. Along x the cathode follows the board, and the gap only increases slowly with x. Finally, a finer map of the drift direction improves the reconstructed angles of chamber A by 16 to 21 %.</div>

<h2>1. The geometry in 3-D</h2>
<p>The figure below draws the two fitted surfaces. The lower surface is the mesh, and the upper surface is the cathode, 28 mm above it. Their shapes are tiny compared with the gap (about 1 mm over 40 cm), so the deformations are exaggerated by the factor on the slider. The gap itself is drawn to scale. The cathode is coloured by the local gap. The lines between the surfaces are drift lines. A drift line leaves the mesh at right angles to the mesh and ends at right angles to the cathode, so its shape follows both surfaces. Its sideways displacement is exaggerated by the same factor. Drag to rotate the view, and scroll to zoom.</p>
<div class="card">
<div class="controls">
<label>exaggeration <input type="range" id="exag" min="1" max="60" value="10"><span id="exv">10×</span></label>
<button id="bMesh" class="on">mesh</button><button id="bCath" class="on">cathode</button><button id="bLines" class="on">drift lines</button><button id="bSpin">rotate</button><button id="bReset">reset view</button>
</div>
<canvas id="c3d" height="520"></canvas>
<div class="cap">Both surfaces are the quadratic fit to all observables ("all" in section 5), drawn over the interior (at least 30 mm from the edge of the active area). The common tilt of the chamber in the reference frame is removed, so the mesh shows only its dish. The colour scale of the cathode is the local gap in mm (blue short, red long). Drift lines start on the cathode on a 7 × 7 grid.</div>
</div>

<h2>2. What the geometry does to the drift field</h2>
<p>The two surfaces change the drift field in two ways. First, the strength of the field is the drift voltage divided by the local gap. Where the gap is shorter, the field is stronger, the electrons drift faster, and they arrive sooner. Second, the direction of the field follows the slopes of the two surfaces. A tilted drift line moves the arriving charge sideways in proportion to its drift time. The reconstruction reads this sideways motion as part of the track angle. The maps below are computed from the fitted surfaces. Hover over a map to read its value.</p>
<div class="grid2" id="fieldmaps"></div>

<h2>3. The measurements</h2>
<p>Each map below is measured directly, in 4 × 4 cells of the interior (cells of about 85 × 75 mm), with the two runs combined. Each map is shown as a change relative to its own chamber average, so that offsets between methods do not enter. Empty cells are outside the reach of that method: inclined reference tracks reach only the centre of the chamber in the direction they are inclined.</p>
<h3>Three measurements of the gap</h3>
<p>The geometric gap uses no timing. An inclined track crosses the gap over a length L along the strips, so the gap is L / tan θ. We measure it with the X strips and, independently, with the Y strips. The third measurement multiplies the angle-scale velocity (from the reference tracks) by the duration of the drift column in each cell. The field dependence cancels in this product, so it measures the gap too.</p>
<div class="grid3" id="gapmaps"></div>
<h3>The field and the drift direction</h3>
<p>The drift time alone and the velocity alone each depend on the gap through the field. Converted to a gap change, they show the same pattern. The mesh height comes from the positions of the tracks (the bow of section 10 of the paper). The rate offset w₀ is the sideways drift speed: it is the drift tilt times the drift velocity.</p>
<div class="grid3" id="fmaps"></div>

<h3>Profiles</h3>
<p>These profiles show the gap along x (averaged over y) and along y (averaged over x), with the model. Along y the gap is longest in the middle, by about 1.2 mm relative to a point 150 mm out. Along x it increases steadily.</p>
<div class="grid2"><div class="card"><div class="mt">gap along x</div><svg id="px" viewBox="0 0 460 300"></svg></div><div class="card"><div class="mt">gap along y</div><svg id="py" viewBox="0 0 460 300"></svg></div></div>
<div class="cap" id="plegend"></div>

<h2>4. Why it is the gap and not the gas</h2>
<p>A change of the gap and a change of the gas both change the drift time, but they change the drift velocity in different ways. At a fixed drift voltage a shorter gap means a stronger field, so the electrons drift faster <i>and</i> have less distance to go. A wetter gas only slows the drift. Across the cells, the logarithm of the velocity changes by −0.50 ± 0.02 times the logarithm of the drift time. A pure change of the gap predicts −0.41 (with the Magboltz slope β = 0.68), and a change of the gas predicts −1. The same pattern appears in two runs taken hours apart, which also argues against a local change of the gas.</p>

<h2>5. The bow and the gap are the same two surfaces</h2>
<p>The test of consistency is to fit the model to subsets of the observables and compare the surfaces they give. The table lists the depth of each surface at 150 mm from the centre (positive means the centre is lower). The drift direction measures the sum of the two slopes, mesh plus cathode, and the gap measures their difference. So together they determine the mesh without any position information ("no positions"). That fit finds the same dish as the position route.</p>
<div class="tw"><table id="armtab"></table></div>
<ul>
<li><b>Along y the cathode is nearly flat</b> in every subset (−0.2 to 0.4 mm at 150 mm, against 1.4–1.6 mm for the mesh).</li>
<li><b>Along x the cathode follows the board.</b> The one subset that disagrees is the one without timing, at 2.5σ. The geometric gap with the X strips reaches only x ≈ 100–320 mm, so it constrains this direction weakly.</li>
<li><b>The reference telescope is not dished.</b> A separate dish of the telescope, added to the position route, fits to −0.25 ± 0.16 and −0.05 ± 0.20 mm.</li>
<li><b>What the quadratic model does not describe</b> sets the error added to each class of measurement: <span id="floors"></span>.</li>
</ul>

<h2>6. Better tracking</h2>
<p>The reconstruction already corrects the drift tilt with a quadratic surface (the bow block). The measured rate offset has more structure than a quadratic, about 0.18 µm/ns, which is about 0.3° in angle. To test whether this can be corrected, we make a map of the rate offset and the velocity on one run and apply it to the other run. The map is indexed by the reconstructed position of the track, and only two constants are measured again on the test run. Since the map and the test use different runs, the gain cannot come from fitting the noise of the test sample.</p>
<div class="grid2"><div class="card"><div class="mt">angle resolution (s68 of θ − θ_ref)</div><svg id="ptrk" viewBox="0 0 460 300"></svg><div class="cap">Filled: Run 1, open: Run 2. Solid: X strips, dashed: Y strips. Left: pass 1. Each map is made on the other run.</div></div>
<div class="card tw"><table id="trktab"></table><div class="cap">42 mm cells improve the angles by 0.15–0.17°, which is 16–21 %, on both views and in both runs. Most of the gain comes from the offset. With 34 mm cells the map made on the smaller Run 1 becomes noisy.</div></div></div>
<p>The structure the map corrects is two-dimensional and smooth. It is not a timing offset between front-end chips. A chip reads 64 strips (50 mm), so a chip offset would show steps along the view's own coordinate, and there are none. A map along that coordinate alone recovers only a fifth to a third of the gain. We therefore attribute the structure to the shape of the drift field on scales of about 50 mm, finer than a quadratic surface can follow.</p>

<h2>7. What this does not show</h2>
<ul>
<li>The size of the dome along y is known to about 30 %. The drift-time product shows about 0.7 of the amplitude of the geometric gap.</li>
<li>The fitted β is 1.0 ± 0.2, against 0.68 from Magboltz and 0.87 from the drift-time scan. The surfaces do not depend on this value.</li>
<li>The drift tilt along x rises faster toward the edges than a quadratic dish allows. The board is not a pure paraboloid, and the r<sub>f</sub> along x is window-dependent.</li>
<li>Only one mounting of chamber A at the operating point is usable. The mean gap, 1.8 mm short of the 30 mm design, is not explained.</li>
<li>Nothing here has been tested at n_TOF. There the boards stand vertical, so the part of the bow caused by gravity is absent, and there are no reference tracks.</li>
<li>The map is not in the production reconstruction. Shipping it is an open decision.</li>
</ul>
<p class="cap">Code and numbers: <span class="k">nTof_x17/rerun/paper1_gap/</span> (README, report.html; joint.py, track_grid.py, make_site_note.py). Paper: <span class="k">papers/paper1_v2/sections/05_gas.tex</span>, <span class="k">A4b_surfaces.tex</span>.</p>
</main>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const tip = $('#tip');
function showTip(e, html){tip.innerHTML=html;tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+10)+'px';}
function hideTip(){tip.style.display='none';}
// theme
(function(){let t=null;try{t=localStorage.getItem('thm')}catch(e){} if(t)document.documentElement.dataset.theme=t;
$('#thm').onclick=()=>{const d=document.documentElement;const dark=d.dataset.theme?d.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
d.dataset.theme=dark?'light':'dark';try{localStorage.setItem('thm',d.dataset.theme)}catch(e){} redrawAll();};})();
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ---------- the model ----------
const P = D.arms.all.params, SURF=['lu','lw','kuu','kww','kuw'];
const coef = pre => SURF.map(s=>P[pre+'_'+s]);
const q = (c,U,W)=>c[0]*U+c[1]*W+c[2]*U*U/2+c[3]*W*W/2+c[4]*U*W;
const dq = (c,U,W)=>[c[0]+c[2]*U+c[4]*W, c[1]+c[3]*W+c[4]*U];
const QH=coef('h'), QC=coef('c'), G0=P.g0, BETA_MB=0.68, VDRIFT=1000;
const XI=[30,368.6], YI=[48,349.9];
function model(x,y){const U=(x-200)/100,W=(y-200)/100;
  const h=q(QH,U,W), c=G0+q(QC,U,W), g=c-h; const [hu,hw]=dq(QH,U,W),[cu,cw]=dq(QC,U,W);
  return {h, c, g, hu:hu/100, hw:hw/100, cu:cu/100, cw:cw/100};}
// mean gap over the interior
let GM=0,nG=0; for(let x=XI[0];x<=XI[1];x+=5)for(let y=YI[0];y<=YI[1];y+=5){GM+=model(x,y).g;nG++} GM/=nG;

// ---------- colour scales ----------
function lerp(a,b,t){return a+(b-a)*t}
function div(t){ // t in [-1,1]: blue - white - red
  t=Math.max(-1,Math.min(1,t)); const dark=document.documentElement.dataset.theme==='dark'||(!document.documentElement.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);
  const mid=dark?[44,52,64]:[247,247,247], lo=[33,102,172], hi=[178,24,43];
  const e=t<0?lo:hi, u=Math.abs(t); return `rgb(${Math.round(lerp(mid[0],e[0],u))},${Math.round(lerp(mid[1],e[1],u))},${Math.round(lerp(mid[2],e[2],u))})`;}
function seq(t){t=Math.max(0,Math.min(1,t)); const a=[237,248,251],b=[11,61,145]; return `rgb(${Math.round(lerp(a[0],b[0],t))},${Math.round(lerp(a[1],b[1],t))},${Math.round(lerp(a[2],b[2],t))})`}

// ---------- 3-D view ----------
const cv=$('#c3d'), ctx=cv.getContext('2d');
let yaw=-0.6, pitch=1.22, zoom=1, exag=10, show={mesh:true,cath:true,lines:true}, spin=false;
function fit3d(){const r=cv.getBoundingClientRect(); const dpr=window.devicePixelRatio||1; cv.width=r.width*dpr; cv.height=520*dpr; cv.style.height='520px'; ctx.setTransform(dpr,0,0,dpr,0,0);}
// mesh plane (the chamber's rigid tilt) is removed from both surfaces
function plane(x,y){const U=(x-200)/100,W=(y-200)/100; return QH[0]*U+QH[1]*W;}
let HREF=0; {let s=0,n=0; for(let x=XI[0];x<=XI[1];x+=10)for(let y=YI[0];y<=YI[1];y+=10){s+=model(x,y).h-plane(x,y);n++} HREF=s/n;}
function zMesh(x,y){return exag*(model(x,y).h-plane(x,y)-HREF);}
function zCath(x,y){const m=model(x,y); return GM+exag*(m.c-plane(x,y)-HREF-GM);}
function proj(x,y,z){ // centre, rotate (yaw about z, then pitch), scale
  let X=x-200, Y=y-200, Z=z-14; const cy=Math.cos(yaw), sy=Math.sin(yaw);
  let x1=X*cy-Y*sy, y1=X*sy+Y*cy; const cp=Math.cos(pitch), sp=Math.sin(pitch);
  let y2=y1*cp-Z*sp, z2=y1*sp+Z*cp; const W=cv.clientWidth, H=520, s=Math.min(W,H*1.3)/520*zoom;
  return [W/2+x1*s, H/2+y2*s*0.98, z2];}
function draw3d(){
  const W=cv.clientWidth, H=520; ctx.clearRect(0,0,W,H); ctx.fillStyle=css('--card'); ctx.fillRect(0,0,W,H);
  const prims=[]; const nx=18, ny=16;
  const xs=[...Array(nx+1)].map((_,i)=>XI[0]+(XI[1]-XI[0])*i/nx), ys=[...Array(ny+1)].map((_,j)=>YI[0]+(YI[1]-YI[0])*j/ny);
  for(const [on,zf,kind] of [[show.mesh,zMesh,'m'],[show.cath,zCath,'c']]){ if(!on)continue;
    for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){
      const pts=[[xs[i],ys[j]],[xs[i+1],ys[j]],[xs[i+1],ys[j+1]],[xs[i],ys[j+1]]].map(([x,y])=>proj(x,y,zf(x,y)));
      const xc=(xs[i]+xs[i+1])/2, yc=(ys[j]+ys[j+1])/2, m=model(xc,yc);
      const col=kind==='c'?div((m.g-GM)/1.3):seq(0.25+0.6*(-(m.h-plane(xc,yc)-HREF)+0.9)/1.8);
      prims.push({d:pts.reduce((a,p)=>a+p[2],0)/4,f:()=>{ctx.beginPath();pts.forEach((p,k)=>k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();
        ctx.globalAlpha=kind==='c'?0.78:0.92; ctx.fillStyle=col; ctx.fill(); ctx.globalAlpha=0.35; ctx.strokeStyle=css('--mut'); ctx.lineWidth=0.4; ctx.stroke(); ctx.globalAlpha=1;}});
    }}
  if(show.lines){ const n=7;
    for(let a=0;a<n;a++)for(let b=0;b<n;b++){
      let x=XI[0]+15+(XI[1]-XI[0]-30)*a/(n-1), y=YI[0]+15+(YI[1]-YI[0]-30)*b/(n-1);
      const x0=x,y0=y; const seg=[]; const steps=24;
      // from the cathode (zeta = 1) to the mesh (zeta = 0): dx/dz = -(h' + zeta g'), z measured up, g = c - h
      let px=x,py=y; const path=[];
      for(let k=steps;k>=0;k--){const zeta=k/steps; const m=model(px,py);
        const zm=zMesh(x0+exag*(px-x0),y0+exag*(py-y0)), zc=zCath(x0+exag*(px-x0),y0+exag*(py-y0));
        path.push(proj(x0+exag*(px-x0), y0+exag*(py-y0), zm+zeta*(zc-zm)));
        const gx=m.cu-m.hu, gy=m.cw-m.hw, dz=m.g/steps; // stepping down by dz
        px+= (m.hu+zeta*gx)*dz; py+=(m.hw+zeta*gy)*dz;}
      for(let k=0;k<path.length-1;k++){const p1=path[k],p2=path[k+1];
        prims.push({d:(p1[2]+p2[2])/2,f:()=>{ctx.beginPath();ctx.moveTo(p1[0],p1[1]);ctx.lineTo(p2[0],p2[1]);ctx.strokeStyle=css('--acc2');ctx.lineWidth=1.3;ctx.stroke();}});}
    }}
  prims.sort((a,b)=>a.d-b.d); prims.forEach(p=>p.f());
  // axes labels
  ctx.fillStyle=css('--mut'); ctx.font='12px system-ui';
  const lab=(x,y,z,t)=>{const p=proj(x,y,z);ctx.fillText(t,p[0],p[1]);};
  lab(XI[1]+25,YI[0],0,'x'); lab(XI[0],YI[1]+25,0,'y'); lab(XI[0]-10,YI[0]-10,0,'(30, 48) mm');
  ctx.fillText('mesh: blue (darker = lower)   cathode: colour = local gap', 10, H-10);
  // gap colour bar
  const bx=W-170, by=14; for(let i=0;i<120;i++){ctx.fillStyle=div((i/119)*2-1);ctx.fillRect(bx+i,by,1,10);}
  ctx.fillStyle=css('--fg'); ctx.fillText((GM-1.3).toFixed(1)+' mm',bx-4,by+24); ctx.fillText((GM+1.3).toFixed(1),bx+100,by+24); ctx.fillText('gap',bx+48,by+24);
}
let drag=null;
cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,yaw,pitch];cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{if(!drag)return;yaw=drag[2]+(e.clientX-drag[0])*0.008;pitch=Math.max(0.05,Math.min(1.5,drag[3]+(e.clientY-drag[1])*0.006));draw3d();});
cv.addEventListener('pointerup',()=>drag=null);
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(0.5,Math.min(3,zoom*(e.deltaY<0?1.08:0.93)));draw3d();},{passive:false});
$('#exag').oninput=e=>{exag=+e.target.value;$('#exv').textContent=exag+'×';draw3d();};
for(const [id,k] of [['#bMesh','mesh'],['#bCath','cath'],['#bLines','lines']]) $(id).onclick=e=>{show[k]=!show[k];e.target.classList.toggle('on',show[k]);draw3d();};
$('#bSpin').onclick=e=>{spin=!spin;e.target.classList.toggle('on',spin);if(spin)tick();};
$('#bReset').onclick=()=>{yaw=-0.6;pitch=1.22;zoom=1;draw3d();};
function tick(){if(!spin)return;yaw+=0.006;draw3d();requestAnimationFrame(tick);}

// ---------- model field maps (canvas heatmaps with hover) ----------
const FIELD=[
 {t:'local drift gap', u:'mm', f:m=>m.g, c:v=>div((v-GM)/1.3), fmt:v=>v.toFixed(2), cap:'The fitted gap, cathode minus mesh. Shortest at the low-x side and at both y ends.'},
 {t:'drift field', u:'V/cm', f:m=>VDRIFT/(m.g/10), c:v=>div((v-VDRIFT/(GM/10))/13), fmt:v=>v.toFixed(0), cap:'The drift voltage (1000 V) divided by the local gap. A 1 mm shorter gap raises the field by about 3.6 %.'},
 {t:'drift velocity change', u:'%', f:m=>100*(Math.pow(m.g/GM,-BETA_MB)-1), c:v=>div(v/3), fmt:v=>(v>0?'+':'')+v.toFixed(2), cap:'Expected from the field, with the Magboltz slope β = 0.68 at 0.97 % water. Faster where the gap is short.'},
 {t:'drift time change', u:'%', f:m=>100*(Math.pow(m.g/GM,1+BETA_MB)-1), c:v=>div(v/8), fmt:v=>(v>0?'+':'')+v.toFixed(2), cap:'The gap is shorter and the drift faster, so the drift time changes by about 1.7 times the relative change of the gap.'},
 {t:'drift tilt (arrows) and sideways shift of cathode charge', u:'mm', f:m=>m.g*Math.hypot((m.hu+m.cu)/2,(m.hw+m.cw)/2), c:v=>seq(v/0.6), fmt:v=>v.toFixed(2), arrows:true, cap:'The drift line tilts by the mean of the mesh and cathode slopes (arrows, in the reference frame including the chamber tilt). The colour is the sideways shift of charge made at the cathode by the time it reaches the mesh.'},
 {t:'cathode minus a flat plane', u:'mm', f:m=>m.c-plane(m.x,m.y), c:v=>div(v/1.5), fmt:v=>v.toFixed(2), rel:true, cap:'The cathode surface with the chamber tilt removed: curved along x, nearly flat along y.'}];
function fieldMaps(){const host=$('#fieldmaps'); host.innerHTML=''; const cards=[];
  for(const F of FIELD){const card=document.createElement('div');card.className='card';
    card.innerHTML=`<div class="mt">${F.t} [${F.u}]</div><canvas height="250"></canvas><div class="cap">${F.cap}</div>`;host.appendChild(card);cards.push([F,card]);}
  for(const [F,card] of cards){
    const c=card.querySelector('canvas'),g=c.getContext('2d');const r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=r.width*dpr;c.height=250*dpr;c.style.height='250px';g.setTransform(dpr,0,0,dpr,0,0);
    const W=r.width,H=250,L=36,B=24,pw=W-L-8,ph=H-B-8; const sx=x=>L+(x-XI[0])/(XI[1]-XI[0])*pw, sy=y=>8+ph-(y-YI[0])/(YI[1]-YI[0])*ph;
    const N=60; let vals=[];
    for(let i=0;i<N;i++)for(let j=0;j<N;j++){const x=XI[0]+(i+.5)*(XI[1]-XI[0])/N,y=YI[0]+(j+.5)*(YI[1]-YI[0])/N;const m=model(x,y);m.x=x;m.y=y;vals.push([x,y,F.f(m)]);}
    let off=0; if(F.rel){off=vals.reduce((a,v)=>a+v[2],0)/vals.length;}
    for(const [x,y,v] of vals){g.fillStyle=F.c(v-off);g.fillRect(sx(x-(XI[1]-XI[0])/N/2),sy(y+(YI[1]-YI[0])/N/2),pw/N+1,ph/N+1);}
    if(F.arrows){g.strokeStyle=css('--fg');g.lineWidth=1.1;for(let i=0;i<7;i++)for(let j=0;j<6;j++){const x=XI[0]+(i+.5)*(XI[1]-XI[0])/7,y=YI[0]+(j+.5)*(YI[1]-YI[0])/6;const m=model(x,y);
      const tx=(m.hu+m.cu)/2,ty=(m.hw+m.cw)/2,s=1800;const x0=sx(x),y0=sy(y),x1=x0+tx*s,y1=y0-ty*s;g.beginPath();g.moveTo(x0,y0);g.lineTo(x1,y1);g.stroke();
      const a=Math.atan2(y1-y0,x1-x0);g.beginPath();g.moveTo(x1,y1);g.lineTo(x1-5*Math.cos(a-.4),y1-5*Math.sin(a-.4));g.lineTo(x1-5*Math.cos(a+.4),y1-5*Math.sin(a+.4));g.closePath();g.fillStyle=css('--fg');g.fill();}}
    g.fillStyle=css('--mut');g.font='11px system-ui';[100,200,300].forEach(x=>g.fillText(x,sx(x)-9,H-6));[100,200,300].forEach(y=>g.fillText(y,4,sy(y)+4));g.fillText('x [mm]',W-48,H-6);g.fillText('y',L-12,12);
    c.onmousemove=e=>{const b=c.getBoundingClientRect();const x=XI[0]+(e.clientX-b.left-L)/pw*(XI[1]-XI[0]),y=YI[0]+(8+ph-(e.clientY-b.top))/ph*(YI[1]-YI[0]);
      if(x<XI[0]||x>XI[1]||y<YI[0]||y>YI[1]){hideTip();return;}const m=model(x,y);m.x=x;m.y=y;showTip(e,`x ${x.toFixed(0)}, y ${y.toFixed(0)} mm<br><b>${F.fmt(F.f(m)-off)} ${F.u}</b>`);};
    c.onmouseleave=hideTip;}}

// ---------- measured cell maps (SVG) ----------
const MEAS1=[['G','geometric gap, X strips','mm',1.5],['GY','geometric gap, Y strips','mm',1.5],['VT','angle-scale velocity × drift time','mm',1.5]];
const MEAS2=[['T','drift time, read as a gap change','mm',1.5],['V','velocity, read as a gap change','mm',1.5],['hp','mesh height from positions (X, Y mean)','mm',1.2],['w0_x','rate offset w₀, X strips','µm/ns',0.8],['w0_y','rate offset w₀, Y strips','µm/ns',0.8]];
function cellMap(host,key,title,unit,scale){
  let cells=key==='hp'?D.maps.hp_x.map((c,i)=>({x:c.x,y:c.y,v:(c.v!=null&&D.maps.hp_y[i].v!=null)?(c.v+D.maps.hp_y[i].v)/2:null})):D.maps[key];
  const xsU=[...new Set(cells.map(c=>+c.x.toFixed(1)))].sort((a,b)=>a-b), ysU=[...new Set(cells.map(c=>+c.y.toFixed(1)))].sort((a,b)=>a-b);
  const W=240,H=210,L=30,B=22,pw=W-L-6,ph=H-B-6,cw=pw/xsU.length,ch=ph/ysU.length;
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(const c of cells){const i=xsU.indexOf(+c.x.toFixed(1)),j=ysU.indexOf(+c.y.toFixed(1));const X=L+i*cw,Y=6+ph-(j+1)*ch;
    const fill=c.v==null?'none':div(c.v/scale);const tt=c.v==null?'no coverage':`${(c.v>0?'+':'')+c.v.toFixed(2)}${c.e!=null?' ± '+c.e.toFixed(2):''} ${unit}`;
    s+=`<rect x="${X}" y="${Y}" width="${cw-1}" height="${ch-1}" fill="${fill}" stroke="${css('--line')}" data-t="x ${c.x.toFixed(0)}, y ${c.y.toFixed(0)} mm: ${tt}"/>`;
    if(c.v!=null)s+=`<text x="${X+cw/2}" y="${Y+ch/2+4}" text-anchor="middle" font-size="11" fill="${Math.abs(c.v/scale)>0.6?'#fff':css('--fg')}" pointer-events="none">${(c.v>0?'+':'')+c.v.toFixed(2)}</text>`;}
  s+=`<text x="${L}" y="${H-5}" font-size="10" fill="${css('--mut')}">x →</text><text x="2" y="14" font-size="10" fill="${css('--mut')}">y ↑</text></svg>`;
  const d=document.createElement('div');d.className='card';d.innerHTML=`<div class="mt">${title} [${unit}]</div>`+s;host.appendChild(d);
  d.querySelectorAll('rect').forEach(r=>{r.onmousemove=e=>showTip(e,r.dataset.t);r.onmouseleave=hideTip;});}
function measMaps(){const a=$('#gapmaps'),b=$('#fmaps');a.innerHTML='';b.innerHTML='';MEAS1.forEach(m=>cellMap(a,...m));MEAS2.forEach(m=>cellMap(b,...m));}

// ---------- SVG plots ----------
function plot(svg,{xr,yr,xl,yl,series,lines,xticks,yticks,xlabels}){
  const W=460,H=300,L=52,R=12,T=12,B=40,pw=W-L-R,ph=H-T-B; const sx=x=>L+(x-xr[0])/(xr[1]-xr[0])*pw, sy=y=>T+ph-(y-yr[0])/(yr[1]-yr[0])*ph;
  let s=`<rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="none" stroke="${css('--line')}"/>`;
  (yticks||[]).forEach(v=>{s+=`<line x1="${L}" x2="${L+pw}" y1="${sy(v)}" y2="${sy(v)}" stroke="${css('--line')}" stroke-width="0.6"/><text x="${L-6}" y="${sy(v)+4}" text-anchor="end" font-size="11" fill="${css('--mut')}">${v}</text>`});
  (xticks||[]).forEach((v,i)=>{s+=`<text x="${sx(v)}" y="${T+ph+16}" text-anchor="middle" font-size="11" fill="${css('--mut')}">${xlabels?xlabels[i]:v}</text>`});
  s+=`<text x="${L+pw/2}" y="${H-6}" text-anchor="middle" font-size="12" fill="${css('--fg')}">${xl}</text><text x="14" y="${T+ph/2}" transform="rotate(-90 14 ${T+ph/2})" text-anchor="middle" font-size="12" fill="${css('--fg')}">${yl}</text>`;
  for(const l of lines||[]){s+=`<polyline fill="none" stroke="${l.c}" stroke-width="${l.w||1.6}" ${l.dash?'stroke-dasharray="5 4"':''} points="${l.pts.map(p=>sx(p[0])+','+sy(p[1])).join(' ')}"/>`;}
  for(const S of series){for(const p of S.pts){if(p[1]==null)continue;const X=sx(p[0]+(S.dx||0)),Y=sy(p[1]);
    if(p[2])s+=`<line x1="${X}" x2="${X}" y1="${sy(p[1]-p[2])}" y2="${sy(p[1]+p[2])}" stroke="${S.c}"/>`;
    const sh=S.m==='s'?`<rect x="${X-4}" y="${Y-4}" width="8" height="8"`:`<circle cx="${X}" cy="${Y}" r="4.2"`;
    s+=`${sh} fill="${S.open?css('--card'):S.c}" stroke="${S.c}" stroke-width="1.5" data-t="${S.name}: ${p[1].toFixed(3)}${p[2]?' ± '+p[2].toFixed(2):''}"/>`;}}
  svg.innerHTML=s; svg.querySelectorAll('[data-t]').forEach(r=>{r.onmousemove=e=>showTip(e,r.dataset.t);r.onmouseleave=hideTip;});}
function profiles(){
  const acc=css('--acc'),acc2=css('--acc2'),fg=css('--fg');
  for(const [id,g,ax] of [['#px','x5','x'],['#py','y5','y']]){
    const Pr=D.prof[g],key=ax==='x'?'x':'y';
    const S=[{name:'geometric, X strips',c:acc,m:'o',dx:-5,pts:Pr.G.map(p=>[p[key],p.v==null?null:28.19+p.v,p.e])},
             {name:'geometric, Y strips',c:acc,m:'o',open:true,dx:0,pts:Pr.GY.map(p=>[p[key],p.v==null?null:28.19+p.v,p.e])},
             {name:'velocity × drift time',c:acc2,m:'s',dx:5,pts:Pr.VT.map(p=>[p[key],p.v==null?null:28.19+p.v,p.e])}];
    const pts=[];for(let t=(ax==='x'?XI[0]:YI[0]);t<=(ax==='x'?XI[1]:YI[1]);t+=4){let s=0,n=0;for(let u=(ax==='x'?YI[0]:XI[0]);u<=(ax==='x'?YI[1]:XI[1]);u+=6){s+=ax==='x'?model(t,u).g:model(u,t).g;n++}pts.push([t,28.19+s/n-GM]);}
    plot($(id),{xr:[30,370],yr:[26.2,29.6],xl:ax+' at mid-gap [mm]',yl:'drift gap [mm]',series:S,lines:[{pts,c:fg,w:1.4}],xticks:[50,100,150,200,250,300,350],yticks:[26.5,27,27.5,28,28.5,29,29.5]});}
  $('#plegend').innerHTML='Filled circles: geometric gap, X strips. Open circles: geometric gap, Y strips (relative, placed on the chamber value 28.19 mm). Squares: angle-scale velocity × drift time. Line: the two-surface model, averaged over the other coordinate. Hover over a point for its value.';}
function tracking(){
  const acc=css('--acc'),acc2=css('--acc2'); const size={4:85,6:57,8:42,10:34}; const S=[],lines=[];
  for(const [p,dash] of [['x',false],['y',true]])for(const [run,open] of [['run1',false],['run2',true]]){const T=D.trk[p+'_'+run];
    const pts=[[130,T.pass1]].concat([4,6,8,10].map(n=>[size[n],T.grid[n]]));
    lines.push({pts:pts.slice(1),c:acc,dash,w:1.3});S.push({name:`${p.toUpperCase()} ${run.replace('run','Run ')}`,c:acc,m:open?'s':'o',open,pts:pts.slice(1)});
    S.push({name:`${p.toUpperCase()} ${run.replace('run','Run ')}, pass 1`,c:acc2,m:open?'s':'o',open,pts:[[130,T.pass1]]});}
  plot($('#ptrk'),{xr:[140,26],yr:[0.58,0.96],xl:'map cell size [mm]',yl:'σθ (s68) [deg]',series:S,lines,xticks:[130,85,57,42,34],xlabels:['pass 1','85','57','42','34'],yticks:[0.6,0.7,0.8,0.9]});
  const rows=[['pass 1 (quadratic bow block)','pass1'],['two-surface model','model'],['map, 85 mm cells',4],['map, 57 mm cells',6],['map, 42 mm cells',8],['map, 42 mm, offset only','w0only'],['map, 34 mm cells',10]];
  let h='<tr><th>angle conversion</th><th>X Run 1</th><th>X Run 2</th><th>Y Run 1</th><th>Y Run 2</th></tr>';
  for(const [l,k] of rows){h+=`<tr><td>${l}</td>`+['x_run1','x_run2','y_run1','y_run2'].map(c=>{const T=D.trk[c];const v=typeof k==='number'?T.grid[k]:T[k];return `<td>${v.toFixed(3)}°</td>`}).join('')+'</tr>';}
  $('#trktab').innerHTML=h;}
function armTable(){const lab={all:'all observables',tilt:'positions + drift direction',gap:'positions + gap',geo:'no timing',no_hp:'no positions',m3dish:'all + free telescope dish'};
  let h='<tr><th>observables used</th><th>mesh, x</th><th>mesh, y</th><th>cathode, x</th><th>cathode, y</th><th>gap tilt along x (mm/100 mm)</th></tr>';
  for(const [a,l] of Object.entries(lab)){const F=D.arms[a],p=F.params,e=F.errors;const d=k=>(1.125*p[k]).toFixed(2)+(e[k]!=null?' ± '+(1.125*e[k]).toFixed(2):'');
    h+=`<tr><td>${l}</td><td>${d('h_kuu')}</td><td>${d('h_kww')}</td><td>${d('c_kuu')}</td><td>${d('c_kww')}</td><td>${(p.c_lu-p.h_lu).toFixed(2)}</td></tr>`;}
  $('#armtab').innerHTML=h; const f=D.floors;
  $('#floors').textContent=`mesh height ${f.hp.toFixed(2)} mm, rate offset ${f.w0.toFixed(2)} µm/ns, geometric gap ${f.G.toFixed(2)} mm, drift time ${(100*f.T).toFixed(1)} %, velocity ${(100*f.v).toFixed(1)} %`;}
function redrawAll(){fit3d();draw3d();fieldMaps();measMaps();profiles();tracking();armTable();}
window.addEventListener('resize',()=>{clearTimeout(window._rt);window._rt=setTimeout(redrawAll,150);});
redrawAll();
</script></body></html>
"""


def main():
    d = data(sys.argv[1])
    html = TEMPLATE.replace('__DATA__', json.dumps(d, separators=(',', ':')))
    open(sys.argv[2], 'w').write(html)
    print('wrote', sys.argv[2], len(html) // 1024, 'kB')


if __name__ == '__main__':
    main()
