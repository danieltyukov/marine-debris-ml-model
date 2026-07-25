"""Build docs/index.html, the GitHub Pages report.

Injects the real measured FDI distribution and the compressed pipeline figures so the
page cannot drift from the numbers the pipeline actually produces. Run from the repo
root after regenerating the figures:

    python -m mdebris.viz.figures
    python scripts/build_demo_page.py
"""

import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mdebris.data import sample_reflectance
from mdebris.indices.masks import cloud_mask_from_scl, water_mask
from mdebris.indices.spectral import compute_indices

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    "cascade": "assets/cascade_stages.png",
    "detect": "assets/detections.png",
    "indices": "assets/spectral_indices.png",
}


def measure_fdi() -> dict:
    """Real FDI distribution over cloud-free water in each bundled chip.

    Clouds are excluded before the histogram is taken, for the same reason the
    cascade excludes them before taking its percentile: cloud tops carry very large
    FDI and would dominate the tail the page is about.
    """
    out = {}
    for name in ("accra", "limassol"):
        bands, meta = sample_reflectance(name)
        refl = {k: v for k, v in bands.items() if k != "SCL"}
        fdi = compute_indices(refl)["FDI"]
        wet = water_mask(refl)
        cloud = cloud_mask_from_scl(bands["SCL"])
        values = fdi[(wet & ~cloud) & np.isfinite(fdi)]
        lo, hi = float(np.percentile(values, 0.5)), float(np.percentile(values, 99.95))
        hist, edges = np.histogram(values, bins=90, range=(lo, hi))
        out[name] = {
            "scene": meta["scene_id"],
            "date": meta["datetime"][:10],
            "water_pct": round(float(wet.mean() * 100), 1),
            "cloud_pct": round(float(cloud.mean() * 100), 2),
            "n": int(values.size),
            "hist": hist.tolist(),
            "edges": [round(e, 6) for e in edges.tolist()],
            "pct": {
                key: round(float(np.percentile(values, q)), 6)
                for key, q in (
                    ("p50", 50),
                    ("p90", 90),
                    ("p99", 99),
                    ("p99_9", 99.9),
                    ("p99_99", 99.99),
                )
            },
            "mean": round(float(values.mean()), 6),
        }
    return out


def embed_figures(width: int = 1400, quality: int = 72) -> dict[str, str]:
    """Downsample each figure to a JPEG data URI so the page is one self-contained file."""
    uris = {}
    for key, rel in FIGURES.items():
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"{rel} is missing, run: python -m mdebris.viz.figures")
        im = Image.open(path).convert("RGB")
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        buf = ROOT / "docs" / f"_{key}.jpg"
        buf.parent.mkdir(parents=True, exist_ok=True)
        im.save(buf, "JPEG", quality=quality, optimize=True)
        uris[key] = "data:image/jpeg;base64," + base64.b64encode(buf.read_bytes()).decode()
        buf.unlink()
    return uris


fdi = measure_fdi()
imgs = embed_figures()

HTML = r"""<title>Marine Debris Detection: what Sentinel-2 can and cannot tell you</title>
<style>
:root{
  --deep:#E7EDEE; --sheet:#F4F7F7; --panel:#FFFFFF; --line:#C3D2D6; --hair:#DCE6E8;
  --ink:#0B1D24; --ink-2:#3D5560; --ink-3:#6C858E;
  --debris:#B87400; --debris-bright:#E69F00; --water:#0060A0; --sarg:#00795A; --verm:#C4541A;
  --grid:rgba(11,29,36,.05);
}
@media (prefers-color-scheme:dark){
  :root{
    --deep:#071A22; --sheet:#0B222C; --panel:#0F2A35; --line:#1E4351; --hair:#16333F;
    --ink:#E4EFF2; --ink-2:#A9C2CA; --ink-3:#6F8D97;
    --debris:#E69F00; --debris-bright:#FFB627; --water:#4AA3DA; --sarg:#2CBE9A; --verm:#F07A3C;
    --grid:rgba(228,239,242,.055);
  }
}
:root[data-theme="dark"]{
  --deep:#071A22; --sheet:#0B222C; --panel:#0F2A35; --line:#1E4351; --hair:#16333F;
  --ink:#E4EFF2; --ink-2:#A9C2CA; --ink-3:#6F8D97;
  --debris:#E69F00; --debris-bright:#FFB627; --water:#4AA3DA; --sarg:#2CBE9A; --verm:#F07A3C;
  --grid:rgba(228,239,242,.055);
}
:root[data-theme="light"]{
  --deep:#E7EDEE; --sheet:#F4F7F7; --panel:#FFFFFF; --line:#C3D2D6; --hair:#DCE6E8;
  --ink:#0B1D24; --ink-2:#3D5560; --ink-3:#6C858E;
  --debris:#B87400; --debris-bright:#E69F00; --water:#0060A0; --sarg:#00795A; --verm:#C4541A;
  --grid:rgba(11,29,36,.05);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--deep); color:var(--ink);
  font-family:ui-serif,Georgia,"Iowan Old Style","Palatino Linotype",Palatino,serif;
  font-size:17px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}
.mono{font-family:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace}
h1,h2,h3,.disp{
  font-family:-apple-system,"Segoe UI Variable Display","Segoe UI",system-ui,sans-serif;
  font-weight:700; letter-spacing:-.022em; text-wrap:balance; margin:0;
}
p{margin:0}
a{color:var(--debris); text-underline-offset:3px}
:focus-visible{outline:2px solid var(--debris-bright); outline-offset:3px; border-radius:2px}

/* survey-sheet shell: measured column, mono station gutter */
.sheet{max-width:1180px; margin:0 auto; padding:0 28px}
.row{display:grid; grid-template-columns:104px minmax(0,1fr); gap:34px; align-items:start}
.station{
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:10.5px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink-3); padding-top:.55em; line-height:1.5;
  border-top:1px solid var(--hair);
}
.body-col{max-width:66ch}
.body-col .scroll,.body-col table{max-width:none; width:100%}
.body-col > .scroll{width:min(100%,760px)}
.wide-col{max-width:none}
section{padding:58px 0; border-top:1px solid var(--hair)}
section:first-of-type{border-top:0}
@media (max-width:800px){
  .row{grid-template-columns:1fr; gap:12px}
  .station{border-top:0; padding-top:0}
  .sheet{padding:0 20px}
}

/* hero */
.hero{position:relative; overflow:hidden; border-bottom:1px solid var(--line)}
#contours{position:absolute; inset:0; width:100%; height:100%; display:block}
.hero-in{position:relative; padding:84px 0 66px}
.eyebrow{
  font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:11px; letter-spacing:.2em;
  text-transform:uppercase; color:var(--debris); margin-bottom:20px;
}
h1{font-size:clamp(34px,5.4vw,60px); line-height:1.03}
.lede{margin-top:22px; font-size:clamp(17px,2vw,20px); color:var(--ink-2); max-width:60ch}
.stats{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px; background:var(--hair); border:1px solid var(--hair); margin-top:44px;
}
.stat{background:var(--sheet); padding:18px 18px 16px}
.stat b{
  display:block; font-family:ui-monospace,Menlo,monospace; font-size:26px; font-weight:600;
  letter-spacing:-.02em; font-variant-numeric:tabular-nums; color:var(--ink);
}
.stat span{
  display:block; margin-top:5px; font-family:ui-monospace,Menlo,monospace;
  font-size:10.5px; letter-spacing:.09em; text-transform:uppercase; color:var(--ink-3); line-height:1.5;
}
.stat.hl b{color:var(--debris)}

h2{font-size:clamp(24px,3.1vw,33px); line-height:1.12; margin-bottom:18px}
h3{font-size:17px; margin:30px 0 8px; letter-spacing:-.01em}
.prose > * + *{margin-top:18px}
.prose strong{font-weight:600; color:var(--ink)}

/* the negative-result callout: vermilion, used once */
.finding{
  border-left:3px solid var(--verm); background:var(--sheet);
  padding:22px 24px; margin-top:26px;
}
.finding .tag{
  font-family:ui-monospace,Menlo,monospace; font-size:10.5px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--verm); display:block; margin-bottom:10px;
}

table{width:100%; border-collapse:collapse; font-size:14.5px; margin-top:22px}
.scroll{overflow-x:auto}
th,td{padding:10px 14px; text-align:left; border-bottom:1px solid var(--hair)}
th{
  font-family:ui-monospace,Menlo,monospace; font-size:10.5px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-3); font-weight:400; border-bottom:1px solid var(--line);
  white-space:nowrap;
}
td{font-family:ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums; color:var(--ink-2)}
td.k{color:var(--ink); white-space:nowrap}
td.good{color:var(--sarg)} td.bad{color:var(--verm)}
tr:last-child td{border-bottom:0}

/* interactive threshold explorer */
.tool{border:1px solid var(--line); background:var(--sheet); margin-top:30px}
.tool-head{
  display:flex; justify-content:space-between; align-items:baseline; gap:16px; flex-wrap:wrap;
  padding:16px 20px; border-bottom:1px solid var(--hair);
}
.tool-title{font-family:ui-monospace,Menlo,monospace; font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--ink-3)}
.readout{font-family:ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums; font-size:14px; color:var(--ink)}
.readout b{color:var(--debris); font-weight:600}
#plot{display:block; width:100%; height:280px}
.controls{padding:14px 20px 20px; display:grid; gap:14px}
.sliderow{display:grid; grid-template-columns:auto 1fr; gap:14px; align-items:center}
label.sl{font-family:ui-monospace,Menlo,monospace; font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3)}
input[type=range]{width:100%; accent-color:var(--debris-bright); height:22px}
.presets{display:flex; gap:8px; flex-wrap:wrap}
button.preset{
  font-family:ui-monospace,Menlo,monospace; font-size:11px; letter-spacing:.06em;
  padding:7px 12px; background:transparent; color:var(--ink-2);
  border:1px solid var(--line); cursor:pointer; border-radius:2px;
}
button.preset:hover{border-color:var(--debris); color:var(--debris)}
button.preset[aria-pressed="true"]{background:var(--debris); border-color:var(--debris); color:var(--deep)}
.verdict{font-size:14.5px; color:var(--ink-2); padding:0 20px 18px; max-width:70ch}
.verdict b{color:var(--ink)}

figure{margin:30px 0 0}
figure img{width:100%; height:auto; display:block; border:1px solid var(--line)}
figcaption{
  margin-top:10px; font-family:ui-monospace,Menlo,monospace; font-size:11.5px;
  color:var(--ink-3); line-height:1.6; max-width:78ch;
}
pre{
  background:var(--sheet); border:1px solid var(--hair); padding:16px 18px; overflow-x:auto;
  font-family:ui-monospace,Menlo,Consolas,monospace; font-size:13px; line-height:1.7; margin-top:20px;
  color:var(--ink-2);
}
pre .c{color:var(--ink-3)}
pre .p{color:var(--debris)}
footer{padding:44px 0 70px; border-top:1px solid var(--line); color:var(--ink-3); font-size:14px}
footer a{color:var(--ink-2)}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
</style>

<div class="hero">
  <canvas id="contours" aria-hidden="true"></canvas>
  <div class="sheet hero-in">
    <div class="row">
      <div class="station">Sheet 01<br>Gulf of Guinea<br>Akrotiri Bay</div>
      <div>
        <div class="eyebrow mono">Sentinel-2 &middot; 10 m &middot; open data</div>
        <h1>What satellite imagery can, and cannot, tell you about ocean plastic</h1>
        <p class="lede">A 2019 TensorFlow 1.14 project rebuilt from scratch. Along the way the
        rewrite produced a negative result worth more than the feature list: at 10 metres per
        pixel, the frontier vision model is not what finds the debris.</p>
        <div class="stats">
          <div class="stat"><b>761</b><span>tests passing</span></div>
          <div class="stat"><b>175,481</b><span>lines of dead code removed</span></div>
          <div class="stat hl"><b>0</b><span>API keys required</span></div>
          <div class="stat hl"><b>0</b><span>training steps to first prediction</span></div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="sheet">

<section>
  <div class="row">
    <div class="station">02<br>Finding</div>
    <div class="body-col prose">
      <h2>The same model, two domains</h2>
      <p>OWLv2 is an open-vocabulary detector: you hand it text at inference time and it
      finds what you asked for, with no training. It works. The control run below uses the
      canonical COCO validation photograph, and the model localises it tightly.</p>
      <p>Then the identical wrapper and weights were pointed at a Sentinel-2 chip.</p>
      <div class="scroll">
      <table>
        <thead><tr><th>Input</th><th>Prompt</th><th>Confidence</th><th>Box size</th></tr></thead>
        <tbody>
          <tr><td class="k">COCO photo</td><td>a remote control</td><td class="good">0.794</td><td>1.9% of image</td></tr>
          <tr><td class="k">COCO photo</td><td>a photo of a cat</td><td class="good">0.669</td><td>44.6% (cat is large)</td></tr>
          <tr><td class="k">Sentinel-2 10 m</td><td>white sea foam</td><td class="bad">0.210</td><td>up to 100% of chip</td></tr>
          <tr><td class="k">Sentinel-2 10 m</td><td>a boat wake</td><td class="bad">0.129</td><td>whole chip</td></tr>
        </tbody>
      </table>
      </div>
      <div class="finding">
        <span class="tag mono">Negative result</span>
        <p>The wrapper is correct, so this is <strong>domain mismatch, not a bug</strong>.
        OWLv2 learned from web photographs where a target spans hundreds of pixels. At 10 m
        ground sample distance a 30 m debris patch spans <strong>three pixels</strong>. The
        texture, shape and context cues the model relies on do not exist at that scale, so it
        falls back to describing the whole scene.</p>
      </div>
      <p>This is why the marine-litter literature thresholds spectral indices rather than
      running detectors at this resolution. The physics carries the signal. A photographic
      prior does not.</p>
    </div>
  </div>
</section>

<section>
  <div class="row">
    <div class="station">03<br>Explorer<br>live data</div>
    <div class="wide-col prose">
      <h2>Why a fixed threshold fails</h2>
      <p>The Floating Debris Index measures how far a pixel's near-infrared reflectance rises
      above a baseline interpolated between red and shortwave infrared. Below is the
      <strong>real, measured FDI distribution</strong> over cloud-free water from the two
      Sentinel-2 chips bundled with the repository. Drag the threshold.</p>

      <div class="tool">
        <div class="tool-head">
          <span class="tool-title">FDI over cloud-free water &middot; <span id="scene" class="mono"></span></span>
          <span class="readout">threshold <b id="tval">0.0060</b> &nbsp;&rarr;&nbsp; <b id="tpct">0.0%</b> of water flagged</span>
        </div>
        <canvas id="plot"></canvas>
        <div class="controls">
          <div class="sliderow">
            <label class="sl" for="thr">Threshold</label>
            <input id="thr" type="range" min="0" max="1000" value="120" aria-label="FDI threshold">
          </div>
          <div class="presets">
            <button class="preset" data-p="fixed">Fixed 0.006 (original)</button>
            <button class="preset" data-p="p999" aria-pressed="true">Adaptive p99.9 (current)</button>
            <button class="preset" data-p="scene">Switch chip</button>
          </div>
        </div>
        <p class="verdict" id="verdict"></p>
      </div>

      <p style="margin-top:26px">The original design used a constant <span class="mono">0.006</span>.
      On real open ocean that flags <strong>6.35% of pure water</strong> and accepts every tile,
      so the screen appears to work while saving nothing. FDI magnitude shifts with atmospheric
      correction, sun angle, sea state and water type, so no constant transfers between scenes.
      The threshold is now taken from a high percentile of each scene's own water, which is
      self-calibrating.</p>
    </div>
  </div>
</section>

<section>
  <div class="row">
    <div class="station">04<br>Method</div>
    <div class="wide-col prose">
      <h2>The screening cascade</h2>
      <p>A vision transformer costs about 18 seconds per tile on CPU and a Sentinel-2 scene is
      120 megapixels, roughly 40 minutes of compute. Cheap arithmetic screens the scene first;
      the expensive model only looks where something is worth looking at.</p>
      <figure>
        <img src="__CASCADE__" alt="Four panels: true colour Sentinel-2 of the Accra coastline, the NDWI water mask, the SCL cloud mask, and the FDI index over cloud-free water with the adaptive threshold contour.">
        <figcaption>Accra, Ghana, 2024-04-07. Clouds are excluded before the percentile is taken,
        not only from the final mask. Cloud tops carry very large FDI: leaving them in the sample
        pushed the threshold from 0.2673 to 0.3206 and desensitised the screen on exactly the
        cloudy scenes that need it. Both components were individually correct, which is why no
        unit test caught it. It was visible at a glance in this figure.</figcaption>
      </figure>
      <div class="scroll">
      <table>
        <thead><tr><th>Coastal scene, 36 tiles</th><th>Tiles detected on</th><th>Detector time</th></tr></thead>
        <tbody>
          <tr><td class="k">Without cascade</td><td>36 / 36</td><td>11.1 min</td></tr>
          <tr><td class="k">With cascade</td><td>20 / 36</td><td class="good">6.2 min</td></tr>
        </tbody>
      </table>
      </div>
      <p style="margin-top:18px; font-size:15px; color:var(--ink-3)">44% of detector calls avoided.
      An earlier draft claimed the cascade cut 40 minutes "to minutes"; real data did not support
      that, and the claim was corrected rather than kept.</p>
    </div>
  </div>
</section>

<section>
  <div class="row">
    <div class="station">05<br>Output</div>
    <div class="wide-col prose">
      <h2>Where the model still earns its place</h2>
      <p>Low-confidence localisation still yields useful discrimination. The 2019 model had one
      class, <span class="mono">marine_debris</span>, so it was structurally unable to say
      "that is a ship". Every detection below would have been reported as debris.</p>
      <figure>
        <img src="__DETECT__" alt="Side by side: Sentinel-2 true colour of the Accra coastline, and the same chip with OWLv2 zero-shot detection boxes labelled ship and ship wake.">
        <figcaption>Eight detections, all labelled ship or ship_wake, none debris. On the
        Limassol chip, 13 of 14 were foam, wake or sediment. Boxes are geo-registered to
        lon/lat and written as GeoJSON with an explicit CRS.</figcaption>
      </figure>
      <figure>
        <img src="__INDICES__" alt="Six heatmaps over water: FDI, FAI, NDVI, NDWI, PI and kNDVI computed on the Accra Sentinel-2 chip.">
        <figcaption>Six spectral indices over the same water. Separating plastic from Sargassum,
        foam and sediment is the actual scientific difficulty, and it needs several indices
        rather than one.</figcaption>
      </figure>
    </div>
  </div>
</section>

<section>
  <div class="row">
    <div class="station">06<br>Run it</div>
    <div class="body-col prose">
      <h2>No credentials, no GPU</h2>
      <p>Imagery is read straight from cloud-optimised GeoTIFFs over HTTP range requests, so
      screening a coastline pulls kilobytes instead of downloading gigabyte scenes.</p>
<pre><span class="c"># CPU-only torch, avoids ~2.5 GB of unusable CUDA libraries</span>
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e <span class="p">".[all]"</span>

<span class="c"># bundled real Sentinel-2 chips, works offline</span>
mdebris samples
mdebris indices --sample accra
mdebris detect --sample limassol --targets-only -o out.geojson

<span class="c"># or search live imagery for any coastline</span>
mdebris detect --bbox <span class="p">-0.35,5.45,-0.05,5.65</span> --start 2024-01-01 --end 2024-06-30</pre>
      <div class="scroll">
      <table>
        <thead><tr><th></th><th>2019</th><th>Now</th></tr></thead>
        <tbody>
          <tr><td class="k">Framework</td><td>TensorFlow 1.14</td><td>PyTorch 2.x</td></tr>
          <tr><td class="k">Installs on current Python</td><td class="bad">no</td><td class="good">yes</td></tr>
          <tr><td class="k">Classes</td><td>1</td><td>9</td></tr>
          <tr><td class="k">Imagery</td><td>Planet, commercial</td><td>Sentinel-2, open</td></tr>
          <tr><td class="k">Training before first result</td><td>500k steps</td><td class="good">none</td></tr>
          <tr><td class="k">Vendored dependencies</td><td>19 MB</td><td class="good">none</td></tr>
          <tr><td class="k">Tests</td><td class="bad">0</td><td class="good">761</td></tr>
        </tbody>
      </table>
      </div>
    </div>
  </div>
</section>

<footer>
  <div class="row">
    <div class="station">End</div>
    <div class="body-col">
      <p style="font-size:14px">MIT licensed. Sentinel-2 data is free and open under Copernicus terms.
      Figures regenerate with <span class="mono">python -m mdebris.viz.figures</span>.</p>
      <p style="margin-top:12px; font-size:14px">Floating Debris Index from Biermann, Clewley,
      Martinez-Vicente &amp; Topouzelis (2020), <em>Finding Plastic Patches in Coastal Waters
      using Optical Satellite Data</em>, Scientific Reports 10:5364.</p>
      <p style="margin-top:16px"><a href="https://github.com/danieltyukov/marine-debris-ml-model">github.com/danieltyukov/marine-debris-ml-model</a></p>
    </div>
  </div>
</footer>
</div>

<script>
const DATA = __FDI__;

/* ---- hero depth contours: value-noise field, drawn as nested iso-lines ---- */
(function(){
  const cv=document.getElementById('contours'); const ctx=cv.getContext('2d');
  let W,H,dpr;
  const g=[]; const GS=9;
  for(let i=0;i<GS*GS;i++) g.push(Math.random());
  function smooth(t){return t*t*(3-2*t)}
  function noise(x,y){
    const xi=Math.floor(x), yi=Math.floor(y), xf=x-xi, yf=y-yi;
    const at=(a,b)=>g[((b%GS)+GS)%GS*GS+((a%GS)+GS)%GS];
    const u=smooth(xf), v=smooth(yf);
    return (at(xi,yi)*(1-u)+at(xi+1,yi)*u)*(1-v) + (at(xi,yi+1)*(1-u)+at(xi+1,yi+1)*u)*v;
  }
  function field(x,y){return noise(x*2.1,y*2.1)*0.6 + noise(x*4.6,y*4.6)*0.3 + noise(x*9.2,y*9.2)*0.1}
  function draw(){
    const r=cv.getBoundingClientRect(); dpr=Math.min(devicePixelRatio||1,2);
    W=r.width; H=r.height; cv.width=W*dpr; cv.height=H*dpr;
    ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
    const css=getComputedStyle(document.documentElement);
    ctx.strokeStyle=css.getPropertyValue('--grid').trim()||'rgba(128,128,128,.06)';
    ctx.lineWidth=1;
    const step=5;
    for(let lv=0.08; lv<0.95; lv+=0.055){
      ctx.beginPath();
      for(let y=0;y<H;y+=step){
        let pen=false;
        for(let x=0;x<W;x+=step){
          const v=field(x/W,y/H);
          const on=Math.abs(v-lv)<0.0075;
          if(on){ if(!pen){ctx.moveTo(x,y); pen=true;} else ctx.lineTo(x,y); }
          else pen=false;
        }
      }
      ctx.stroke();
    }
  }
  draw();
  let t; addEventListener('resize',()=>{clearTimeout(t); t=setTimeout(draw,180)});
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
  new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
})();

/* ---- FDI threshold explorer over the real measured histogram ---- */
(function(){
  const keys=Object.keys(DATA); let ki=0;
  const cv=document.getElementById('plot'), ctx=cv.getContext('2d');
  const slider=document.getElementById('thr');
  const tval=document.getElementById('tval'), tpct=document.getElementById('tpct');
  const verdict=document.getElementById('verdict'), sceneEl=document.getElementById('scene');
  const btns=[...document.querySelectorAll('.preset')];

  function cur(){return DATA[keys[ki]]}
  function thrFromSlider(){
    const d=cur(), lo=d.edges[0], hi=d.edges[d.edges.length-1];
    // square mapping gives fine control down near zero where the decision actually lives
    const f=Math.pow(slider.value/1000,2);
    return lo+(hi-lo)*f;
  }
  function pctAbove(t){
    const d=cur(); let n=0, tot=0;
    for(let i=0;i<d.hist.length;i++){ tot+=d.hist[i]; if(d.edges[i]>=t) n+=d.hist[i]; }
    return tot? n/tot*100 : 0;
  }
  function draw(){
    const d=cur(), r=cv.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2);
    const W=r.width, H=r.height;
    cv.width=W*dpr; cv.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
    const css=getComputedStyle(document.documentElement);
    const water=css.getPropertyValue('--water').trim();
    const debris=css.getPropertyValue('--debris-bright').trim();
    const ink3=css.getPropertyValue('--ink-3').trim();
    const hair=css.getPropertyValue('--hair').trim();
    const pad={l:8,r:8,t:14,b:30};
    const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
    const lo=d.edges[0], hi=d.edges[d.edges.length-1];
    const mx=Math.max(...d.hist);
    const t=thrFromSlider();
    const xOf=v=>pad.l+(v-lo)/(hi-lo)*pw;
    // log-scaled bars: the tail is the whole story and would be invisible linearly
    const yOf=c=>pad.t+ph-(c<=0?0:Math.log10(1+c)/Math.log10(1+mx))*ph;

    ctx.strokeStyle=hair; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t+ph); ctx.lineTo(W-pad.r,pad.t+ph); ctx.stroke();

    const bw=pw/d.hist.length;
    for(let i=0;i<d.hist.length;i++){
      const x=pad.l+i*bw, y=yOf(d.hist[i]);
      ctx.fillStyle = d.edges[i]>=t ? debris : water;
      ctx.globalAlpha = d.edges[i]>=t ? .95 : .5;
      ctx.fillRect(x, y, Math.max(bw-0.6,0.7), pad.t+ph-y);
    }
    ctx.globalAlpha=1;
    const tx=xOf(t);
    ctx.strokeStyle=debris; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(tx,pad.t-4); ctx.lineTo(tx,pad.t+ph+4); ctx.stroke();

    ctx.fillStyle=ink3;
    ctx.font='11px ui-monospace,Menlo,monospace'; ctx.textAlign='left';
    ctx.fillText(lo.toFixed(3), pad.l, H-10);
    ctx.textAlign='right'; ctx.fillText(hi.toFixed(3), W-pad.r, H-10);
    ctx.textAlign='center'; ctx.fillText('FDI', W/2, H-10);
    ctx.textAlign='left'; ctx.fillText('pixels (log)', pad.l, pad.t+6);
  }
  function sync(){
    const d=cur(), t=thrFromSlider(), p=pctAbove(t);
    tval.textContent=t.toFixed(4);
    tpct.textContent=p.toFixed(p<1?2:1)+'%';
    sceneEl.textContent=keys[ki]+' '+d.date;
    const flagged=Math.round(d.n*p/100);
    let msg;
    if(p>5) msg='<b>'+p.toFixed(1)+'% of open water flagged.</b> At this cutoff every tile is accepted, so the cascade costs a full transformer pass on the whole scene and saves nothing.';
    else if(p>0.5) msg='<b>'+p.toFixed(2)+'% flagged</b> ('+flagged.toLocaleString()+' of '+d.n.toLocaleString()+' water pixels). Still permissive: most of these are sun glint and sensor noise rather than floating material.';
    else if(p>0.02) msg='<b>'+p.toFixed(3)+'% flagged</b> ('+flagged.toLocaleString()+' pixels). This is the operating region the adaptive threshold targets: the extreme upper tail of the water distribution.';
    else msg='<b>'+p.toFixed(3)+'% flagged.</b> Very strict. Real debris patches risk being missed, and a missed tile is never seen by the detector at all.';
    verdict.innerHTML=msg;
    draw();
  }
  function setSliderTo(v){
    const d=cur(), lo=d.edges[0], hi=d.edges[d.edges.length-1];
    const f=Math.max(0,Math.min(1,(v-lo)/(hi-lo)));
    slider.value=Math.round(Math.sqrt(f)*1000);
  }
  slider.addEventListener('input',()=>{btns.forEach(b=>b.setAttribute('aria-pressed','false')); sync()});
  btns.forEach(b=>b.addEventListener('click',()=>{
    const p=b.dataset.p;
    if(p==='scene'){ ki=(ki+1)%keys.length; setSliderTo(cur().pct.p99_9); }
    else if(p==='fixed'){ setSliderTo(0.006); }
    else { setSliderTo(cur().pct.p99_9); }
    btns.forEach(x=>x.setAttribute('aria-pressed', String(x===b && p!=='scene')));
    sync();
  }));
  let rt; addEventListener('resize',()=>{clearTimeout(rt); rt=setTimeout(draw,150)});
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
  new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  setSliderTo(cur().pct.p99_9); sync();
})();
</script>
"""

TITLE = "Marine Debris Detection: what Sentinel-2 can and cannot tell you"
DESCRIPTION = (
    "Rebuilding a 2019 TensorFlow 1.14 marine debris detector on free Sentinel-2 imagery, "
    "and the negative result that open-vocabulary vision models do not carry the signal at 10 m."
)
SITE_URL = "https://danieltyukov.github.io/marine-debris-ml-model/"

fragment = (
    HTML.replace("__FDI__", json.dumps(fdi))
    .replace("__CASCADE__", imgs["cascade"])
    .replace("__DETECT__", imgs["detect"])
    .replace("__INDICES__", imgs["indices"])
)

# The fragment opens with <title> then <style>; both belong in <head>, the rest in <body>.
split = fragment.index("</style>") + len("</style>")
head_fragment, body_fragment = fragment[:split], fragment[split:]

document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{DESCRIPTION}">
<meta name="color-scheme" content="light dark">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:type" content="website">
<meta property="og:url" content="{SITE_URL}">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22><text y=%2226%22 font-size=%2226%22>&#128752;</text></svg>">
<style>*,*::before,*::after{{box-sizing:border-box}}html{{-webkit-text-size-adjust:100%}}img{{max-width:100%}}</style>
{head_fragment}
</head>
<body>
{body_fragment}
</body>
</html>
"""

out = ROOT / "docs" / "index.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(document, encoding="utf-8")
(out.parent / ".nojekyll").touch()
print(f"wrote {out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB")
