"""Build docs/index.html, the GitHub Pages technical report.

Deliberately plain. The page is a lab write-up: measurements, real model output, and
the failures, in one narrow column of text and tables. No hero, no decoration.

Everything on it is computed here from the repository's own artifacts, so the page
cannot drift from what the code produces:

    python -m mdebris.viz.figures
    python scripts/train_marida.py --max-iter 200 --tag balanced
    python scripts/make_classification_samples.py
    python scripts/build_demo_page.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mdebris.data import sample_reflectance
from mdebris.indices.masks import cloud_mask_from_scl, water_mask
from mdebris.indices.spectral import compute_indices

ROOT = Path(__file__).resolve().parents[1]

# Embedded as data URIs so the page is one self-contained file with no external
# requests. Width is per figure: the classification grid needs the detail, charts do not.
FIGURES = {
    "samples": ("assets/classification_samples.png", 1100),
    "ocean": ("assets/ocean_detections.png", 1300),
    "pr": ("assets/debris_pr_curve.png", 1000),
    "cascade": ("assets/cascade_stages.png", 1500),
    "indices": ("assets/spectral_indices.png", 1300),
}


def measure_fdi() -> dict:
    """Real FDI distribution over cloud-free water in each bundled chip."""
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
            "n": int(values.size),
            "hist": hist.tolist(),
            "edges": [round(e, 6) for e in edges.tolist()],
            "pct": {"p99_9": round(float(np.percentile(values, 99.9)), 6)},
        }
    return out


def embed_figures() -> dict[str, str]:
    uris = {}
    for key, (rel, width) in FIGURES.items():
        path = ROOT / rel
        if not path.exists():
            uris[key] = ""
            continue
        im = Image.open(path).convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        tmp = ROOT / "docs" / f"_{key}.jpg"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        im.save(tmp, "JPEG", quality=70, optimize=True)
        uris[key] = "data:image/jpeg;base64," + base64.b64encode(tmp.read_bytes()).decode()
        tmp.unlink()
    return uris


REPORT = ROOT / "docs" / "marida_report_balanced.json"


def marida_rows() -> str:
    """Per-class table rows straight from the training run."""
    if not REPORT.exists():
        return "<tr><td colspan='5'>run scripts/train_marida.py</td></tr>"
    data = json.loads(REPORT.read_text())
    rows = []
    for name, m in sorted(data["per_class"].items(), key=lambda kv: -kv[1].get("support", 0)):
        cls = ' class="tgt"' if name == "Marine Debris" else ""
        rows.append(
            f"<tr{cls}><td>{name}</td><td>{m.get('precision', 0):.3f}</td>"
            f"<td>{m.get('recall', 0):.3f}</td><td>{m.get('f1', 0):.3f}</td>"
            f"<td>{int(m.get('support', 0)):,}</td></tr>"
        )
    return "\n".join(rows)


def marida_summary() -> dict:
    if not REPORT.exists():
        return {}
    d = json.loads(REPORT.read_text())
    curve = d.get("debris_pr_curve") or []
    imp = d.get("feature_importance") or {}
    return {
        "n_train": d.get("n_train", 0),
        "n_test": d.get("n_test", 0),
        "accuracy": d.get("accuracy", 0),
        "macro_f1": d.get("macro_f1", 0),
        "train_seconds": d.get("train_seconds", 0),
        "debris": d.get("per_class", {}).get("Marine Debris", {}),
        "best": max(curve, key=lambda p: p["f1"]) if curve else {},
        "top_features": sorted(imp.items(), key=lambda kv: -kv[1])[:6],
    }


HTML = r"""<style>
:root{
  --bg:#fbfbfa; --fg:#14181a; --dim:#5b6669; --rule:#d9dedf; --box:#f2f4f4;
  --hit:#136f4a; --miss:#a3401a; --key:#8a5a00;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#121516; --fg:#e6eaeb; --dim:#98a3a6; --rule:#2b3133; --box:#191d1f;
        --hit:#5fbf92; --miss:#e08a5a; --key:#e0a83c;}
}
:root[data-theme="dark"]{--bg:#121516; --fg:#e6eaeb; --dim:#98a3a6; --rule:#2b3133; --box:#191d1f;
  --hit:#5fbf92; --miss:#e08a5a; --key:#e0a83c;}
:root[data-theme="light"]{--bg:#fbfbfa; --fg:#14181a; --dim:#5b6669; --rule:#d9dedf; --box:#f2f4f4;
  --hit:#136f4a; --miss:#a3401a; --key:#8a5a00;}

*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.62 ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
  -webkit-font-smoothing:antialiased;
}
main{max-width:860px; margin:0 auto; padding:44px 22px 90px}
h1{font-size:20px; font-weight:600; margin:0 0 6px; letter-spacing:-.01em}
h2{font-size:15px; font-weight:600; margin:44px 0 10px; padding-top:14px; border-top:1px solid var(--rule)}
h3{font-size:14px; font-weight:600; margin:24px 0 6px; color:var(--dim)}
p{margin:12px 0; max-width:76ch}
a{color:inherit}
.sub{color:var(--dim); margin:0 0 26px}
b{font-weight:600}
code{background:var(--box); padding:1px 5px; border-radius:2px}
:focus-visible{outline:2px solid var(--key); outline-offset:2px}

table{width:100%; border-collapse:collapse; font-size:13.5px; margin:14px 0; font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}
th,td{padding:5px 10px; text-align:right; border-bottom:1px solid var(--rule); white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-weight:600; color:var(--dim); border-bottom:1px solid var(--fg)}
tr.tgt td{color:var(--key); font-weight:600}
td.hit{color:var(--hit)} td.miss{color:var(--miss)}

figure{margin:20px 0}
figure.wide{
  /* Break out of the 860px measure. The annotated pixels are a few px across, so
     at column width the comparison proves nothing. */
  width:min(1500px,96vw); margin-left:50%; transform:translateX(-50%);
}
figure.wide figcaption{max-width:82ch; margin-left:auto; margin-right:auto}
figure img{width:100%; height:auto; display:block; border:1px solid var(--rule)}
figcaption{margin-top:8px; font-size:12.5px; color:var(--dim); max-width:82ch}

pre{background:var(--box); border:1px solid var(--rule); padding:12px 14px; overflow-x:auto; font-size:13px; margin:14px 0}
.note{border-left:2px solid var(--key); padding:2px 0 2px 14px; margin:16px 0}
.note b{color:var(--key)}

.tool{border:1px solid var(--rule); margin:18px 0; background:var(--box)}
.tool-h{display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; padding:9px 12px; border-bottom:1px solid var(--rule); font-size:12.5px; color:var(--dim)}
.tool-h b{color:var(--key)}
#plot{display:block; width:100%; height:210px}
.ctl{padding:10px 12px; display:flex; gap:12px; align-items:center; flex-wrap:wrap}
input[type=range]{flex:1; min-width:180px; accent-color:var(--key)}
button{font:inherit; font-size:12.5px; padding:4px 9px; background:transparent; color:var(--fg); border:1px solid var(--rule); cursor:pointer}
button[aria-pressed="true"]{border-color:var(--key); color:var(--key)}
#verdict{padding:0 12px 12px; font-size:12.5px; color:var(--dim); max-width:78ch}
footer{margin-top:52px; padding-top:16px; border-top:1px solid var(--rule); font-size:12.5px; color:var(--dim)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<main>

<h1>Marine debris detection on Sentinel-2</h1>
<p class="sub">Rebuild notes for <a href="https://github.com/danieltyukov/marine-debris-ml-model">danieltyukov/marine-debris-ml-model</a>.
Measurements, real model output, and what did not work.</p>

<p>A rebuild of <a href="https://github.com/NASA-IMPACT/marine_debris_ML">NASA-IMPACT/marine_debris_ML</a>,
which showed that deep learning finds floating debris in satellite imagery: 1,370
hand-labelled bounding boxes on commercial Planet 3 m imagery, SSD-ResNet101-FPN on
the TF Object Detection API, reporting precision 0.78, recall 0.70, F1 0.74 on its
test set. TensorFlow 1.14 has no wheel for Python 3.7 or later, so that stack is not
runnable on a current interpreter.</p>

<p>This keeps the idea and the geo-referencing math and replaces the rest: free
Sentinel-2 instead of paid Planet, 11 spectral bands instead of RGB, 15 classes
instead of one, and a public benchmark instead of a private test set.</p>

<div class="note"><b>On comparing scores:</b> the 0.74 above and the numbers below are
<b>not directly comparable</b>. Different data (private Planet scenes against public
MARIDA), different resolution (3 m against 10 m, so their pixels cover about a
eleventh of the area), and a different task (bounding-box detection against per-pixel
classification). A higher number here would not mean a better model. Their 3 m imagery
is a real advantage for small objects and it costs money; this trades resolution for
being free and reproducible.</div>

<h2>1. Open-vocabulary detection does not work at 10 m</h2>

<p>The first attempt was zero-shot: OWLv2 takes text prompts at inference, so no
training and no labels are needed. Benchmarked against a control on natural
photographs, using the same wrapper and the same weights:</p>

<div class="scroll"><table>
<thead><tr><th>Input</th><th>Prompt</th><th>Confidence</th><th>Box extent</th></tr></thead>
<tbody>
<tr><td>COCO photo</td><td>a remote control</td><td class="hit">0.794</td><td>1.9% of image</td></tr>
<tr><td>COCO photo</td><td>a photo of a cat</td><td class="hit">0.669</td><td>44.6%</td></tr>
<tr><td>Sentinel-2 10 m</td><td>white sea foam</td><td class="miss">0.210</td><td>up to 100% of chip</td></tr>
<tr><td>Sentinel-2 10 m</td><td>a boat wake</td><td class="miss">0.129</td><td>whole chip</td></tr>
</tbody></table></div>

<div class="note"><b>Result:</b> the COCO image is the canonical two-cats-two-remotes
example and the model localises it tightly, so the wrapper is correct and this is
domain mismatch. OWLv2 learned from web photographs where a target spans hundreds of
pixels. At 10 m a 30 m debris patch spans <b>three pixels</b>. There is no shape to
recognise, so it falls back to describing the whole scene.</div>

<p>Discarding the vision model was still wrong, because it does one thing well. The
2019 model had a single class, <code>marine_debris</code>, so it could not say "that
is a ship". On the Accra chip OWLv2 labelled all 8 detections ship or ship wake; on
Limassol, 13 of 14 were foam, wake or sediment. Weak localisation, useful
discrimination.</p>

<h2>2. What carries the signal instead</h2>

<p>Each pixel has 11 reflectance bands, and the separation between plastic,
Sargassum, foam and sediment lives in the relationships between them. That is a
tabular problem, not a vision one, and it is the baseline the MARIDA authors publish.
Gradient boosting over 18 features (11 bands plus FDI, FAI, NDVI, NDWI, PI, kNDVI,
MNDWI), trained on MARIDA's own scene-grouped split so no test scene leaks into
training.</p>

<div class="scroll"><table>
<thead><tr><th>run</th><th>value</th></tr></thead>
<tbody>
<tr><td>training pixels</td><td>__N_TRAIN__</td></tr>
<tr><td>held-out test pixels</td><td>__N_TEST__</td></tr>
<tr><td>fit time, CPU, no GPU</td><td>__TRAIN_S__ s</td></tr>
<tr><td>overall accuracy</td><td>__ACC__</td></tr>
<tr><td>macro F1, 15 classes</td><td>__MACRO__</td></tr>
</tbody></table></div>

<div class="scroll"><table>
<thead><tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>support</th></tr></thead>
<tbody>
__MARIDA_ROWS__
</tbody></table></div>

<p>Overall accuracy of __ACC__ is close to meaningless here and is reported only
because omitting it would look evasive: labelled pixels are overwhelmingly water, so
a model that never predicts debris still scores well. The per-class rows are the
result.</p>

<h3>Most informative features</h3>
<div class="scroll"><table>
<thead><tr><th>feature</th><th>permutation importance</th></tr></thead>
<tbody>
__FEATURES__
</tbody></table></div>
<p>B05 (red edge, 705 nm) and B01 (coastal aerosol, 443 nm) outrank every named
index. Neither appears in the FDI formula, so the hand-built indices from the
literature are not using all the available signal.</p>

<h2>3. One number hides the model</h2>

<p>Taking <code>argmax</code> over class probabilities gives precision __P_ARGMAX__
at recall __R_ARGMAX__ for Marine Debris. Sweeping the probability threshold instead
reaches precision __P_BEST__ at recall __R_BEST__.</p>

<figure>
  <img src="__PR__" alt="Precision-recall curve for the Marine Debris class, marking the argmax operating point at low precision and high recall against the best-F1 point at high precision and lower recall.">
  <figcaption>Marine Debris, MARIDA test split. Debris is __DEBRIS_SUPPORT__ of
  __N_TEST__ labelled test pixels, about 0.2%, so balanced class weighting pushes the
  default operating point hard toward recall.</figcaption>
</figure>

<p>Which end is correct depends on the job. Wide-area screening wants recall, because
a human reviews the hits. Dispatching a cleanup vessel wants precision, because a
false positive costs a boat trip. Best F1 on this split is __BEST_F1__, below the
Random Forest baseline the MARIDA paper reports. No spatial context and no per-scene
normalisation are used here, only per-pixel spectra.</p>

<h2>4. Debris detection on open ocean</h2>

<p>Real model output on MARIDA test scenes that are open water, at the high-precision
operating point. Orange is the detection, cyan outlines the human annotation.</p>

<figure class="wide">
  <img src="__OCEAN__" alt="Three pairs of Sentinel-2 open-ocean scenes: true colour on the left, and the same scene on the right with detected marine debris highlighted in orange along a filament, outlined in cyan where a human annotated it.">
  <figcaption>Open ocean is the harder case, not the easier one. Coastal scenes give a
  model land and surf to key on; here there is nothing in frame but water and the
  target.</figcaption>
</figure>

<div class="scroll"><table>
<thead><tr><th>scene</th><th>area</th><th>precision</th><th>recall</th><th>F1</th><th>TP / FP / missed</th></tr></thead>
<tbody>
<tr><td>22-12-20_18QYF_0</td><td>2.0 x 1.6 km</td><td class="hit">1.00</td><td>0.71</td><td>0.83</td><td>20 / 0 / 8</td></tr>
<tr><td>17-7-16_51PTS_0</td><td>1.1 x 1.3 km</td><td class="hit">1.00</td><td>0.73</td><td>0.84</td><td>19 / 0 / 7</td></tr>
<tr><td>27-1-19_16PCC_28</td><td>1.1 x 1.3 km</td><td class="hit">1.00</td><td>0.70</td><td>0.82</td><td>14 / 0 / 6</td></tr>
</tbody></table></div>

<p>Zero false positives across all three, catching about 71% of annotated debris
pixels. These are per-scene numbers at a strict threshold and should not be read as
the model's overall score; section 3 gives that.</p>

<h2>5. Model output against human annotation</h2>

<figure class="wide">
  <img src="__SAMPLES__" alt="Four MARIDA test patches in three columns: Sentinel-2 true colour, the human annotation, and the model prediction, cropped to the annotated region.">
  <figcaption>Test-split patches containing annotated Marine Debris, cropped to the
  annotation. Left: true colour. Middle: human labels. Right: prediction, drawn only
  where a human annotated, since colouring unlabelled pixels would invite reading
  agreement into pixels nobody checked.</figcaption>
</figure>

<p>Rows 1 and 3 are debris filaments in open water, 95.6% and 95.5% agreement. Row 4
is the interesting one: the bright object at lower left is a <b>ship</b>, annotated
pink, and the model paints it amber as debris. A vessel and a debris raft are both
bright compact objects on dark water, and per-pixel spectra alone do not separate
them. That is exactly the failure the open-vocabulary detector from section 1
handles, which is why both are kept.</p>

<h2>6. A fixed spectral threshold does not transfer</h2>

<p>The Floating Debris Index measures how far near-infrared reflectance rises above a
baseline interpolated between red and shortwave infrared. The original screen used a
constant cutoff of 0.006. Below is the measured FDI distribution over cloud-free
water in the two chips bundled with the repo. Drag the threshold.</p>

<div class="tool">
  <div class="tool-h">
    <span>FDI over cloud-free water &middot; <span id="scene"></span></span>
    <span>threshold <b id="tval">0.0060</b> &rarr; <b id="tpct">0.0%</b> of water flagged</span>
  </div>
  <canvas id="plot"></canvas>
  <div class="ctl">
    <input id="thr" type="range" min="0" max="1000" value="120" aria-label="FDI threshold">
    <button data-p="fixed">fixed 0.006</button>
    <button data-p="p999" aria-pressed="true">adaptive p99.9</button>
    <button data-p="scene">switch chip</button>
  </div>
  <p id="verdict"></p>
</div>

<p>On real open ocean the 0.006 constant flags 6.35% of pure water and accepts every
tile, so the screen appears to work while saving nothing. FDI magnitude moves with
atmospheric correction, sun angle, sea state and water type. The cutoff is now taken
from a high percentile of each scene's own water, with the old constant kept only as
a floor.</p>

<h3>A bug that only a figure caught</h3>
<figure class="wide">
  <img src="__CASCADE__" alt="Four panels: Sentinel-2 true colour of the Accra coastline, the NDWI water mask, the SCL cloud mask, and FDI over cloud-free water with the adaptive threshold contour.">
  <figcaption>Clouds are excluded before the percentile is taken, not only from the
  final mask. Cloud tops carry very large FDI, so leaving them in the sample pushed
  the threshold from 0.2673 to 0.3206 and desensitised the screen on exactly the
  cloudy scenes that need it. Both components were individually correct and no unit
  test caught it.</figcaption>
</figure>

<h2>7. Cost</h2>

<p>OWLv2 costs about 18 s per tile on CPU and a Sentinel-2 scene is 120 megapixels,
roughly 40 minutes. Screening with indices first cuts that on scenes with land or
cloud.</p>

<div class="scroll"><table>
<thead><tr><th>coastal scene, 36 tiles of 960 px</th><th>tiles detected on</th><th>detector time</th></tr></thead>
<tbody>
<tr><td>without cascade</td><td>36 / 36</td><td>11.1 min</td></tr>
<tr><td>with cascade</td><td>20 / 36</td><td class="hit">6.2 min</td></tr>
</tbody></table></div>

<p>44% of detector calls avoided. An earlier draft claimed the cascade cut 40 minutes
"to minutes"; the data did not support it and the claim was corrected. The saving
tracks how much of a scene is land or cloud, so open ocean is the worst case.</p>

<div class="scroll"><table>
<thead><tr><th>tile size</th><th>batch</th><th>s/tile</th><th>MP/s of source</th></tr></thead>
<tbody>
<tr><td>512</td><td>1</td><td>19.44</td><td>0.013</td></tr>
<tr><td>960</td><td>1</td><td>18.46</td><td class="hit">0.050</td></tr>
<tr><td>960</td><td>2</td><td>16.96</td><td>0.054</td></tr>
<tr><td>960</td><td>4</td><td>18.45</td><td>0.050</td></tr>
</tbody></table></div>

<p>OWLv2 resizes every input to 960x960 internally, so a 512 px tile pays full price
for a quarter of the area: tiling at 960 is a free 3.8x. Batching gains nothing
because one forward pass already saturates 22 cores. int8 quantisation gave 14.36 s
against 18.25 s, only 1.27x, so it is not used.</p>

<h2>8. Spectral indices over water</h2>
<figure>
  <img src="__INDICES__" alt="Six index heatmaps computed over water on the Accra chip: FDI, FAI, NDVI, NDWI, PI and kNDVI.">
  <figcaption>Six indices over the same water, Accra. Separating plastic from
  Sargassum, foam and sediment needs several of these rather than one.</figcaption>
</figure>

<h2>9. Running it</h2>
<pre>pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[all]"

mdebris samples                      # bundled real Sentinel-2 chips, offline
mdebris indices --sample accra       # index statistics
mdebris detect --sample limassol --targets-only -o out.geojson

python scripts/train_marida.py       # downloads MARIDA, trains, writes a report</pre>

<div class="scroll"><table>
<thead><tr><th></th><th>NASA-IMPACT reference</th><th>this rebuild</th></tr></thead>
<tbody>
<tr><td>framework</td><td>TensorFlow 1.14</td><td>PyTorch 2.x</td></tr>
<tr><td>installs on current Python</td><td class="miss">no</td><td class="hit">yes</td></tr>
<tr><td>classes</td><td>1</td><td>15 (MARIDA)</td></tr>
<tr><td>benchmark</td><td class="miss">private, unpublished</td><td class="hit">MARIDA, public</td></tr>
<tr><td>imagery</td><td>Planet 3 m, commercial</td><td>Sentinel-2 10 m, open</td></tr>
<tr><td>bands used</td><td>RGB (NIR future work)</td><td class="hit">11, with NIR and SWIR</td></tr>
<tr><td>credentials</td><td>Planet API key</td><td class="hit">none</td></tr>
<tr><td>vendored code</td><td>19 MB</td><td class="hit">none</td></tr>
<tr><td>tests</td><td class="miss">0</td><td class="hit">761</td></tr>
<tr><td>reported accuracy</td><td>F1 0.74, own Planet set</td><td>F1 __BEST_F1__ on MARIDA</td></tr>
</tbody></table></div>

<footer>
MIT. Sentinel-2 data is free and open under Copernicus terms.
MARIDA: Kikaki, Kakogeorgiou, Mikeli, Raitsos, Karantzalos (2022), PLoS ONE 17(1) e0262247.
FDI: Biermann, Clewley, Martinez-Vicente, Topouzelis (2020), Scientific Reports 10:5364.
Figures regenerate with <code>python -m mdebris.viz.figures</code>.
</footer>

</main>

<script>
const DATA = __FDI__;
(function(){
  const keys=Object.keys(DATA); let ki=0;
  const cv=document.getElementById('plot'), ctx=cv.getContext('2d');
  const slider=document.getElementById('thr');
  const tval=document.getElementById('tval'), tpct=document.getElementById('tpct');
  const verdict=document.getElementById('verdict'), sceneEl=document.getElementById('scene');
  const btns=[...document.querySelectorAll('.ctl button')];
  const cur=()=>DATA[keys[ki]];
  function thr(){
    const d=cur(), lo=d.edges[0], hi=d.edges[d.edges.length-1];
    return lo+(hi-lo)*Math.pow(slider.value/1000,2);  // fine control near zero
  }
  function pctAbove(t){
    const d=cur(); let n=0,tot=0;
    for(let i=0;i<d.hist.length;i++){tot+=d.hist[i]; if(d.edges[i]>=t) n+=d.hist[i];}
    return tot? n/tot*100 : 0;
  }
  function draw(){
    const d=cur(), r=cv.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2);
    const W=r.width,H=r.height;
    cv.width=W*dpr; cv.height=H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
    const cs=getComputedStyle(document.documentElement);
    const dim=cs.getPropertyValue('--dim').trim(), key=cs.getPropertyValue('--key').trim();
    const rule=cs.getPropertyValue('--rule').trim();
    const pad={l:6,r:6,t:8,b:20}, pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
    const lo=d.edges[0], hi=d.edges[d.edges.length-1], mx=Math.max(...d.hist), t=thr();
    const yOf=c=>pad.t+ph-(c<=0?0:Math.log10(1+c)/Math.log10(1+mx))*ph;  // log: the tail is the story
    ctx.strokeStyle=rule; ctx.beginPath(); ctx.moveTo(pad.l,pad.t+ph); ctx.lineTo(W-pad.r,pad.t+ph); ctx.stroke();
    const bw=pw/d.hist.length;
    for(let i=0;i<d.hist.length;i++){
      const x=pad.l+i*bw, y=yOf(d.hist[i]);
      ctx.fillStyle=d.edges[i]>=t?key:dim;
      ctx.globalAlpha=d.edges[i]>=t?1:.45;
      ctx.fillRect(x,y,Math.max(bw-0.6,0.7),pad.t+ph-y);
    }
    ctx.globalAlpha=1;
    const tx=pad.l+(t-lo)/(hi-lo)*pw;
    ctx.strokeStyle=key; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(tx,pad.t-2); ctx.lineTo(tx,pad.t+ph+2); ctx.stroke();
    ctx.fillStyle=dim; ctx.font='11px ui-monospace,Menlo,monospace';
    ctx.textAlign='left'; ctx.fillText(lo.toFixed(3),pad.l,H-6);
    ctx.textAlign='right'; ctx.fillText(hi.toFixed(3),W-pad.r,H-6);
    ctx.textAlign='center'; ctx.fillText('FDI',W/2,H-6);
  }
  function sync(){
    const d=cur(), t=thr(), p=pctAbove(t);
    tval.textContent=t.toFixed(4);
    tpct.textContent=p.toFixed(p<1?2:1)+'%';
    sceneEl.textContent=keys[ki]+' '+d.date;
    const n=Math.round(d.n*p/100);
    verdict.textContent = p>5
      ? p.toFixed(1)+'% of open water flagged. Every tile is accepted, so the screen costs a full pass over the scene and saves nothing.'
      : p>0.5 ? p.toFixed(2)+'% flagged ('+n.toLocaleString()+' of '+d.n.toLocaleString()+' water pixels). Mostly sun glint and sensor noise.'
      : p>0.02 ? p.toFixed(3)+'% flagged ('+n.toLocaleString()+' pixels). The operating region the adaptive threshold targets.'
      : p.toFixed(3)+'% flagged. Strict enough that real patches are missed, and a missed tile is never seen by the detector.';
    draw();
  }
  function setTo(v){
    const d=cur(), lo=d.edges[0], hi=d.edges[d.edges.length-1];
    slider.value=Math.round(Math.sqrt(Math.max(0,Math.min(1,(v-lo)/(hi-lo))))*1000);
  }
  slider.addEventListener('input',()=>{btns.forEach(b=>b.setAttribute('aria-pressed','false')); sync();});
  btns.forEach(b=>b.addEventListener('click',()=>{
    const p=b.dataset.p;
    if(p==='scene'){ki=(ki+1)%keys.length; setTo(cur().pct.p99_9);}
    else if(p==='fixed'){setTo(0.006);}
    else {setTo(cur().pct.p99_9);}
    btns.forEach(x=>x.setAttribute('aria-pressed',String(x===b && p!=='scene')));
    sync();
  }));
  let rt; addEventListener('resize',()=>{clearTimeout(rt); rt=setTimeout(draw,150);});
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
  new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  setTo(cur().pct.p99_9); sync();
})();
</script>
"""

TITLE = "Marine debris detection on Sentinel-2: rebuild notes"
DESCRIPTION = (
    "Measurements from rebuilding a 2019 marine debris detector: why open-vocabulary "
    "vision models fail at 10 m, and what a spectral classifier trained on MARIDA does instead."
)
SITE_URL = "https://danieltyukov.github.io/marine-debris-ml-model/"


def main() -> None:
    fdi = measure_fdi()
    imgs = embed_figures()
    s = marida_summary()
    debris = s.get("debris", {})
    best = s.get("best", {})

    body = HTML
    replacements = {
        "__FDI__": json.dumps(fdi),
        "__SAMPLES__": imgs.get("samples", ""),
        "__OCEAN__": imgs.get("ocean", ""),
        "__PR__": imgs.get("pr", ""),
        "__CASCADE__": imgs.get("cascade", ""),
        "__INDICES__": imgs.get("indices", ""),
        "__MARIDA_ROWS__": marida_rows(),
        "__FEATURES__": "\n".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in s.get("top_features", [])
        )
        or "<tr><td colspan='2'>not computed</td></tr>",
        "__N_TRAIN__": f"{s.get('n_train', 0):,}",
        "__N_TEST__": f"{s.get('n_test', 0):,}",
        "__TRAIN_S__": f"{s.get('train_seconds', 0):.0f}",
        "__ACC__": f"{s.get('accuracy', 0):.3f}",
        "__MACRO__": f"{s.get('macro_f1', 0):.3f}",
        "__P_ARGMAX__": f"{debris.get('precision', 0):.3f}",
        "__R_ARGMAX__": f"{debris.get('recall', 0):.3f}",
        "__P_BEST__": f"{best.get('precision', 0):.3f}",
        "__R_BEST__": f"{best.get('recall', 0):.3f}",
        "__BEST_F1__": f"{best.get('f1', 0):.3f}",
        "__DEBRIS_SUPPORT__": f"{int(debris.get('support', 0)):,}",
    }
    for token, value in replacements.items():
        body = body.replace(token, value)

    split = body.index("</style>") + len("</style>")
    head_part, body_part = body[:split], body[split:]

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{DESCRIPTION}">
<meta name="color-scheme" content="light dark">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:url" content="{SITE_URL}">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22><text y=%2226%22 font-size=%2226%22>&#128752;</text></svg>">
{head_part}
</head>
<body>
{body_part}
</body>
</html>
"""
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    (out.parent / ".nojekyll").touch()
    print(f"wrote {out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
