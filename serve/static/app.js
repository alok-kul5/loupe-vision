// Inspection bench: pick or drop a part, scan it, show the verdict and a comparable heatmap.

const $ = (id) => document.getElementById(id);
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const pretty = (c) => c.replace(/_/g, " ");
const state = { file: null, blob: null, method: "patchcore", bench: {}, samples: [], categories: [] };
const METHOD = { patchcore: "PatchCore", cnn: "CNN + Grad-CAM" };

/* ── Setup ─────────────────────────────────────────────── */
async function init() {
  const [cats, samples, bench] = await Promise.all([
    fetch("/api/categories").then((r) => (r.ok ? r.json() : Promise.reject(r))),
    fetch("/static/samples/samples.json").then((r) => r.json()).catch(() => []),
    fetch("/static/benchmark.json").then((r) => r.json()).catch(() => ({})),
  ]).catch(async (r) => {
    $("err").textContent = (await r.json?.())?.detail || "Models are not loaded.";
    $("bar-meta").textContent = "Models not loaded";
    return [[], [], {}];
  });
  state.categories = cats; state.samples = samples; state.bench = bench;
  $("category").innerHTML = cats.map((c) => `<option value="${c}">${pretty(c)}</option>`).join("");
  if (cats.length) $("bar-meta").textContent = `${cats.length} part types · ONNX Runtime`;
  renderSamples();
  renderContext();
}

function renderSamples() {
  $("samples").innerHTML = state.samples.map((s, i) => `
    <button class="sample" type="button" style="--i:${i}" data-i="${i}" aria-label="${pretty(s.category)}, ${s.truth === "good" ? "good part" : s.defect}">
      <div class="ph"><img src="/static/samples/${s.file}" alt="" loading="lazy" width="448" height="448"></div>
      <div class="cap"><b>${pretty(s.category)}</b><span>${s.truth === "good" ? "good" : s.defect}</span></div>
    </button>`).join("");
  $("samples").querySelectorAll(".sample").forEach((b) => (b.onclick = () => pickSample(+b.dataset.i, b)));
}

function renderContext() {
  const c = $("category").value, m = state.bench[c];
  if (!m) { $("context").innerHTML = ""; return; }
  $("context").innerHTML = `<strong>On ${pretty(c)} in testing</strong>, scored on holdout images neither model saw:
    <div class="nums"><div><b>${m.patchcore.image_auroc.toFixed(3)}</b>PatchCore AUROC</div><div><b>${m.cnn.image_auroc.toFixed(3)}</b>CNN AUROC</div></div>`;
}

/* ── Choosing a part ───────────────────────────────────── */
async function pickSample(i, btn) {
  const s = state.samples[i];
  document.querySelectorAll(".sample").forEach((x) => x.classList.toggle("current", x === btn));
  $("category").value = s.category;
  renderContext();
  const blob = await fetch(`/static/samples/${s.file}`).then((r) => r.blob());
  await setImage(blob);
  $("stage").scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "center" });
  inspect();
}

async function setImage(blob) {
  state.blob = blob;
  // Show what the model sees: shorter side to 256, centre 224 (the same crop as the server).
  const img = await createImageBitmap(blob);
  const s = 256 / Math.min(img.width, img.height), crop = 224 / s;
  const cv = Object.assign(document.createElement("canvas"), { width: 448, height: 448 });
  cv.getContext("2d").drawImage(img, (img.width - crop) / 2, (img.height - crop) / 2, crop, crop, 0, 0, 448, 448);
  const url = cv.toDataURL("image/jpeg", 0.9);
  document.querySelectorAll(".compare").forEach((c) => {
    c.classList.remove("done", "scanning");
    c.querySelector(".before").src = url;
    c.querySelector(".after").removeAttribute("src");
    c.querySelector(".stamp").className = "stamp";
    c.querySelector(".stamp").textContent = "";
    setReveal(c, 100);
  });
  $("drop").hidden = true;
  $("views").hidden = false;
  $("go").disabled = false;
  $("again").hidden = false;
  $("result").hidden = true;
  layoutViews();
}

function layoutViews() {
  const both = state.method === "both";
  $("view-b").hidden = !both;
  $("views").style.setProperty("--cols", both ? 2 : 1);
  document.querySelectorAll(".compare").forEach((c) => c.style.setProperty("--h", `${c.clientHeight}px`));
}

/* ── Inspecting ────────────────────────────────────────── */
async function runOne(method, view, scoreEl) {
  const cmp = view.querySelector(".compare");
  cmp.classList.remove("done");
  cmp.style.setProperty("--h", `${cmp.clientHeight}px`);
  cmp.classList.add("scanning");
  const body = new FormData();
  body.append("file", state.blob, "part.jpg");
  body.append("category", $("category").value);
  body.append("method", method);
  const minScan = new Promise((r) => setTimeout(r, reduced() ? 0 : 900)); // let the scan read as a scan
  const res = await fetch("/api/inspect", { method: "POST", body });
  const d = await res.json();
  await minScan;
  cmp.classList.remove("scanning");
  if (!res.ok) throw new Error(d.detail || res.statusText);
  const after = cmp.querySelector(".after");
  after.src = `data:image/png;base64,${d.heatmap_png_base64}`;
  await after.decode().catch(() => {});
  cmp.classList.add("done");
  setReveal(cmp, 100);
  const stamp = cmp.querySelector(".stamp");
  stamp.textContent = d.verdict === "pass" ? "PASS" : "FAIL";
  stamp.className = `stamp ${d.verdict}`;
  void stamp.offsetWidth;
  stamp.classList.add("show");
  view.querySelector(".view-label").textContent = `${METHOD[method]} · drag to compare with the photo`;
  // Score card: the fill is the score against a scale where the threshold sits in the middle.
  const frac = Math.max(0, Math.min(1, d.score / (2 * d.threshold)));
  scoreEl.hidden = false;
  scoreEl.innerHTML = `<div class="row"><span class="name">${METHOD[method]}</span><span class="verdict ${d.verdict}">${d.verdict === "pass" ? "Pass" : "Fail"}</span></div>
    <div class="gauge"><div class="fill ${d.verdict}"></div><div class="mark" title="Threshold"></div></div>
    <div class="meta"><span>Score ${d.score.toFixed(3)} · limit ${d.threshold.toFixed(3)}</span><span>${Math.round(d.latency_ms)} ms</span></div>`;
  requestAnimationFrame(() => requestAnimationFrame(() => (scoreEl.querySelector(".fill").style.width = `${frac * 100}%`)));
  return d;
}

async function inspect() {
  if (!state.blob) return;
  $("err").textContent = "";
  const go = $("go");
  go.disabled = true; go.classList.add("busy"); go.textContent = "Scanning…";
  $("result").hidden = false;
  $("score-a").hidden = true; $("score-b").hidden = true;
  const stage = $("stage");
  try {
    if (state.method === "both") {
      await Promise.all([runOne("patchcore", $("view-a"), $("score-a")), runOne("cnn", $("view-b"), $("score-b"))]);
    } else {
      await runOne(state.method, $("view-a"), $("score-a"));
    }
    stage.classList.remove("shake"); void stage.offsetWidth; stage.classList.add("shake");
  } catch (e) {
    $("err").textContent = e.message;
  } finally {
    go.disabled = false; go.classList.remove("busy"); go.textContent = "Inspect again";
  }
}

/* ── Compare slider ────────────────────────────────────── */
function setReveal(cmp, v) {
  cmp.style.setProperty("--reveal", `${v}%`);
  cmp.querySelector(".slider").value = v;
}
document.querySelectorAll(".compare").forEach((c) => {
  c.querySelector(".slider").addEventListener("input", (e) => setReveal(c, e.target.value));
});

/* ── Controls ──────────────────────────────────────────── */
const seg = document.querySelector(".seg"), thumb = document.createElement("span");
thumb.className = "thumb"; thumb.setAttribute("aria-hidden", "true"); seg.prepend(thumb);
seg.querySelector('[aria-checked="true"]').classList.add("settled");
seg.querySelectorAll("button").forEach((b, i) => {
  b.onclick = () => {
    seg.querySelectorAll("button").forEach((x) => { x.setAttribute("aria-checked", String(x === b)); x.classList.remove("settled"); });
    thumb.style.transform = `translateX(${i * 100}%)`;
    // Once the thumb lands, the option carries its own fill (same look; readable without the thumb).
    clearTimeout(seg.settleT);
    seg.settleT = setTimeout(() => b.classList.add("settled"), reduced() ? 0 : 460);
    state.method = b.dataset.method;
    layoutViews();
    if (state.blob) inspect();
  };
});
$("category").onchange = () => { renderContext(); if (state.blob) inspect(); };
$("go").onclick = inspect;
$("again").onclick = () => $("file").click();
$("file").onchange = async () => { const f = $("file").files[0]; if (f) { await setImage(f); inspect(); } };
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", async (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) { await setImage(f); inspect(); }
});
// Drop onto the stage again after a result to swap the part.
$("stage").addEventListener("dragover", (e) => e.preventDefault());
$("stage").addEventListener("drop", async (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) { await setImage(f); inspect(); }
});
addEventListener("resize", layoutViews);

init();
