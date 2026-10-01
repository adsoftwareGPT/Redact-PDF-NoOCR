/* pdf-redact-web — viewer + box editing */
const $ = s => document.querySelector(s);
let job = null;            // {id, npages, name}
let boxes = [];            // {id,page,bbox:[x1,y1,x2,y2] 0-1000,category,text,enabled,custom}
let zoom = 1;
let pollTimer = null;

/* ---------- upload ---------- */
const dz = $("#dropzone"), fi = $("#fileInput");
dz.onclick = () => fi.click();
dz.ondragover = e => { e.preventDefault(); dz.classList.add("drag"); };
dz.ondragleave = () => dz.classList.remove("drag");
dz.ondrop = e => { e.preventDefault(); dz.classList.remove("drag"); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); };
fi.onchange = () => fi.files[0] && upload(fi.files[0]);

async function upload(file) {
  const fd = new FormData(); fd.append("file", file);
  toast("Uploading & rasterizing…");
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  if (!r.ok) { toast("Upload failed: " + (await r.text())); return; }
  const j = await r.json();
  job = { id: j.job_id, npages: j.npages, name: j.name };
  location.hash = j.job_id;
  startUI();
  detect();
}

function startUI() {
  $("#uploadCard").hidden = true;
  $("#detectCard").hidden = false;
  $("#helpCard").hidden = false;
  $("#docInfo").textContent = `${job.name} · ${job.npages} page(s)`;
  const v = $("#viewer"); v.innerHTML = "";
  for (let n = 1; n <= job.npages; n++) {
    const w = document.createElement("div"); w.className = "pwrap"; w.dataset.page = n;
    const num = document.createElement("div"); num.className = "pnum"; num.textContent = "page " + n;
    const img = new Image(); img.src = `/api/page/${job.id}/${n}`; img.draggable = false;
    const cv = document.createElement("canvas");
    w.append(num, img, cv); v.append(w);
    img.onload = () => layoutPage(w);
    w._img = img; w._cv = cv;
    attachEvents(w);
  }
  layout();
}

async function detect() {
  await fetch(`/api/detect/${job.id}`, { method: "POST" });
  $("#detectState").textContent = "detecting…"; $("#detectState").className = "chip run";
  pollTimer = setInterval(poll, 1500);
}

async function poll() {
  const r = await fetch(`/api/status/${job.id}`); const s = await r.json();
  $("#progBar").style.width = (100 * s.pages_done / s.npages) + "%";
  if (s.state === "detected") {
    clearInterval(pollTimer);
    $("#detectState").textContent = "done"; $("#detectState").className = "chip ok";
    $("#exportBtn").disabled = false;
  } else if (s.state === "error") {
    clearInterval(pollTimer);
    $("#detectState").textContent = "error"; $("#detectState").className = "chip err";
    toast("Detection error: " + s.error); return;
  } else { $("#detectState").textContent = `detecting… ${s.pages_done}/${s.npages}`; }
  // merge boxes: keep user edits for known ids; NEVER drop customs (server
  // echoes them back, but a just-drawn one may not be stored yet)
  const have = new Map(boxes.map(b => [b.id, b]));
  const ids = new Set(s.boxes.map(b => b.id));
  boxes = [...s.boxes.map(b => have.get(b.id) || b), ...boxes.filter(b => b.custom && !ids.has(b.id))];
  renderCats(); drawAll(); stats(); if (s.usage) showUsage(s.usage);
}

$("#redetectBtn").onclick = () => { boxes = boxes.filter(b => b.custom); drawAll(); stats(); saveCustom(); detect(); };

/* ---------- custom boxes: persist server-side (survive refresh/restart) ---------- */
let saveCustomT = null;
function saveCustom() {
  if (!job) return;
  clearTimeout(saveCustomT);
  const cs = boxes.filter(b => b.custom).map(({ id, page, bbox }) => ({ id, page, bbox }));
  saveCustomT = setTimeout(() => fetch(`/api/custom/${job.id}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ custom: cs })
  }).catch(() => {}), 300);
}

/* ---------- categories ---------- */
let catOff = new Set();
function renderCats() {
  const counts = {};
  boxes.filter(b => !b.custom).forEach(b => counts[b.category] = (counts[b.category] || 0) + 1);
  const el = $("#cats"); el.innerHTML = "";
  Object.keys(counts).sort().forEach(c => {
    const d = document.createElement("div");
    d.className = "cat" + (catOff.has(c) ? " off" : "");
    d.innerHTML = `${c}<span class="n">${counts[c]}</span>`;
    d.onclick = () => {
      catOff.has(c) ? catOff.delete(c) : catOff.add(c);
      boxes.forEach(b => { if (b.category === c) b.enabled = !catOff.has(c); });
      renderCats(); drawAll(); stats();
    };
    el.append(d);
  });
}
function fmtN(n) {
  if (n == null) return "–";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}
function fmtU(u) {
  if (!u || !u.calls) return "<i>none</i>";
  const cost = u.cost_known && u.cost > 0 ? ` · ≈ $${u.cost.toFixed(3)}` : "";
  return `${u.calls} calls · ${fmtN(u.prompt)} in / ${fmtN(u.completion)} out tokens${cost}`;
}
function showUsage(u) {
  const card = $("#usageCard"); card.hidden = false;
  $("#usageOut").innerHTML =
    `🔍 detection: ${fmtU(u.detect)}<br>🔎 leak audit: ${fmtU(u.audit)}`;
}
function stats() {
  const on = boxes.filter(b => b.enabled).length, cu = boxes.filter(b => b.custom).length;
  $("#boxStats").textContent = `${on} of ${boxes.length} boxes active · ${cu} custom`;
}

/* ---------- canvas rendering ---------- */
function layoutPage(w) {
  const img = w._img, cv = w._cv;
  cv.width = img.clientWidth; cv.height = img.clientHeight;
  cv.style.width = img.clientWidth + "px"; cv.style.height = img.clientHeight + "px";
  drawPage(w);
}
function layout() { document.querySelectorAll(".pwrap").forEach(layoutPage); }
function drawPage(w) {
  const cv = w._cv, ctx = cv.getContext("2d"), page = +w.dataset.page;
  ctx.clearRect(0, 0, cv.width, cv.height);
  const sx = cv.width / 1000, sy = cv.height / 1000;
  boxes.filter(b => b.page === page).forEach(b => {
    const [x1, y1, x2, y2] = b.bbox;
    const X = x1 * sx, Y = y1 * sy, W = (x2 - x1) * sx, H = (y2 - y1) * sy;
    if (b.custom) {
      ctx.fillStyle = "rgba(37,99,235,.35)"; ctx.fillRect(X, Y, W, H);
      ctx.strokeStyle = "#2563eb"; ctx.lineWidth = 2; ctx.strokeRect(X, Y, W, H);
      ctx.fillStyle = "#2563eb"; ctx.font = "bold 11px sans-serif"; ctx.fillText("×", X + W - 4, Y + 12);
    } else if (b.enabled) {
      ctx.fillStyle = "rgba(0,0,0,.62)"; ctx.fillRect(X, Y, W, H);
      ctx.strokeStyle = "#dc2626"; ctx.lineWidth = 1.5; ctx.strokeRect(X, Y, W, H);
    } else {
      ctx.setLineDash([5, 4]); ctx.strokeStyle = "#16a34a"; ctx.lineWidth = 1.5;
      ctx.strokeRect(X, Y, W, H); ctx.setLineDash([]);
    }
    ctx.fillStyle = b.custom ? "#1d4ed8" : (b.enabled ? "#fecaca" : "#166534");
    ctx.font = "10px sans-serif"; ctx.fillText(b.category, X + 2, Y - 3 > 10 ? Y - 3 : Y + 10);
  });
}
function drawAll() { document.querySelectorAll(".pwrap").forEach(drawPage); }

/* ---------- interaction ----------
   click on AI box   -> toggle keep/redact
   click on custom × / double-click -> remove that custom box
   ANY drag (also starting inside an existing box) -> draw a NEW custom box */
function attachEvents(w) {
  const cv = w._cv, page = +w.dataset.page;
  let drag = null, downHit = null, downDetail = 1, moved = false;
  const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const norm = (x, y) => [Math.round(x / Math.max(1, cv.width) * 1000), Math.round(y / Math.max(1, cv.height) * 1000)];

  cv.addEventListener("mousedown", e => {
    const [mx, my] = pos(e);
    drag = [mx, my]; downHit = hitTest(page, mx, my, cv); downDetail = e.detail; moved = false;
  });
  cv.addEventListener("mousemove", e => {
    if (!drag) return;
    const [mx, my] = pos(e);
    if (Math.abs(mx - drag[0]) > 4 || Math.abs(my - drag[1]) > 4) moved = true;
    if (!moved) return;
    const ctx = cv.getContext("2d"); drawPage(w);
    ctx.fillStyle = "rgba(37,99,235,.25)"; ctx.strokeStyle = "#2563eb"; ctx.lineWidth = 1;
    const X = Math.min(drag[0], mx), Y = Math.min(drag[1], my);
    ctx.fillRect(X, Y, Math.abs(mx - drag[0]), Math.abs(my - drag[1]));
    ctx.strokeRect(X, Y, Math.abs(mx - drag[0]), Math.abs(my - drag[1]));
  });
  cv.addEventListener("mouseup", e => {
    if (!drag) return;
    const [mx, my] = pos(e);
    const sx = cv.width / 1000, sy = cv.height / 1000;
    const a = norm(drag[0], drag[1]), b = norm(mx, my);
    const bbox = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
    const bigEnough = bbox[2] - bbox[0] >= 8 && bbox[3] - bbox[1] >= 6;
    const hit = downHit, detail = downDetail;
    drag = null; drawPage(w);
    if (moved && bigEnough) {                      // real drag -> new custom box
      boxes.push({ id: "c" + Date.now() + Math.random().toString(36).slice(2, 6), page, bbox, category: "manual", text: "", enabled: true, custom: true });
      saveCustom(); drawAll(); stats();
      return;
    }
    if (!hit) return;                              // click on empty page -> nothing
    if (hit.custom) {                              // × corner or double-click removes
      const [x1, y1, x2, y2] = hit.bbox, X = x1 * sx, Y = y1 * sy, W = (x2 - x1) * sx, H = (y2 - y1) * sy;
      const onX = mx >= X + W - 14 && mx <= X + W + 4 && my >= Y - 4 && my <= Y + 16;
      if (onX || detail === 2) { boxes = boxes.filter(b => b.id !== hit.id); saveCustom(); drawAll(); stats(); toast("custom box removed"); }
      else toast("custom box — click its × or double-click to remove it");
    } else { hit.enabled = !hit.enabled; drawAll(); stats(); }
  });
}
function hitTest(page, x, y, cv) {
  const sx = cv.width / 1000, sy = cv.height / 1000;
  const on = boxes.filter(b => b.page === page);
  for (let i = on.length - 1; i >= 0; i--) {
    const [x1, y1, x2, y2] = on[i].bbox;
    const X = x1 * sx, Y = y1 * sy, W = (x2 - x1) * sx, H = (y2 - y1) * sy;
    const pad = 4;
    if (x >= X - pad && x <= X + W + pad && y >= Y - pad && y <= Y + H + pad) return on[i];
  }
  return null;
}

async function fetchUsage() {
  const s = await (await fetch(`/api/status/${job.id}`)).json();
  if (s.usage) showUsage(s.usage);
}

/* ---------- new file ---------- */
$("#newBtn").onclick = () => {
  if (pollTimer) clearInterval(pollTimer);
  if (job) fetch(`/api/custom/${job.id}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ custom: [] }) }).catch(() => {});
  job = null; boxes = []; catOff = new Set();
  history.replaceState(null, "", location.pathname);
  $("#uploadCard").hidden = false;
  $("#detectCard").hidden = true;
  $("#helpCard").hidden = true;
  $("#auditCard").hidden = true;
  $("#viewer").innerHTML = "";
  $("#docInfo").textContent = "";
  $("#exportBtn").disabled = true;
  $("#cats").innerHTML = ""; $("#boxStats").textContent = "";
  $("#progBar").style.width = "0";
  $("#detectState").textContent = "…"; $("#detectState").className = "chip";
  window.scrollTo(0, 0);
};

/* ---------- zoom ---------- */
function setZoom(z) { zoom = Math.min(2.5, Math.max(.4, z)); $("#zoomLabel").textContent = Math.round(zoom * 100) + "%";
  const v = $("#viewer"); v.style.maxWidth = (zoom * 900) + "px"; layout(); }
$("#zoomIn").onclick = () => setZoom(zoom + .15);
$("#zoomOut").onclick = () => setZoom(zoom - .15);

/* ---------- export ---------- */
$("#exportBtn").onclick = async () => {
  $("#exportBtn").disabled = true; $("#exportBtn").textContent = "exporting…";
  const body = {
    enabled: boxes.filter(b => b.enabled && !b.custom).map(b => b.id),
    custom: boxes.filter(b => b.custom).map(b => ({ page: b.page, bbox: b.bbox })),
    audit: true
  };
  try {
    const r = await fetch(`/api/export/${job.id}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!j.ok) { toast("Export failed"); return; }
    showAudit(j);
    fetchUsage();  // refresh sidebar usage (detect + audit)
    toast(`Exported ${j.pages} pages · ${j.burned_boxes + j.custom_boxes} boxes burned · saved to ${j.path}`);
    const a = document.createElement("a"); a.href = `/api/download/${job.id}`; a.download = ""; a.click();
  } finally { $("#exportBtn").disabled = false; $("#exportBtn").textContent = "Export redacted PDF"; }
};
function showAudit(j) {
  const card = $("#auditCard"); card.hidden = false; const out = $("#auditOut"); out.innerHTML = "";
  const leaks = j.audit || {};
  const keys = Object.keys(leaks);
  if (j.extractable_chars !== 0) out.innerHTML = `<div class="audit-p"><b>⚠ text layer present (${j.extractable_chars} chars) — NOT safe!</b></div>`;
  if (!keys.length) out.innerHTML += `<div class="audit-p"><b>✓ CLEAN</b> — no personal data found readable on the burned pages. Output is image-only (0 extractable chars).</div>`;
  else {
    out.innerHTML += `<div class="audit-p"><b>⚠ Possible leftovers</b> — draw a box over them and re-export:</div>`;
    keys.forEach(p => {
      const d = document.createElement("div"); d.className = "audit-p";
      d.innerHTML = `<b>page ${p}</b>` + leaks[p].map(l => `<div class="leak">· [${l.kind || l.category}] ${l.text || ""}</div>`).join("");
      out.append(d);
    });
  }
}

/* ---------- misc ---------- */
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => t.hidden = true, 6000); }

/* deep link */
(async () => {
  const jid = location.hash.replace("#", "");
  if (!jid) return;
  const r = await fetch(`/api/status/${jid}`);
  if (!r.ok) return;
  const s = await r.json();
  job = { id: jid, npages: s.npages, name: "job " + jid };
  startUI();
  if (s.state === "detected") { $("#detectState").textContent = "done"; $("#detectState").className = "chip ok"; $("#exportBtn").disabled = false; }
  boxes = s.boxes; renderCats(); drawAll(); stats();
  if (s.state === "detecting") { pollTimer = setInterval(poll, 1500); }
})();
window.addEventListener("resize", layout);
setZoom(1);
