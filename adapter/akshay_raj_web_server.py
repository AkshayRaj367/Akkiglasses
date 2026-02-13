"""
Flask Web Server for the Auto-Training Adapter
Provides:
  - Upload dataset (zip / folder)         POST /api/upload
  - Start training                        POST /api/train
  - Live training status (poll / SSE)     GET  /api/status
  - List trained models                   GET  /api/models
  - Web dashboard                         GET  /

Run:
    python akshay_raj_web_server.py
    Then open http://localhost:5000 in a browser.
"""

import os
import sys
import json
import zipfile
import shutil
import threading
from pathlib import Path

from flask import (Flask, request, jsonify, render_template_string,
                   send_from_directory, Response)

# Make sure the adapter module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from akshay_raj_adapter import (run_training, status_tracker,
                                 hot_reloader, detect_dataset_type,
                                 UPLOAD_DIR, MODELS_DIR, BASE_DIR)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload limit

ALLOWED_ZIP_EXTS = {".zip"}

# ---- background training thread handle ----
_training_thread = None


# ===================================================================
#  API ENDPOINTS
# ===================================================================

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload a dataset (zip file).  Extracts into uploads/<name>/"""
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "No file selected"}), 400

    fname = Path(file.filename)
    if fname.suffix.lower() not in ALLOWED_ZIP_EXTS:
        return jsonify({"ok": False, "error": "Only .zip files are accepted"}), 400

    # Save & extract
    dest = UPLOAD_DIR / fname.stem
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = UPLOAD_DIR / fname.name
    file.save(str(zip_path))

    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(dest))
    except zipfile.BadZipFile:
        return jsonify({"ok": False, "error": "Uploaded file is not a valid zip"}), 400
    finally:
        zip_path.unlink(missing_ok=True)

    # Auto-detect type
    try:
        ds_type = detect_dataset_type(str(dest))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "dataset_path": str(dest),
                    "dataset_type": ds_type,
                    "message": f"Dataset uploaded & extracted ({ds_type})."})


@app.route("/api/train", methods=["POST"])
def api_train():
    """Start training in a background thread."""
    global _training_thread

    data = request.get_json(force=True, silent=True) or {}
    dataset_path = data.get("dataset_path")
    force_type   = data.get("type")
    epochs       = data.get("epochs")
    batch_size   = data.get("batch_size")
    lr           = data.get("lr")

    if not dataset_path or not Path(dataset_path).exists():
        return jsonify({"ok": False, "error": "dataset_path is missing or does not exist"}), 400

    # Don't start if already training
    current = status_tracker.get()
    if current["status"] == "training":
        return jsonify({"ok": False, "error": "Training already in progress"}), 409

    def _train():
        run_training(dataset_path, force_type=force_type,
                     epochs=int(epochs) if epochs else None,
                     batch_size=int(batch_size) if batch_size else None,
                     lr=float(lr) if lr else None)

    _training_thread = threading.Thread(target=_train, daemon=True)
    _training_thread.start()

    return jsonify({"ok": True, "message": "Training started in background."})


@app.route("/api/status")
def api_status():
    """Return current training status."""
    return jsonify(status_tracker.get())


@app.route("/api/status/stream")
def api_status_stream():
    """SSE stream – the browser EventSource connects here for live updates."""
    def generate():
        last = None
        while True:
            s = json.dumps(status_tracker.get(), default=str)
            if s != last:
                yield f"data: {s}\n\n"
                last = s
            import time
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/models")
def api_models():
    """List all trained models."""
    return jsonify(hot_reloader.get_all_status())


# ===================================================================
#  WEB DASHBOARD  (single-page, no build step)
# ===================================================================

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Smart Glasses – Auto-Training Dashboard</title>
<style>
  :root{--bg:#0f172a;--card:#1e293b;--accent:#38bdf8;--green:#22c55e;--red:#ef4444;--text:#e2e8f0;--muted:#94a3b8}
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
  .container{max-width:960px;margin:0 auto;padding:2rem 1rem}
  h1{font-size:1.8rem;margin-bottom:.3rem}
  .subtitle{color:var(--muted);margin-bottom:2rem}
  .card{background:var(--card);border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;box-shadow:0 4px 24px rgba(0,0,0,.3)}
  .card h2{font-size:1.15rem;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}
  .badge{display:inline-block;padding:2px 10px;border-radius:9999px;font-size:.75rem;font-weight:600;text-transform:uppercase}
  .badge.idle{background:#334155;color:var(--muted)}
  .badge.detecting{background:#1e40af;color:#93c5fd}
  .badge.training{background:#854d0e;color:#fde68a}
  .badge.done{background:#166534;color:#86efac}
  .badge.error{background:#7f1d1d;color:#fca5a5}

  /* upload area */
  .upload-zone{border:2px dashed #475569;border-radius:10px;padding:2.5rem;text-align:center;cursor:pointer;transition:border .2s}
  .upload-zone:hover,.upload-zone.drag{border-color:var(--accent)}
  .upload-zone input{display:none}
  .upload-zone p{color:var(--muted);margin-top:.5rem;font-size:.9rem}

  /* progress */
  .progress-bar{height:10px;border-radius:5px;background:#334155;overflow:hidden;margin:.8rem 0}
  .progress-bar .fill{height:100%;border-radius:5px;background:linear-gradient(90deg,var(--accent),var(--green));transition:width .4s}

  /* params */
  .params{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:.8rem;margin-bottom:1rem}
  .params label{font-size:.8rem;color:var(--muted);display:block;margin-bottom:2px}
  .params input,.params select{width:100%;padding:.4rem .6rem;border:1px solid #475569;border-radius:6px;background:#0f172a;color:var(--text);font-size:.9rem}

  .btn{padding:.55rem 1.4rem;border:none;border-radius:8px;font-size:.9rem;font-weight:600;cursor:pointer;transition:opacity .2s}
  .btn:disabled{opacity:.4;cursor:not-allowed}
  .btn-primary{background:var(--accent);color:#0f172a}
  .btn-danger{background:var(--red);color:#fff;margin-left:.5rem}

  /* log */
  .log{background:#0f172a;border-radius:8px;padding:1rem;max-height:220px;overflow-y:auto;font-family:monospace;font-size:.82rem;line-height:1.6;color:var(--muted)}
  .log .entry{border-bottom:1px solid #1e293b;padding:2px 0}

  /* chart */
  canvas{width:100%!important;max-height:220px}

  /* models table */
  table{width:100%;border-collapse:collapse;font-size:.85rem}
  th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid #334155}
  th{color:var(--muted);font-weight:500;text-transform:uppercase;font-size:.72rem;letter-spacing:.5px}

  /* dataset type indicator */
  .type-tag{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:6px;font-size:.78rem;font-weight:600}
  .type-tag.image{background:#312e81;color:#a5b4fc}
  .type-tag.voice{background:#4a2040;color:#f0abfc}
  .type-tag.text{background:#1a3a2a;color:#86efac}

  .flex{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
  .mt{margin-top:1rem}

  @media(max-width:600px){
    .params{grid-template-columns:1fr 1fr}
  }
</style>
</head>
<body>

<div class="container">
  <h1>&#128374; Smart Glasses – Auto-Training Adapter</h1>
  <p class="subtitle">Upload a dataset (image, voice, or text) and training starts automatically.</p>

  <!-- UPLOAD CARD -->
  <div class="card" id="uploadCard">
    <h2>&#128228; Upload Dataset</h2>
    <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="currentColor" style="color:var(--accent)"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 16V4m0 0l-4 4m4-4l4 4M4 14v4a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg>
      <p><strong>Click or drag &amp; drop</strong> a <code>.zip</code> file containing your dataset</p>
      <p style="font-size:.78rem">Folder structure: <code>dataset/class_a/files…</code> or a CSV with <code>text,label</code> columns</p>
      <input type="file" id="fileInput" accept=".zip"/>
    </div>
    <div id="uploadStatus" style="margin-top:.8rem"></div>
  </div>

  <!-- TRAINING CONFIG CARD -->
  <div class="card" id="configCard" style="display:none">
    <h2>&#9881;&#65039; Training Configuration</h2>
    <div class="flex" style="margin-bottom:1rem">
      <span>Detected type:</span>
      <span class="type-tag" id="dsTypeTag">—</span>
      <span style="color:var(--muted);font-size:.85rem" id="dsPath"></span>
    </div>
    <div class="params">
      <div><label>Epochs</label><input id="cfgEpochs" type="number" value="10" min="1"/></div>
      <div><label>Batch Size</label><input id="cfgBatch" type="number" value="16" min="1"/></div>
      <div><label>Learning Rate</label><input id="cfgLR" type="number" value="0.001" step="0.0001"/></div>
      <div><label>Force Type</label>
        <select id="cfgType"><option value="">Auto</option><option value="image">Image</option><option value="voice">Voice</option><option value="text">Text</option></select>
      </div>
    </div>
    <button class="btn btn-primary" id="btnTrain" onclick="startTraining()">&#9654; Start Training</button>
  </div>

  <!-- TRAINING PROGRESS CARD -->
  <div class="card" id="progressCard" style="display:none">
    <h2>&#128200; Training Progress <span class="badge idle" id="statusBadge">idle</span></h2>
    <p id="statusMsg" style="color:var(--muted);font-size:.9rem;margin-bottom:.5rem"></p>
    <div class="progress-bar"><div class="fill" id="progressFill" style="width:0%"></div></div>
    <div class="flex" style="font-size:.85rem;color:var(--muted)">
      <span>Epoch: <strong id="epochText">0/0</strong></span>
      <span>Loss: <strong id="lossText">—</strong></span>
      <span>Accuracy: <strong id="accText">—</strong></span>
      <span>Progress: <strong id="pctText">0%</strong></span>
    </div>

    <div class="mt">
      <canvas id="chart"></canvas>
    </div>

    <h2 class="mt" style="font-size:1rem">&#128466; Training Log</h2>
    <div class="log" id="logBox"></div>
  </div>

  <!-- MODELS CARD -->
  <div class="card">
    <h2>&#129302; Trained Models</h2>
    <table>
      <thead><tr><th>Type</th><th>Classes</th><th>Accuracy</th><th>Trained At</th></tr></thead>
      <tbody id="modelsTbody"><tr><td colspan="4" style="color:var(--muted)">No models trained yet.</td></tr></tbody>
    </table>
    <button class="btn btn-primary mt" onclick="fetchModels()" style="font-size:.8rem;padding:.35rem 1rem">&#128260; Refresh</button>
  </div>
</div>

<!-- Chart.js CDN -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>

<script>
// ---- state ----
let datasetPath = null;
let dsType = null;
let chart = null;
let sse = null;

// ---- upload ----
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

['dragenter','dragover'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag'); }));
['dragleave','drop'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag'); }));
dropZone.addEventListener('drop', ev => { if(ev.dataTransfer.files.length) uploadFile(ev.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if(fileInput.files.length) uploadFile(fileInput.files[0]); });

async function uploadFile(file){
  const s = document.getElementById('uploadStatus');
  s.innerHTML = '<span style="color:var(--accent)">Uploading… please wait</span>';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/api/upload', {method:'POST', body:fd});
    const j = await res.json();
    if(j.ok){
      datasetPath = j.dataset_path;
      dsType = j.dataset_type;
      s.innerHTML = '<span style="color:var(--green)">&#10003; ' + j.message + '</span>';
      showConfig();
    } else {
      s.innerHTML = '<span style="color:var(--red)">&#10007; ' + j.error + '</span>';
    }
  } catch(e){
    s.innerHTML = '<span style="color:var(--red)">Upload failed: '+e.message+'</span>';
  }
}

function showConfig(){
  document.getElementById('configCard').style.display = '';
  const tag = document.getElementById('dsTypeTag');
  tag.textContent = dsType;
  tag.className = 'type-tag ' + dsType;
  document.getElementById('dsPath').textContent = datasetPath;
}

// ---- training ----
async function startTraining(){
  const btn = document.getElementById('btnTrain');
  btn.disabled = true;
  const body = {
    dataset_path: datasetPath,
    type: document.getElementById('cfgType').value || null,
    epochs: document.getElementById('cfgEpochs').value,
    batch_size: document.getElementById('cfgBatch').value,
    lr: document.getElementById('cfgLR').value,
  };
  try {
    const res = await fetch('/api/train', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const j = await res.json();
    if(j.ok){
      document.getElementById('progressCard').style.display = '';
      startSSE();
    } else {
      alert(j.error);
      btn.disabled = false;
    }
  } catch(e){ alert(e.message); btn.disabled = false; }
}

// ---- SSE live updates ----
function startSSE(){
  if(sse) sse.close();
  sse = new EventSource('/api/status/stream');
  sse.onmessage = function(ev){
    const d = JSON.parse(ev.data);
    updateUI(d);
    if(d.status === 'done' || d.status === 'error'){
      sse.close();
      document.getElementById('btnTrain').disabled = false;
      fetchModels();
    }
  };
}

function updateUI(d){
  // badge
  const badge = document.getElementById('statusBadge');
  badge.textContent = d.status;
  badge.className = 'badge ' + d.status;
  // message
  document.getElementById('statusMsg').textContent = d.message || '';
  // progress
  document.getElementById('progressFill').style.width = d.progress + '%';
  document.getElementById('pctText').textContent = d.progress + '%';
  document.getElementById('epochText').textContent = d.epoch + '/' + d.total_epochs;
  document.getElementById('lossText').textContent = d.loss !== null ? d.loss : '—';
  document.getElementById('accText').textContent = d.accuracy !== null ? (d.accuracy * 100).toFixed(1) + '%' : '—';

  // log
  const box = document.getElementById('logBox');
  if(d.message && !box.innerHTML.endsWith(d.message+'</div>')){
    box.innerHTML += '<div class="entry">' + d.message + '</div>';
    box.scrollTop = box.scrollHeight;
  }

  // chart
  if(d.history && d.history.length){
    updateChart(d.history);
  }
}

function updateChart(history){
  const labels = history.map(h => 'E'+h.epoch);
  const losses = history.map(h => h.loss);
  const accs   = history.map(h => h.accuracy);

  if(!chart){
    const ctx = document.getElementById('chart').getContext('2d');
    chart = new Chart(ctx, {
      type:'line',
      data:{
        labels,
        datasets:[
          {label:'Loss', data:losses, borderColor:'#f97316', tension:.3, borderWidth:2, pointRadius:3},
          {label:'Accuracy', data:accs, borderColor:'#22c55e', tension:.3, borderWidth:2, pointRadius:3},
        ]
      },
      options:{
        responsive:true,
        plugins:{legend:{labels:{color:'#94a3b8'}}},
        scales:{
          x:{ticks:{color:'#64748b'}, grid:{color:'#1e293b'}},
          y:{ticks:{color:'#64748b'}, grid:{color:'#1e293b'}, beginAtZero:true}
        }
      }
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = losses;
    chart.data.datasets[1].data = accs;
    chart.update();
  }
}

// ---- models list ----
async function fetchModels(){
  try {
    const res = await fetch('/api/models');
    const j = await res.json();
    const tbody = document.getElementById('modelsTbody');
    let html = '';
    for(const [type, meta] of Object.entries(j)){
      if(!meta) continue;
      html += `<tr>
        <td><span class="type-tag ${type}">${type}</span></td>
        <td>${(meta.classes||[]).join(', ')}</td>
        <td>${meta.accuracy ? (meta.accuracy*100).toFixed(1)+'%' : '—'}</td>
        <td>${meta.trained_at || '—'}</td>
      </tr>`;
    }
    tbody.innerHTML = html || '<tr><td colspan="4" style="color:var(--muted)">No models trained yet.</td></tr>';
  } catch(e){ console.error(e); }
}

// initial load
fetchModels();
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ===================================================================
#  MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Glasses – Auto-Training Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
