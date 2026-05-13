# Threat Intelligence Risk Prioritizer

ML-driven CVE × Asset risk prioritization using NVD, MITRE ATT&CK, and CodeBERT semantic embeddings.

Each CVE from the NVD feed is scored against every asset in your organization. The score combines CVSS metrics, asset criticality, internet exposure, MITRE ATT&CK technique coverage, and semantic embeddings from `microsoft/codebert-base` — producing a ranked list of what to patch first.

---

## Features

- Ingest CVEs from the **NVD JSON bulk feed** (downloaded file) or the **live NVD API**
- Load **MITRE ATT&CK Enterprise** techniques for attacker-behavior context (auto-cached)
- Encode CVE descriptions with **CodeBERT** (`microsoft/codebert-base`) → 768-dim → PCA 32-dim semantic features
- Score each **CVE × Asset pair** with a **Gradient Boosting** model (scikit-learn)
- Auto-detects and uses **GPU (CUDA)** for CodeBERT if available — falls back to CPU
- Streamlit dashboard with 5 tabs: Overview · Risk Table · CVE Explorer · Asset View · Model Insights
- Export results as CSV

---

## Project Layout

```
threat-intel-prioritizer/
├── app/
│   └── streamlit_app.py              # Streamlit UI — entry point
├── src/
│   ├── schema.py                     # Pydantic types: CVERecord, AssetRecord, RiskRecord
│   ├── ingestion/
│   │   ├── nvd_fetcher.py            # NVD API client + NVD JSON feed parser
│   │   ├── mitre_loader.py           # MITRE ATT&CK STIX loader (cached to disk)
│   │   └── asset_loader.py           # Asset registry (JSON file or built-in samples)
│   ├── features/
│   │   ├── feature_builder.py        # CVE × Asset feature matrix builder
│   │   └── vulberta_embedder.py      # CodeBERT encoder (PCA-reduced, disk-cached)
│   ├── models/
│   │   ├── risk_scorer.py            # GradientBoostingRegressor wrapper
│   │   └── train.py                  # Standalone training script (synthetic data)
│   └── prioritization/
│       └── engine.py                 # Full pipeline orchestrator
├── data/
│   └── samples/
│       ├── assets_example.json       # Minimal asset format example
│       └── realistic_assets.json     # 25 real-world assets (Apache, MySQL, Cisco, etc.)
├── tests/
│   ├── test_features.py
│   └── test_risk_scorer.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python **3.11** or newer
- `pip` (comes with Python)
- Internet access on first run (downloads CodeBERT ~500 MB and MITRE ATT&CK ~10 MB, both cached afterwards)

---

### Option A — CPU only (simplest)

```bash
# 1. Clone or download the project
cd threat-intel-prioritizer

# 2. (Recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

### Option B — GPU (CUDA) — recommended for large feeds

Using a GPU makes CodeBERT encoding **~20× faster** (minutes instead of hours on large feeds).

**Step 1 — Check your CUDA version**

```bash
nvidia-smi
```

Look for `CUDA Version: XX.X` in the top-right corner.

**Step 2 — Install PyTorch with the matching CUDA build**

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 / 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Step 3 — Install the rest of the dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Verify GPU is detected**

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output: `CUDA: True | NVIDIA GeForce RTX XXXX`

**Step 5 — Launch the dashboard**

```bash
streamlit run app/streamlit_app.py
```

The GPU/CPU status is shown live in the sidebar under the CodeBERT toggle.

---

### Option C — Anaconda / conda environment

If you already have an Anaconda environment (e.g. `gpu-env`):

```bash
conda activate gpu-env

# Install PyTorch for your CUDA version first (see Option B Step 2)
# Then install the rest:
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

---

## Default Usage (Recommended Workflow)

This is the standard way to run the project with real data.

### Step 1 — Get the NVD JSON feed

Download the full CVE dataset from NVD:

1. Go to https://nvd.nist.gov/vuln/data-feeds
2. Download **nvdcve-1.1-recent.json.gz** (recent CVEs) or any year file
3. Extract the `.json` file and place it in the project folder (e.g. `data/nvd_feed.json`)

Alternatively use the NVD 2.0 API export — any JSON file with a `"vulnerabilities"` array works.

---

### Step 2 — Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

### Step 3 — Configure the sidebar

#### CodeBERT
- Leave **"Enable CodeBERT embeddings"** toggled **ON**
- The sidebar shows your device: green = GPU, blue = CPU
- CodeBERT downloads once (~500 MB) and caches to `data/vulberta_emb_cache.pkl`

#### CVE Data Source
- Select **"Upload NVD JSON feed"**
- Click **Browse files** and upload your downloaded NVD JSON file

#### Performance — Max CVEs to process
- Default is **500** — good for a first run (~30 sec on CPU, ~15 sec on GPU)
- Increase to 2500–5000 once you confirm it works
- Full 23,845 CVEs: ~10 min on GPU, hours on CPU

#### Assets
- Select **"Upload assets JSON"**
- Upload `data/samples/realistic_assets.json` (included in the project)
- This file contains 25 real-world assets: Apache, MySQL, PostgreSQL, Windows Server, Cisco routers, Palo Alto firewall, WordPress, Confluence, Exchange, OpenVPN, and more

#### MITRE ATT&CK
- Leave checked — downloads once (~10 MB) and caches to `data/mitre_enterprise.json`

---

### Step 4 — Run the pipeline

Click **▶ Run Pipeline**.

The terminal shows live progress:
```
INFO Loading CVEs from uploaded feed: data/nvd_feed.json
INFO Loaded 23845 CVE records from feed
INFO Capping CVEs: 23845 → 500 (max_cves limit)
INFO Loaded 25 assets from data/samples/realistic_assets.json
INFO Building features: 500 CVEs × 25 assets
INFO Feature build progress: 0 / 500 CVEs
INFO Feature build progress: 50 / 500 CVEs
...
INFO Encoding 500 unique CVE descriptions with CodeBERT…
INFO RiskScorer trained on 12500 samples with 41 features
INFO Pipeline complete.
```

---

### Step 5 — Explore results

| Tab | What you see |
|---|---|
| **Overview** | Priority pie chart, score histogram, top-20 risk bar chart |
| **Risk Table** | Full sortable table, filterable by priority, downloadable as CSV |
| **CVE Explorer** | Select any CVE → see risk score across all your assets |
| **Asset View** | Which of your assets has the highest maximum risk |
| **Model Insights** | Feature importance (CodeBERT vs tabular), device (GPU/CPU), score scatter |

---

## Priority Thresholds

| Risk Score | Priority | Recommended Action |
|---|---|---|
| ≥ 0.75 | **CRITICAL** | Patch immediately. Isolate asset if patch unavailable. |
| ≥ 0.50 | **HIGH** | Patch within 24–72 hours. Apply compensating controls. |
| ≥ 0.30 | **MEDIUM** | Include in next patch cycle (within 30 days). Monitor for PoC. |
| < 0.30 | **LOW** | Track and patch in routine maintenance window. |

---

## Assets JSON Format

The assets file is a JSON array. Each object requires these fields:

```json
[
  {
    "asset_id": "WEB-001",
    "name": "Apache Web Server",
    "asset_type": "server",
    "criticality": 9.0,
    "exposure": 1.0,
    "cpe": "cpe:2.3:a:apache:http_server:2.4.54:*:*:*:*:*:*:*"
  }
]
```

| Field | Type | Description |
|---|---|---|
| `asset_id` | string | Unique identifier |
| `name` | string | Human-readable name |
| `asset_type` | string | `server`, `workstation`, `network_device`, `cloud` |
| `criticality` | float 0–10 | How damaging a compromise would be |
| `exposure` | float 0–1 | `1.0` = internet-facing, `0.1` = internal only |
| `cpe` | string | CPE identifier for CVE matching (optional, leave `""` if unknown) |

Look up CPE strings at https://nvd.nist.gov/products/cpe/search

---

## How the Risk Score Works

Each CVE × Asset pair is scored by a **Gradient Boosting Regressor** trained on these features:

| Feature | Source |
|---|---|
| `cvss_score` | NVD CVSS base score |
| `exploitability` | NVD exploitability sub-score |
| `impact_score` | NVD impact sub-score |
| `asset_criticality` | Your asset data |
| `asset_exposure` | Your asset data |
| `exposure × exploitability` | Derived interaction |
| `criticality × impact` | Derived interaction |
| `high_risk_cwe` | CWE-78/79/89/119/120/22/434 flag |
| `mitre_coverage` | Overlap with MITRE ATT&CK technique descriptions |
| `emb_0 … emb_31` | CodeBERT semantic embedding (32 PCA dims from 768) |

The model is trained on the ingested data itself using a heuristic label (weighted CVSS + asset factors). No external labelled dataset is required.

---

## Live NVD API Mode

As an alternative to uploading a feed file, the live API fetches recent CVEs directly:

1. Select **"Live NVD API"** in the sidebar
2. Set the date range (7–90 days back)
3. Optionally enter keywords (each fetched separately — blank = all recent CVEs)
4. Optionally enter a free **NVD API key** (from https://nvd.nist.gov/developers/request-an-api-key) to remove rate limits

Without an API key, the API allows ~1.5 requests/second. With a key: 6 requests/second.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Tech Stack

| Component | Library |
|---|---|
| Web UI | Streamlit + Plotly |
| ML model | scikit-learn GradientBoostingRegressor |
| Text encoder | Hugging Face Transformers — `microsoft/codebert-base` |
| CVE data | NVD via `nvdlib` + NVD JSON bulk feed |
| Threat intel | MITRE ATT&CK STIX via GitHub |
| Schemas | Pydantic v2 |
| Python | 3.11+ |
