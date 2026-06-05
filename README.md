# End-to-End MLOps Pipeline
**MLOps | PGD AI Program | IIT Jodhpur**

> SMS Spam Classification using **DistilBERT** (`distilbert-base-uncased`), fine-tuned on Kaggle with three hyperparameter versions, tracked with W&B, served via Docker (GHCR), and automated through GitHub Actions.

---

##  Live Links

| Resource | URL |
|---|---|
|  GitHub Repository | [`https://github.com/g25ait2032-prog/MLOPS_Group`](https://github.com/g25ait2032-prog/MLOPS_Group) |
|  HF Model — v1 | [`https://huggingface.co/nagaananth/MLOPS_group-v1`](https://huggingface.co/nagaananth/MLOPS_group-v1) |
|  HF Model — v2 ★ Best | [`https://huggingface.co/nagaananth/MLOPS_group-v2`](https://huggingface.co/nagaananth/MLOPS_group-v2) |
|  HF Model — v3 | [`https://huggingface.co/nagaananth/MLOPS_group-v3`](https://huggingface.co/nagaananth/MLOPS_group-v3) |
|  W&B Project Dashboard | `https://wandb.ai/g25ait2032-iit-jodhpur/MLOPS_Group` |
|  Docker Image (GHCR) | `ghcr.io/g25ait2032-prog/mlops_group-inference:latest` |
|  Kaggle Notebook (v1) | `https://www.kaggle.com/code/your-username/sms-spam-v1` |
|  Kaggle Notebook (v2) | `https://www.kaggle.com/code/your-username/sms-spam-v2` |
|  Kaggle Notebook (v3) | `https://www.kaggle.com/code/your-username/sms-spam-v3` |

>  Replace placeholder Kaggle notebook URLs and member roll numbers before submission.

---

##  Pipeline Status

![CI](https://github.com/g25ait2032-prog/MLOPS_Group/actions/workflows/ci.yml/badge.svg)
![Inference Pipeline](https://github.com/g25ait2032-prog/MLOPS_Group/actions/workflows/inference.yml/badge.svg)

---

##  Group Members

| Name | Roll Number | Contributions |
|---|---|---|
| Anukumar K | G25AIT2016 | Data preparation, model training (v1, v2, v3), and W&B experiment tracking. |
| Duggirala Vnaga Ananth | G25AIT2032 | Containerization (Docker), GitHub Actions CI/CD, and inference workflows. |
| Shrikrishna Tripathi | G25AIT2103 | Documentation (README/Report), code review, and evaluation diagnostics. |
| Sudeb Ghosh | G25AIT2113 | Experiment analysis, version documentation, and final reporting. |


---

##  Project Structure

```
MLOPS_Group/
├── src/
│   ├── prepare_data.py      # Task 2 — Data cleaning & normalisation
│   ├── inference.py         # Task 6 & 7 — Inference script (HF Hub → prediction)
│   └── evaluate.py          # Optional — local evaluation helper
├── notebooks/
│   └── mlops-gr-assignment.ipynb   # Tasks 3, 4, 5 — Kaggle training notebook (v1/v2/v3)
├── .github/
│   └── workflows/
│       ├── ci.yml           # Task 7.1 — CI linting on push to develop
│       └── inference.yml    # Task 7.2 — Manual inference trigger
├── data/
│   └── id2label.json        # Label mapping (only file committed — no large data)
├── Dockerfile               # Task 6 — Inference container
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Model

**`distilbert-base-uncased`** — a distilled version of BERT that retains 97 % of language understanding at 40 % smaller size (~66 M parameters, ~265 MB). Chosen because it:

- Fits within Kaggle free-tier GPU quota (T4 x2)
- Is well-suited to short, noisy SMS text
- Has a clear, well-documented HF model card

---

##  Dataset

**UCI SMS Spam Collection** loaded via `datasets` (`sms_spam`).

| Split | Samples | Ham % | Spam % |
|---|---|---|---|
| Train (70 %) | ~3,611 | 86.6 | 13.4 |
| Validation (15 %) | ~774 | 86.6 | 13.4 |
| Test (15 %) | ~774 | 86.6 | 13.4 |

**Cleaning steps:** lowercased → whitespace normalised → deduplicated (415 removed) → zero-leakage stratified split.

---

##  Experiment Comparison

| Hyperparameter | v1 ★| v2  | v3 |
|---|---|---|---|
| Learning Rate | 3e-5 | 2e-5 | 3e-5 |
| Epochs | 3 | 5 | 4 |
| Batch Size | 16 | 32 | 16 |
| Warmup Steps | 100 | 200 | 100 |
| Padding | dynamic | max_length | max_length |
| Early Stopping | No | Yes (p=2) | Yes (p=2) |
| fp16 | Yes | Yes | Yes  |

### Training Configuration and Performance

Three fine-tuning experiments were conducted using different hyperparameter configurations. Versions **v2** and **v3** achieved the lowest validation loss while maintaining the same high classification performance (**99.35% accuracy**). Among all runs, **v2** produced the best validation loss (**0.0292**), making it the selected model for final evaluation.

| Version | Learning Rate | Epochs | Batch Size | Accuracy | F1 Weighted | F1 Macro | Precision | Recall | Val Loss |
| ------- | ------------- | ------ | ---------- | -------- | ----------- | -------- | --------- | ------ | -------- |
| v1      | 3e-5          | 3      | 16         | 0.9935   | 0.9935      | 0.9849   | 0.9935    | 0.9935 | 0.0539   |
| v2      | 2e-5          | 5      | 32         | 0.9935   | 0.9935      | 0.9851   | 0.9935    | 0.9935 | 0.0292   |
| v3      | 2e-5          | 5      | 32         | 0.9935   | 0.9935      | 0.9851   | 0.9935    | 0.9935 | 0.0376   |


> Fill in Accuracy, F1, and Val Loss after running both Kaggle experiments. See [W&B dashboard](https://wandb.ai/g25ait2032-iit-jodhpur/MLOPS_Group) for live charts.

---

##  Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/g25ait2032-prog/MLOPS_Group.git
cd MLOPS_Group

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

##  Running Each Step

### Step 1 — Prepare Data (local)
```bash
python src/prepare_data.py
# Outputs: data/train.csv, data/validation.csv, data/test.csv,
#          data/id2label.json, data/label2id.json
```

### Step 2 — Train on Kaggle
1. Upload `notebooks/mlops-gr-assignment.ipynb` into a new Kaggle Notebook.
2. Upload `data/train.csv`, `data/test.csv`, `data/id2label.json` as a Kaggle Dataset.
3. Enable GPU: **Settings → Accelerator → GPU T4 x2**.
4. Add Kaggle Secrets: `WANDB_API_KEY_gr`, `HF_TOKEN_gr`.
5. Set `VERSION = "v1"` in Cell 4 and run. Repeat for `"v2"` and `"v3"`.

### Step 3 — Run Inference Locally
```bash
export HF_TOKEN=your_token
export HF_MODEL=nagaananth/MLOPS_group-v2
export INPUT_TEXT="Free prize! Click now to claim."
python src/inference.py
```

### Step 4 — Docker
```bash
# Build
docker build --build-arg HF_MODEL_NAME=nagaananth/MLOPS_group-v2 \
             -t mlops_group-inference:latest .

# Run locally
docker run --rm \
  -e HF_TOKEN=your_token \
  -e INPUT_TEXT="Win a free car now!" \
  mlops_group-inference:latest

# Pull from GHCR (no build needed)
docker pull ghcr.io/g25ait2032-prog/mlops_group-inference:latest
docker run --rm \
  -e HF_TOKEN=your_token \
  -e INPUT_TEXT="Win a free car now!" \
  ghcr.io/g25ait2032-prog/mlops_group-inference:latest
```

### Step 5 — GitHub Actions Inference (manual)
1. Go to **Actions → Inference → Run workflow**.
2. Enter the text you want to classify.
3. Check the run logs for the predicted label and confidence score.

---

##  GitHub Secrets Required

| Secret | Description |
|---|---|
| `HF_TOKEN` | Hugging Face write token |
| `HF_MODEL` | HF repo ID (e.g. `nagaananth/MLOPS_group-v2`) |
| `WANDB_API_KEY` | W&B API key |

Add via: **Settings → Secrets and Variables → Actions → New repository secret**

---

##  Notes

- Training is done **only on Kaggle** — GitHub Actions handles CI (linting) and inference only.
- Large data files (`train.csv`, `test.csv`) are excluded via `.gitignore`; only `id2label.json` is committed.
- Three experiment versions (v1, v2, v3) are visible in the W&B dashboard — set project visibility to **Public** before submission.
- All links must be publicly accessible at submission time.
