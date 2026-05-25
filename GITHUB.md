# GitHub + Colab Workflow

Use GitHub so code changes on your Mac appear in Colab with `git pull` — no re-zipping.

**Data stays on Google Drive** (~442 MB mammo zip). GitHub holds **code only**.

---

## Part 1 — One-time setup on Mac

### 1. Create a GitHub repository

1. Go to [github.com/new](https://github.com/new)
2. **Repository name:** `mbc` (or `hybrid-breast-cancer-qml`)
3. **Private** recommended (research code)
4. Do **not** add README, .gitignore, or license (you already have them)
5. Click **Create repository**

### 2. Initialize and push from your Mac

Open Terminal:

```bash
cd "/Users/muthu/ResTest/paper1/mbc"

# First-time git setup
git init
git add .
git status   # should NOT list .venv, data/processed, *.pt

git commit -m "Initial commit: hybrid quantum breast cancer framework"

# Replace YOUR_USER and REPO with your GitHub username and repo name
git branch -M main
git remote add origin https://github.com/YOUR_USER/mbc.git
git push -u origin main
```

GitHub will ask you to sign in. Use a **Personal Access Token (PAT)** as the password:

1. GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic)**
2. **Generate new token** → enable **`repo`**
3. Copy token → paste when `git push` asks for password

---

## Part 2 — Colab (first time)

### Cell 1 — Clone from GitHub

```python
GITHUB_USER = 'YOUR_USER'   # ← edit
REPO = 'mbc'                # ← edit
TOKEN = ''                  # ← paste PAT (or leave empty for public repo)

import os
if TOKEN:
    url = f'https://{TOKEN}@github.com/{GITHUB_USER}/{REPO}.git'
else:
    url = f'https://github.com/{GITHUB_USER}/{REPO}.git'

if os.path.isdir('/content/mbc'):
    !rm -rf /content/mbc

!git clone {url} /content/mbc
%cd /content/mbc
!git log -1 --oneline
```

### Cell 2 — Install dependencies

```python
!pip install -q torch torchvision pennylane pydicom requests pandas tqdm shap pyyaml pillow scikit-learn matplotlib seaborn
```

**Runtime → Change runtime type → GPU**

### Cell 3 — Data from Drive (unchanged)

```python
from google.colab import drive
import os

drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/mbc_colab'

if not os.path.isdir('data/processed/mammo'):
    !unzip -q -o "{DRIVE_ROOT}/mbc_mammo.zip" -d /content/
    !mkdir -p data/processed
    !ln -sf /content/data/processed/mammo data/processed/mammo

!python data/download/setup_datasets.py
```

### Cell 4 — Train

```python
!python experiments/run_training.py --experiment classical --modality mammo --quick
# then full hybrid:
# !python experiments/run_training.py --experiment hybrid --modality mammo
```

### Cell 5 — Save results to Drive

```python
!cp -r results checkpoints /content/drive/MyDrive/mbc_colab/ 2>/dev/null
```

---

## Part 3 — Daily workflow (after code changes on Mac)

### On Mac

```bash
cd "/Users/muthu/ResTest/paper1/mbc"
git add .
git commit -m "Describe your change"
git push
```

### On Colab (new session or before training)

```python
%cd /content/mbc
!git pull
!git log -1 --oneline   # confirm latest commit
```

Then re-run install (if `requirements.txt` changed) and training.

---

## What goes where

| Item | GitHub | Google Drive |
|------|--------|--------------|
| Python source (`src/`, `experiments/`) | ✅ | ❌ |
| Configs, scripts, paper | ✅ | ❌ |
| `mbc_mammo.zip` / images | ❌ | ✅ |
| Checkpoints, `results/` | ❌ | ✅ |
| `.venv`, `data/processed/` | ❌ (gitignored) | optional |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `git push` rejected | `git pull --rebase origin main` then push again |
| Colab clone asks for password | Use PAT in clone URL (Cell 1) |
| `git pull` conflicts | `git stash` or re-clone fresh |
| Large file push fails | File >100 MB — add to `.gitignore`, use Drive |
| Token expired | Generate new PAT on GitHub |

---

## Optional — SSH instead of token

On Mac:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub   # add to GitHub → Settings → SSH keys
git remote set-url origin git@github.com:YOUR_USER/mbc.git
```

Colab SSH is harder; **HTTPS + PAT** is simpler for Colab.

---

## Security note

- Never commit `.env`, tokens, or API keys
- Keep the repo **private** if unpublished
- Do not commit `data/processed/` (patient images)
