# Jenkins for ML — minimal CI/CD demo

An end-to-end ML pipeline run by Jenkins: **generate synthetic data → validate →
train → evaluate → quality gate → archive artifacts**. Jenkins runs in Docker so
the shell steps work the same on Windows, macOS, or Linux.

## Concepts covered (for the demo talk)

| Jenkins concept            | Where it shows up                          |
|----------------------------|--------------------------------------------|
| Declarative pipeline       | `Jenkinsfile`                              |
| Stages = ML lifecycle      | Checkout → Setup → Data → Train → Evaluate |
| Build parameters           | `THRESHOLD`, `CLASS_SEP`                    |
| Reproducible agent env     | venv + pinned `requirements.txt`           |
| **Data validation gate**   | `Validate Data` stage (non-zero exit stops build) |
| **Model quality gate**     | `Evaluate` stage fails build if accuracy < threshold |
| Artifact archiving         | `post { archiveArtifacts ... }`            |
| Post actions / cleanup     | `post { success/failure/cleanup }`         |
| Triggers (poll / webhook)  | commented `triggers` block in `Jenkinsfile`|

## 1. Start Jenkins

Requires Docker (Docker Desktop on Win/Mac, Docker Engine on Linux).

```bash
docker compose up -d --build
```

Get the first-run admin password:

```bash
docker exec jenkins-mlops cat /var/jenkins_home/secrets/initialAdminPassword
```

Open http://localhost:8080 → paste password → **Install suggested plugins** →
create an admin user.

## 2. Put this repo in Git

Jenkins pulls the pipeline from SCM. Push to GitHub (or any Git remote):

```bash
git init
git add .
git commit -m "Jenkins MLOps demo"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## 3. Create the pipeline job

1. **New Item** → name it `ml-pipeline` → **Pipeline** → OK
2. Under **Pipeline**, set **Definition = Pipeline script from SCM**
3. **SCM = Git**, Repository URL = your repo, Branch = `*/main`
4. **Script Path = `Jenkinsfile`** → Save

## 4. Run it

**Build with Parameters** → keep defaults → **Build**. Watch the stage view go
green. Open the build → **metrics.json**, **confusion_matrix.png**, and
**model.pkl** are under archived artifacts.

## 5. Demo the gates (the interesting part)

- **Model quality gate fails:** run with `THRESHOLD = 0.99` (unrealistically
  high) → build goes **red at `Evaluate & Quality Gate`**. Or set
  `CLASS_SEP = 0.3` to make the data hard so accuracy genuinely drops below 0.85.
- **Back to green:** `THRESHOLD = 0.85`, `CLASS_SEP = 1.0`.
- **Data gate fails:** temporarily set `--min-rows 100000` in `validate_data.py`,
  commit, rebuild → build stops at `Validate Data` before training.

## 6. (Optional) Automatic builds

Uncomment the `triggers { pollSCM('H/5 * * * *') }` line in the `Jenkinsfile` to
poll Git every ~5 minutes, or add a webhook in your Git host for push-based
builds. Now a `git push` runs the whole pipeline unattended — the actual point of
CI for ML.

## Run the scripts without Jenkins (sanity check)

```bash
pip install -r requirements.txt
python3 src/generate_data.py
python3 src/validate_data.py
python3 src/train.py
python3 src/evaluate.py --threshold 0.85
```

## Teardown

```bash
docker compose down          # keep Jenkins data
docker compose down -v       # also delete the jenkins_home volume
```
#checking ci/cd for now
