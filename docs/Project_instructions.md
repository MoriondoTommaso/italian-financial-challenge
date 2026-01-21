
---

## `docs/Project_instructions.md` (versione ottimizzata, pronta da sostituire)

````md
# Project Playbook (KIS) — Italian Financial Challenge (Task 3)
Workflow Git + venv + regole operative per lavorare in team senza conflitti.

**Repo owner:** Tommaso Moriondo  
**Team:** Tommaso, Lorenzo, Eleonora, Carla  
**Notebook di consegna:** `notebooks/final.ipynb`  
**Slides:** Canva → export `slides/final.pdf` (versionato)

---

## 0) Regole d’oro (anti-caos)
1) **Non toccare** `notebooks/final.ipynb` se non per aggiungere versioni definitive.
2) Ognuno lavora su:
   - `notebooks/work_<nome>.ipynb`
   - `reports/notes/*` e `reports/metrics/*`
   - `docs/canva_link.md`
3) **Commit piccoli e frequenti.** Meglio 3 piccoli che 1 enorme.
4) Non committare file inutili/pesanti (cache, output non richiesti, artefatti locali).

---

## 1) ML Guardrails (anti-leakage) — da rispettare sempre
Queste regole **non si negoziano** (vedi anche `docs/decisions.md`):
- Split **time-aware**: Train = 2019–2020, Validation = 2021; **2018 solo history per lag**.
- Qualunque trasformazione (imputation/encoding/scaling/clipping) è **fit solo sul train** e poi applicata a val/test.
- Lag features: create solo dal passato con `groupby(company_id).shift(1)` dopo sorting per `fiscal_year`.
- Implementazione preferita: **sklearn Pipeline** per garantire fit/apply corretto.

---

## 2) Setup iniziale (OGNUNO — una volta sola)

### 2.1 Clona la repo
```bash
git clone https://github.com/MoriondoTommaso/italian-financial-challenge.git
cd italian-financial-challenge
````

### 2.2 Config Git (se non già fatto)

```bash
git config --global user.name "Nome Cognome"
git config --global user.email "email@..."
```

### 2.3 Line endings (Windows/macOS)

* **Windows**

```bash
git config --global core.autocrlf true
```

* **macOS**

```bash
git config --global core.autocrlf input
```

### 2.4 Crea il tuo branch personale e pusha

Branch naming:

* `tommaso`, `lorenzo`, `carla`, `eleonora`

Esempio:

```bash
git switch -c carla
git push -u origin carla
```

---

## 3) Routine quotidiana (OGNUNO)

### 3.1 Prima di lavorare (sempre)

1. Aggiorna `main`:

```bash
git switch main
git pull
```

2. Torna sul tuo branch e allinealo a `main`:

```bash
git switch carla
git merge main
```

✅ Check:

```bash
git status
```

deve dire “working tree clean” (o comunque nessun conflitto aperto).

---

### 3.2 Durante la sessione

* Lavora SOLO su file “tuoi” (work notebook + notes/metrics).
* Evita di toccare `notebooks/final.ipynb`.

Controllo rapido:

```bash
git status
```

---

### 3.3 Commit (salvataggio)

1. Aggiungi file specifici:

```bash
git add notebooks/work_carla.ipynb reports/notes/eda_takeaways.md
```

2. Commit chiaro:

```bash
git commit -m "EDA: target distribution and key takeaways"
```

---

### 3.4 Push

```bash
git push
```

---

## 4) Pull Request (integrazione)

### OGNUNO: apri una PR

1. GitHub → repo → “New pull request”
2. Base: `main` ← Compare: `tuo-branch`
3. Titolo chiaro (es. “EDA takeaways + target plots”)
4. Descrizione: 2 righe con cosa hai aggiunto e dove (path).

### TOMMASO: review e merge

Checklist per il merge:

* file modificati corretti (no `final.ipynb` da altri)
* output richiesto presente:

  * note in `reports/notes`
  * metriche in `reports/metrics`

---

## 5) Convenzioni (naming + commit)

### 5.1 Naming file

* Notebook personali:

  * `notebooks/work_tommaso.ipynb`
  * `notebooks/work_lorenzo.ipynb`
  * `notebooks/work_carla.ipynb`
  * `notebooks/work_eleonora.ipynb`
* Note:

  * `reports/notes/eda_takeaways.md`
  * `reports/notes/transform_decision.md`
  * `reports/notes/slide_outline.md`
* Metriche:

  * `reports/metrics/transform_compare.csv`
  * `reports/metrics/models_compare.csv`

### 5.2 Commit message template

* `EDA: ...`
* `Model: ...`
* `Docs: ...`
* `Chore: ...`

---

## 6) Virtual Environment (KIS) — Setup per tutti

### Regole

* Un venv per repo nella cartella `.venv/`
* Non committare `.venv/` (deve stare in `.gitignore`)
* Installare dipendenze da `requirements.txt`

### Windows (Git Bash / VS Code)

```bash
cd /c/Users/morio/OneDrive/Desktop/Projects/italian-financial-challenge
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

VS Code:

* `Ctrl+Shift+P` → Python: Select Interpreter → `./.venv/Scripts/python.exe`
* In notebook: seleziona Kernel = `.venv`

### macOS (Terminal / VS Code)

```bash
cd ~/path/to/italian-financial-challenge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

VS Code:

* `Cmd+Shift+P` → Python: Select Interpreter → `./.venv/bin/python`
* In notebook: Kernel = `.venv`

### Verifica rapida (tutti)

```bash
python -c "import pandas, sklearn; print('OK')"
```

---

## 7) Quando aggiungete nuove librerie

1. Install nel venv:

```bash
pip install <package>
```

2. Aggiorna `requirements.txt` (solo quando siete sicuri, per non fare rumore):

```bash
pip freeze > requirements.txt
```

3. Commit del `requirements.txt`:

```bash
git add requirements.txt
git commit -m "Chore: update requirements"
git push
```

---

## 8) FAQ rapida

### 8.1 “I have local changes and git pull fails”

```bash
git stash
git pull
git stash pop
```

### 8.2 Merge conflict

```bash
git status
```

Risolvi in VS Code → poi:

```bash
git add <file>
git commit -m "Chore: resolve merge conflict"
git push
```

### 8.3 “Ho committato per sbaglio final.ipynb”

* Se NON committato:

```bash
git restore notebooks/final.ipynb
```

* Se già committato:
  scrivi a Tommaso, **non fare merge** (si risolve con revert/reset sul branch).

---

## 9) Checklist prima di aprire una PR

* `git status` pulito
* oggi hai fatto `git pull` su `main`
* hai mergiato `main` nel tuo branch
* hai toccato solo file previsti
* hai pushato

```