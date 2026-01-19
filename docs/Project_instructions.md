Perfetto. Qui sotto hai un **playbook super dettagliato ma KIS** per lavorare in 4 su GitHub senza intoppi (Windows + macOS).

---

# GitHub Team Workflow (KIS) — Italian Financial Challenge

## Regole del progetto (anti-caos)

1.  `notebooks/final.ipynb` **Notebook finale da consegnare** pulito e profumato

2. Ognuno lavora su:

   * `notebooks/work_<nome>.ipynb`
   * `reports/metrics/*` e `reports/notes/*`
   * `docs/canva_link.md`

3. No file pesanti inutili: salvate solo ciò che serve.

---

## Setup iniziale (OGNUNO — una volta sola)

### 1) Clona la repo privata

```bash
git clone https://github.com/MoriondoTommaso/italian-financial-challenge.git
cd italian-financial-challenge
```

### 2) Config Git (se non già fatto)

```bash
git config --global user.name "Nome Cognome"
git config --global user.email "email@..."
```

### 3) Line endings (importante per Windows/macOS)

* **Tommaso (Windows):**

```bash
git config --global core.autocrlf true
```

* **Team (macOS):**

```bash
git config --global core.autocrlf input
```

### 4) Crea il tuo branch personale e pusha

Scegli nome branch uguale al tuo nome:

* `tommaso`
* `lorenzo`
* `carla`
* `eleonora`

Esempio (Carla):

```bash
git switch -c carla
git push -u origin carla
```

---

## Routine quotidiana (OGNUNO)

### A) Prima di lavorare (sempre)

1. Vai su `main` e aggiornati:


git switch main
git pull


2. Torna sul tuo branch e allinealo a `main`:

git switch carla
git merge main


> Se ti compare un editor per il merge message: salva/chiudi e fine.

✅ Done: `git status` deve dire “working tree clean”.


### B) Lavoro (durante la sessione)

* Lavora SOLO su file “tuoi” (work notebook + notes/metrics).
* Evita di toccare `notebooks/final.ipynb`.

Controlla cosa hai cambiato:

git status

---

### C) Salvataggio (commit)

1. Aggiungi file specifici:

```bash
git add notebooks/work_carla.ipynb reports/notes/eda_takeaways.md
```

2. Commit chiaro:

```bash
git commit -m "EDA: target distribution and key takeaways"
```

> Fai commit piccoli e frequenti (meglio 3 piccoli che 1 enorme).

---

### D) Push su GitHub

```bash
git push
```

---

## Come integrare il lavoro (Pull Request)

### OGNUNO: apri una PR

1. Vai su GitHub → repo → Compare & pull request (o “New pull request”)
2. Base: `main` ← Compare: `tuo-branch`
3. Titolo chiaro, es:

   * “EDA takeaways + target plots”
   * “Transform comparison (Ridge) results”
4. Descrizione: 2 righe con cosa hai aggiunto e dove.

### TOMMASO: review e merge

Tommaso controlla:

* file modificati corretti (no `final.ipynb` da altri)
* presence output richiesto (csv in `reports/metrics`, note in `reports/notes`)
  poi fa **Merge**.

---

## Convenzioni (super importanti)

### Naming file

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

### Commit message template

* `EDA: ...`
* `Model: ...`
* `Docs: ...`
* `Chore: ...`

---

## Cosa fare se succede un problema (FAQ)

### 1) “I have local changes and git pull fails”

Usa stash:

```bash
git stash
git pull
git stash pop
```

### 2) “Merge conflict”

Prima regola: non succede quasi mai se non toccate `final.ipynb`.
Se succede:

```bash
git status
```

Apri i file in conflitto in VS Code, risolvi, poi:

```bash
git add <file>
git commit -m "Resolve merge conflict"
git push
```

### 3) “Ho committato per sbaglio final.ipynb”

Se l’hai solo modificato ma NON committato:

```bash
git restore notebooks/final.ipynb
```

Se l’hai già committato:

* scrivi a Tommaso, NON fare merge; si sistema con un revert o un reset sul branch.

### 4) “Remote origin già esiste”

Vedi remote:

```bash
git remote -v
```

Aggiorna:

```bash
git remote set-url origin <URL>
```

---

## Checklist “prima di aprire una PR”

* `git status` pulito
* hai fatto `git pull` su main oggi
* hai mergiato main nel tuo branch
* i file aggiornati sono solo quelli previsti
* hai pushato
