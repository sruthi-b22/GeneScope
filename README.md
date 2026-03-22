# 🧬 GeneScope — AI-Powered Gene Analytics Dashboard

A bioinformatics web prototype built with Python and Streamlit that enables interactive exploration of human gene sequences, clinical variants, protein translation, mutation simulation, and 3D protein structure visualization.

> **Live App:** [genescope-port-al.streamlit.app](https://genescope-port-al.streamlit.app)  

---

## What GeneScope Does

GeneScope lets you search any human gene and instantly see:
- DNA sequence composition and thermodynamic properties
- Clinical variants with ClinVar pathogenicity classification
- Interactive 2D sequence viewer with per-nucleotide hover (position + base)
- Protein translation with amino acid color-coding
- Mutation simulator — substitution, deletion, insertion — with before/after comparison
- 3D protein structure viewer (PDB integration)
- BLAST-style cross-species conservation analysis
- Live NCBI search for any gene or disease not in the local database

---

## Features

### Gene Database
- 20 curated human genes with real RefSeq sequences, GO annotations, UniProt localization, and ClinVar variants
- Genes include: BRCA1, TP53, CFTR, HBB, HTT, EGFR, KRAS, APP, PTEN, RB1, and more
- Each gene has PDB structure ID for 3D visualization

### Sequence Analysis
- GC content (%) with stability interpretation
- Molecular weight (Da)
- Wallace rule Tm and Empirical Tm
- Nucleotide composition donut chart
- GC stability heatmap across sequence segments

### 2D Sequence Viewer
- Color-coded nucleotides: A = green, T = red, G = blue, C = amber
- Hover over any base to see position and identity
- Zoom slider for font size control
- Works for both DNA and translated protein sequences

### Mutation Simulator
- **Substitution** — replace a single base
- **Deletion** — remove a base (causes frameshift)
- **Insertion** — add a base before a position (causes frameshift)
- Before/after DNA side by side with mutated position highlighted in yellow
- Before/after protein with amino acid color-coding
- Automatic mutation classification: Silent / Missense / Frameshift
- 3D original structure vs 2D mutated sequence comparison

### Live NCBI Search
- Search any gene symbol, disease name, or condition
- Fetches full gene profile from NCBI in real time
- Displays RefSeq sequence, full description, chromosome location, aliases
- Renders complete GC analysis and 2D sequence viewer for fetched genes
- Falls back gracefully with descriptive error messages

### 3D Protein Viewer
- Renders PDB structures via py3Dmol
- Spectrum cartoon coloring (blue = N-terminus, red = C-terminus)
- Hover labels showing residue name and number
- Scroll to zoom, drag to rotate, right-click to pan

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Streamlit |
| Language | Python 3.10+ |
| Live Gene Search | NCBI Entrez API (Biopython) |
| 3D Visualization | py3Dmol |
| Charts | Plotly Express + Graph Objects |
| Data | Pandas, NumPy |
| Deployment | Streamlit Community Cloud |
| Version Control | Git + GitHub |
| AI Tools Used | Grok (scaffold), Claude/Anthropic (features + UI) |

---

## Project Structure
```
genescope/
├── app.py           # Main Streamlit app — all UI, analysis, visualization
├── genes.py         # Curated gene database (20 human genes)
├── requirements.txt # Python dependencies
└── README.md
```

---

## Setup & Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/genescope.git
cd genescope

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Requirements
```
streamlit
py3Dmol
pandas
numpy
plotly
biopython
ipython_genutils
```

---

## How It Works

### Search Flow
1. User types a gene name or disease in the search box
2. App checks local `genes.py` for a match — instant response
3. If not found → live NCBI Entrez query fetches full gene profile
4. Full profile rendered: sequence, metrics, 2D viewer, translation

### Mutation Flow
1. Select mutation type (substitution / deletion / insertion)
2. Enter position and new base
3. App applies mutation to the sequence
4. Before/after DNA shown side by side with highlighted position
5. Sequences translated → before/after protein compared
6. Mutation classified automatically (Silent / Missense / Frameshift)

---

## AI Usage

This prototype was developed using AI-assisted code generation:

| Tool | Used For |
|---|---|
| **Grok (xAI)** | Initial app scaffold, genes.py structure |
| **Claude (Anthropic)** | NCBI integration, 2D viewer, mutation simulator, UI uniformity, bug fixes |

---

## References

- NCBI Gene Database — https://www.ncbi.nlm.nih.gov/gene
- RCSB Protein Data Bank — https://www.rcsb.org
- UniProt Knowledgebase — https://www.uniprot.org
- ClinVar — https://www.ncbi.nlm.nih.gov/clinvar
- Biopython Entrez — https://biopython.org
- Streamlit Docs — https://docs.streamlit.io
- py3Dmol — https://3dmol.csb.pitt.edu
