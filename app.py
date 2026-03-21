import subprocess
import sys
from collections import Counter

# Force install the missing "adapter" if it's not found
try:
    import ipython_genutils
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ipython_genutils"])

import streamlit as st
import streamlit.components.v1 as components
import py3Dmol

import pandas as pd
from genes import GENE_DB

try:
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # Plotly not available in this environment
    np = None
    px = None
    go = None


# Approximate cross-species DNA identity percentages for selected genes
CONSERVATION = {
    "HBB": {"Pan troglodytes": 0.99, "Mus musculus": 0.87},
    "BRCA1": {"Pan troglodytes": 0.99, "Mus musculus": 0.56},
    "HTT": {"Pan troglodytes": 0.99, "Mus musculus": 0.90},
    "CFTR": {"Pan troglodytes": 0.98, "Mus musculus": 0.78},
    "TP53": {"Pan troglodytes": 0.99, "Mus musculus": 0.80},
}
def load_genes():
    genes = []
    for symbol, meta in GENE_DB.items():
        genes.append(
            {
                "gene": symbol,
                "category": meta.get("category", "N/A"),
                "disease": meta.get("disease", "N/A"),
                "description": meta.get("description", ""),
                "sequence": meta.get("sequence", ""),
                "refseq_mrna": meta.get("refseq_mrna", ""),
                "sequence_note": meta.get("sequence_note", ""),
                "protein_name": meta.get("protein_name", ""),
                "go_function": meta.get("go_function", ""),
                "subcellular_location": meta.get("subcellular_location", ""),
                "variants": meta.get("variants", []),
                "pdb_id": meta.get("pdb_id", ""),
            }
        )
    return genes


def normalize_seq(seq: str) -> str:
    return "".join(str(seq or "").upper().split())


def gc_content_percent(dna_sequence: str) -> float:
    seq = normalize_seq(dna_sequence)
    if not seq:
        return 0.0
    gc = sum(1 for ch in seq if ch in ("G", "C"))
    return (gc / len(seq)) * 100.0


def molecular_weight_dna(dna_sequence: str) -> float:
    """Approximate molecular weight of DNA (Daltons), using 330 Da per nucleotide."""
    seq = normalize_seq(dna_sequence)
    return 330.0 * len(seq)


def melting_temperature_tm(dna_sequence: str) -> float:
    """
    Approximate melting temperature (Tm) in °C.

    Uses a simple empirical formula:
      Tm = 64.9 + 41 * (yGC - 16.4) / N
    where:
      N   = total length of the sequence
      yGC = count of G and C in the sequence
    """
    seq = normalize_seq(dna_sequence)
    N = len(seq)
    if N == 0:
        return 0.0
    y_gc = sum(1 for ch in seq if ch in ("G", "C"))
    return 64.9 + 41.0 * (y_gc - 16.4) / N


def wallace_tm(dna_sequence: str) -> float:
    """
    Wallace rule Tm for short oligos:
      Tm = 2 × (A + T) + 4 × (G + C)
    """
    seq = normalize_seq(dna_sequence)
    if not seq:
        return 0.0
    a = seq.count("A")
    t = seq.count("T")
    g = seq.count("G")
    c = seq.count("C")
    return 2.0 * (a + t) + 4.0 * (g + c)


CODON_TABLE = {
    "UUU": "F",
    "UUC": "F",
    "UUA": "L",
    "UUG": "L",
    "UCU": "S",
    "UCC": "S",
    "UCA": "S",
    "UCG": "S",
    "UAU": "Y",
    "UAC": "Y",
    "UAA": "*",
    "UAG": "*",
    "UGU": "C",
    "UGC": "C",
    "UGA": "*",
    "UGG": "W",
    "CUU": "L",
    "CUC": "L",
    "CUA": "L",
    "CUG": "L",
    "CCU": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAU": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGU": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AUU": "I",
    "AUC": "I",
    "AUA": "I",
    "AUG": "M",
    "ACU": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAU": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGU": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GUU": "V",
    "GUC": "V",
    "GUA": "V",
    "GUG": "V",
    "GCU": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAU": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGU": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def transcribe_dna_to_rna(dna_sequence: str) -> str:
    return normalize_seq(dna_sequence).replace("T", "U")


def translate_dna_to_protein(dna_sequence: str, stop_at_stop: bool = True) -> str:
    rna = transcribe_dna_to_rna(dna_sequence)
    if not rna:
        return ""
    protein = []
    for i in range(0, len(rna) - 2, 3):
        codon = rna[i : i + 3]
        aa = CODON_TABLE.get(codon, "X")
        if aa == "*":
            if stop_at_stop:
                break
            protein.append("*")
        else:
            protein.append(aa)
    return "".join(protein)


def average_hydrophobicity(protein: str) -> float:
    """
    Approximate average hydrophobicity using Kyte-Doolittle scale.
    """
    if not protein:
        return 0.0
    kd = {
        "I": 4.5,
        "V": 4.2,
        "L": 3.8,
        "F": 2.8,
        "C": 2.5,
        "M": 1.9,
        "A": 1.8,
        "G": -0.4,
        "T": -0.7,
        "S": -0.8,
        "W": -0.9,
        "Y": -1.3,
        "P": -1.6,
        "H": -3.2,
        "E": -3.5,
        "Q": -3.5,
        "D": -3.5,
        "N": -3.5,
        "K": -3.9,
        "R": -4.5,
    }
    total = 0.0
    count = 0
    for aa in protein:
        if aa in kd:
            total += kd[aa]
            count += 1
    if count == 0:
        return 0.0
    return total / count


def show_3d_protein(pdb_id: str):
    """
    Renders a 3D protein structure by converting py3Dmol view to HTML and embedding it in Streamlit.
    """
    st.subheader(f"3D Protein Structure: {pdb_id}")
    view = py3Dmol.view(query=f"pdb:{pdb_id}", height=500, width=800)
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addSurface(py3Dmol.VDW, {"opacity": 0.2, "color": "white"})
    view.setHoverable({}, True,
        """function(atom,viewer,tooltip,container) { if(!atom.label) { atom.label = viewer.addLabel(atom.resn + ':' + atom.resi, {position: atom, backgroundColor: 'mintcream', fontColor:'black'}); } }""",
        """function(atom,viewer) { if(atom.label) { viewer.removeLabel(atom.label); delete atom.label; } }"""
    )
    view.spin(False)
    view.zoomTo()

    html_blob = view._make_html()
    components.html(html_blob, height=550, width=850)


def interpret_gc(gc_percent):
    if gc_percent < 40:
        return "Low GC content → DNA may be less stable and easier to denature."
    elif 40 <= gc_percent <= 60:
        return "Moderate GC content → typical for many organisms."
    else:
        return "High GC content → DNA is more stable due to stronger bonding."


def interpret_protein(protein_seq):
    if "*" in protein_seq:
        return "Stop codon detected → translation terminated."
    elif len(protein_seq) < 20:
        return "Short peptide → may not form a functional protein."
    else:
        return "Protein sequence generated → potential functional molecule."


def interpret_similarity(score):
    if score > 90:
        return "Very high similarity → likely the same or closely related gene."
    elif score > 70:
        return "Moderate similarity → may share functional regions."
    else:
        return "Low similarity → likely unrelated sequences."


def amino_acid_composition(protein_seq):
    return dict(Counter(protein_seq))


def main():
    st.set_page_config(page_title="GeneScope Biotech Dashboard", layout="wide")

    # Global styling / soft UI theme
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Open+Sans:wght@400;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
            background: #FDFDFD;
        }
        h1, h2, h3, h4, h5 {
            font-family: 'Montserrat', sans-serif;
        }
        :root, .block-container, .css-1d391kg, .stApp, .stApp > section {
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 0 0.85rem !important;
        }
        .block-container {
            padding: 2rem 0.85rem 1rem !important;
        }
        .soft-card {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid rgba(15, 23, 42, 0.12);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
            padding: 1.25rem;
            margin-bottom: 1rem;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }
        .card-title { font-size: 1.05rem; margin: 0 0 0.5rem 0; font-weight: 700; color:#0b3f80; }
        .hero {
            background: linear-gradient(135deg, #E6F0FF 0%, #F8FBFF 100%);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(71, 118, 209, 0.2);
            box-shadow: 0 8px 24px rgba(25, 68, 145, 0.08);
        }
        .hero h1 { margin:0; color:#0b3c7f; font-size:2rem; letter-spacing:0.2px; }
        .hero p { margin:0.2rem 0 0; color:#2f4f7d; font-size:0.92rem; }
        .metric-row { display:flex; gap:0.5rem; margin-bottom:0.55rem; }
        .metric-pill { background:#ffffff; border:1px solid rgba(88, 166, 255, 0.3); border-radius:12px; padding:0.45rem 0.55rem; flex:1; min-width:130px; box-shadow:0 8px 20px rgba(20,43,104,0.05); }
        .metric-label { font-size:0.68rem; letter-spacing:0.44px; color:#5e6b94; text-transform:uppercase; margin:0; font-weight:600; font-family:'Open Sans', sans-serif; }
        .metric-value { font-size:48px; font-weight:800; color:#153e82; margin:0; font-family:'Montserrat', sans-serif; line-height:1; }
        .metric-note { font-size:0.76rem; color:#4861a1; margin-top:0.15rem; }
        .path-card { background:#161b22; border-left:4px solid #dc2626; border-radius:10px; padding:0.7rem; margin-bottom:0.45rem; color:#f3f4f6; }
        .path-variant { font-family:'Courier New', monospace; font-size:0.95rem; margin:0; font-weight:700; }
        .path-condition { font-size:0.88rem; color:#dbeafe; margin:0.22rem 0; }
        .path-significance { font-size:0.8rem; color:#ff8f8f; margin:0; }
        .spec-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:0.45rem; margin-top:0.6rem; }
        .spec-pill { background:#f4f7ff; border:1px solid #dbe4fd; border-radius:10px; padding:0.45rem 0.55rem; color:#1f3570; }
        .spec-label { font-size:0.7rem; color:#5f6d90; margin:0; }
        .spec-value { font-size:0.95rem; font-weight:700; margin:0; }
        .fact-pill { border-radius: 12px; background:#eef5ff; border:1px solid #d4e3ff; padding:0.42rem 0.9rem; color:#0a2a6b; font-size:0.86rem; font-weight:600; }
        .viewer-frame { background: rgba(255,255,255,0.72); border-radius: 15px; border: 1px solid #dceafe; backdrop-filter: blur(10px); padding: 0.6rem; box-shadow: 0 12px 30px rgba(0,0,0,0.08); max-width: 820px; margin: 0 auto; }
        .viewer-legend { color: #1f3d72; font-size: 0.82rem; margin-top: 0.35rem; text-align: center; }
        .section-header { font-weight: 700; color:#09306f; margin-bottom:0.2rem; display:flex; align-items:center; gap:0.5rem; }
        .section-divider { margin-bottom:1rem; border-bottom:1px solid #e2eaf7; width:100%; }
        .info-card { background:#f7fbff; border-left:4px solid #4b82ff; border-radius:14px; padding:0.8rem; margin-bottom:0.65rem; }
        .center-content { display:flex; justify-content:center; }
        .center-main { max-width: 900px; width:100%; }
        .pathvariant { background:#ffffff; border:1px solid #dae4f4; border-radius:10px; padding:0.65rem; margin-bottom:0.5rem; width:100%; }
        .timeline-header { font-weight: 700; margin-bottom:0.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    genes = load_genes()
    gene_ids = [g["gene"] for g in genes]

    # Top navigation header with gene selection
    with st.container():
        st.markdown(
            """
            <div style='max-width:1200px; margin:0 auto; padding:0.35rem 0 1rem; border-bottom:1px solid #e8edf8; display:flex; justify-content:center;'>
            <div style='width:100%;'>
            """,
            unsafe_allow_html=True,
        )
        nav1, nav2, nav3 = st.columns([2, 3, 1])
        with nav1:
            st.markdown("<div style='font-family:Montserrat, sans-serif; font-size:1.9rem; font-weight:800; color:#10366f; margin-bottom:0.2rem;'>🧬 GeneScope</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-family:Open Sans, sans-serif; color:#3e4f74; font-size:0.82rem;'>Gene analytics and structure insights</div>", unsafe_allow_html=True)
        with nav2:
            selected_id = st.selectbox("Gene Selection", options=gene_ids, index=0, label_visibility='visible', help="Search and select a gene")
        with nav3:
            if st.button("About"):
                st.info("GeneScope: A concise gene analytics dashboard with 3D protein preview.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Loading animation when (re)computing for a selected gene
    with st.spinner("Loading gene profile..."):
        selected_gene = next(g for g in genes if g["gene"] == selected_id)
        seq = normalize_seq(selected_gene["sequence"])
        gc = gc_content_percent(seq)
        mw = molecular_weight_dna(seq)
        tm_empirical = melting_temperature_tm(seq)
        tm_wallace_value = wallace_tm(seq)
        pdb_id = selected_gene.get("pdb_id", "")

    # Summary section with compressed metric cards
    with st.container():
        st.markdown("<div style='border-left:4px solid #3b6fd3; padding-left:0.45rem; margin-bottom:0.6rem; color:#0f4f8b; font-weight:700; font-size:1.1rem;'>Gene Snapshot</div>", unsafe_allow_html=True)
        mark_col1, mark_col2, mark_col3, mark_col4 = st.columns(4)
        mark_col1.markdown(f"<div style='background:#ffffff; border:1px solid #dbe6ff; border-radius:10px; padding:0.4rem 0.55rem;'><div style='font-size:0.7rem; color:#66729a; margin-bottom:0.25rem;'>Gene</div><div style='font-weight:700; color:#143f7f;'>{selected_gene.get('gene','—')}</div></div>", unsafe_allow_html=True)
        mark_col2.markdown(f"<div style='background:#ffffff; border:1px solid #dbe6ff; border-radius:10px; padding:0.4rem 0.55rem;'><div style='font-size:0.7rem; color:#66729a; margin-bottom:0.25rem;'>Category</div><div style='font-weight:700; color:#143f7f;'>{selected_gene.get('category','—')}</div></div>", unsafe_allow_html=True)
        with mark_col3:
            st.metric("GC Content (%)", f"{gc:.2f}")
            st.caption(interpret_gc(gc))
        mark_col4.markdown(f"<div style='background:#ffffff; border:1px solid #dbe6ff; border-radius:10px; padding:0.4rem 0.55rem;'><div style='font-size:0.7rem; color:#66729a; margin-bottom:0.25rem;'>Length</div><div style='font-weight:700; color:#143f7f;'>{len(seq)} bp</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.45rem; border-left:4px solid #3b6fd3; padding-left:0.45rem; color:#0f4f8b; font-weight:700; font-size:1.1rem;'>Thermodynamic Metrics</div>", unsafe_allow_html=True)
        th1, th2, th3 = st.columns(3)
        th1.markdown(f"<div style='background:#ffffff; border:1px solid #dbe6ff; border-radius:10px; padding:0.4rem 0.55rem;'><div style='font-size:0.7rem; color:#66729a; margin-bottom:0.25rem;'>Molecular Weight</div><div style='font-weight:700; color:#143f7f;'>{mw:,.0f} Da</div></div>", unsafe_allow_html=True)
        th2.markdown(f"<div style='background:#ffffff; border:1px solid #dbe6ff; border-radius:10px; padding:0.4rem 0.55rem;'><div style='font-size:0.7rem; color:#66729a; margin-bottom:0.25rem;'>Wallace Tm</div><div style='font-weight:700; color:#143f7f;'>{tm_wallace_value:.1f} °C</div></div>", unsafe_allow_html=True)
        th3.markdown(f"<div style='background:#ffffff; border:1px solid #dbe6ff; border-radius:10px; padding:0.4rem 0.55rem;'><div style='font-size:0.7rem; color:#66729a; margin-bottom:0.25rem;'>Empirical Tm</div><div style='font-weight:700; color:#143f7f;'>{tm_empirical:.1f} °C</div></div>", unsafe_allow_html=True)

    # Main centered body and tabs for different views
    main_left, main_center, main_right = st.columns([1, 5, 1])
    with main_center:
        tab_seq, tab_viz, tab_trans = st.tabs(
            ["🧬 Sequence Analysis", "📊 Visualization", "🧪 Translation"]
        )

        with tab_seq:
            st.markdown("<div class='soft-card'><div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.45rem;'><div><h4 style='margin:0 0 0.15rem 0; color:#0f4f8b;'>Entry Overview</h4><p style='margin:0; color:#3d4f69; font-size:0.9rem;'>Quick gene metadata in spec format.</p></div></div><div class='spec-grid'>", unsafe_allow_html=True)
            st.markdown(f"<div class='spec-pill'><p class='spec-label'>Protein</p><p class='spec-value'>{selected_gene.get('protein_name','—')}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='spec-pill'><p class='spec-label'>Location</p><p class='spec-value'>{selected_gene.get('subcellular_location','—')}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='spec-pill'><p class='spec-label'>Category</p><p class='spec-value'>{selected_gene.get('category','—')}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='spec-pill'><p class='spec-label'>Condition</p><p class='spec-value'>{selected_gene.get('disease','—')}</p></div>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

            st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
            st.markdown("<h5 class='card-title'>Function & Sequence</h5>", unsafe_allow_html=True)
            if selected_gene.get("go_function"):
                st.markdown(selected_gene["go_function"])
            else:
                st.markdown(selected_gene.get("description", ""))

            st.markdown(f"<div class='info-card'><strong>Category:</strong> {selected_gene.get('category','N/A')}<br><strong>Disease:</strong> {selected_gene.get('disease','N/A')}<br><strong>Biological Role:</strong> {selected_gene.get('go_function','N/A')}</div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:0.5rem;'><strong>DNA Sequence</strong></div>", unsafe_allow_html=True)
            if selected_gene.get("refseq_mrna"):
                st.caption(
                    f"Sequence source: RefSeq {selected_gene['refseq_mrna']} ({selected_gene.get('sequence_note','').strip()})"
                )

            with st.expander("View Raw DNA Sequence", expanded=False):
                st.code(seq, language="text")

            st.markdown("<div style='padding:0.35rem 0.4rem; font-size:0.82rem; color:#4f5f7d; background:#f5f7ff; border-radius:8px; margin-top:0.35rem; width:fit-content;'>Length: " + str(len(seq)) + " bases</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            variants = selected_gene.get("variants", []) or []
            if variants:
                st.markdown("<div class='soft-card'><div style='border-left:4px solid #3b6fd3; padding-left:0.45rem; margin-bottom:0.35rem; color:#0f4f8b; font-weight:700; font-size:1rem;'>🧬 Clinical Variant Analysis</div>", unsafe_allow_html=True)
                df_vars = pd.DataFrame(variants).reset_index(drop=True)
                cols = st.columns(2)
                for idx, row in df_vars.iterrows():
                    with cols[idx % 2]:
                        significance = str(row.get("significance", "Unknown"))
                        border_color = "#dc2626" if significance.lower() == "pathogenic" else "#2e7d32"
                        st.markdown(
                            f"<div style='border:1px solid #dbe4fd; border-radius:12px; background:#fff; padding:0.8rem; margin-bottom:0.5rem; font-family:Open Sans, sans-serif;'>"
                            f"<div style='font-family:Montserrat, sans-serif; font-size:0.95rem; font-weight:700; color:#11396e; margin-bottom:0.25rem;'>{row.get('variant', 'Variant')}</div>"
                            f"<div style='font-size:0.85rem; color:#3f4d6e; margin-bottom:0.2rem;'>Condition: {row.get('condition', 'N/A')}</div>"
                            f"<div style='font-size:0.82rem; color:#3e4d6f; margin-bottom:0.2rem;'>Note: {row.get('note', '—')}</div>"
                            f"<div style='font-size:0.8rem; font-weight:600; color:{border_color};'>Significance: {significance}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)

    with tab_viz:
        st.markdown("#### High‑Tech GC & Nucleotide Visualization")
        if px is None or np is None or go is None:
            st.warning(
                "Plotly is not installed in this environment. "
                "Install it with `pip install plotly` to see interactive charts."
            )
        else:
            series = pd.Series(list(seq))
            counts = series.value_counts().reindex(["A", "C", "G", "T"]).fillna(0)
            total = counts.sum()

            st.markdown("<div class='soft-card'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;'><div><h5 class='card-title'>Composition Card</h5><p style='margin:0;color:#40577b;font-size:0.9rem;'>GC stability gauge and nucleotide composition in a unified card.</p></div></div>", unsafe_allow_html=True)
            with st.container():
                compose_col1, compose_col2 = st.columns([1, 1])
                with compose_col1:
                    st.markdown("### GC Content Gauge")
                    if total > 0:
                        if gc > 60:
                            gauge_color = "red"
                            gauge_label = "High Stability"
                        elif gc >= 40:
                            gauge_color = "green"
                            gauge_label = "Normal"
                        else:
                            gauge_color = "blue"
                            gauge_label = "Low GC"

                        gauge_fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=gc,
                                title={"text": f"GC % – {gauge_label}"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {"color": gauge_color},
                                    "steps": [
                                        {"range": [0, 40], "color": "rgba(37, 99, 235, 0.35)"},
                                        {"range": [40, 60], "color": "rgba(34, 197, 94, 0.35)"},
                                        {"range": [60, 100], "color": "rgba(239, 68, 68, 0.35)"},
                                    ],
                                    "threshold": {
                                        "line": {"color": gauge_color, "width": 4},
                                        "thickness": 0.75,
                                        "value": gc,
                                    },
                                },
                            )
                        )
                        gauge_fig.update_layout(template="plotly_white", height=260, font=dict(family='Open Sans, sans-serif', color='#0f3a74'))
                        st.plotly_chart(gauge_fig, use_container_width=True)

                with compose_col2:
                    st.markdown("### Nucleotide Composition Donut")
                    if total > 0:
                        counts = counts.reset_index()
                        counts.columns = ["Nucleotide", "Count"]
                        counts["Percent"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
                        donut_fig = px.pie(
                            counts,
                            values="Count",
                            names="Nucleotide",
                            hole=0.5,
                            color_discrete_sequence=["#2f6fd3", "#7fc0ff", "#2f9f9f", "#82c07f"],
                            title="Nucleotide Composition",
                        )
                        donut_fig.update_traces(pull=[0.03, 0, 0, 0], textinfo="label+percent")
                        donut_fig.update_layout(template="plotly_white", legend_title_text="Nucleotide", margin=dict(t=40, l=0, r=0, b=0), font=dict(family='Open Sans, sans-serif', color='#0f3a74'))
                        st.plotly_chart(donut_fig, use_container_width=True)
                        st.markdown(
                            "<div style='font-size:0.85rem;color:#1f3a70;'>Legend: A, T, C, G counts and percentages shown on chart slices.</div>",
                            unsafe_allow_html=True,
                        )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='soft-card'><div style='display:flex;justify-content:space-between;align-items:center; margin-bottom:0.4rem;'><div><h5 class='card-title'>Sequence Stability Analysis</h5><p style='margin:0;color:#40577b;font-size:0.9rem;'>Heatmap of GC density across sequence segments.</p></div></div>", unsafe_allow_html=True)
            if len(seq) > 0:
                gc_matrix = np.zeros((10, 10))
                length = len(seq)
                for idx in range(100):
                    start = int(idx * length / 100)
                    end = int((idx + 1) * length / 100)
                    if end <= start:
                        end = min(start + 1, length)
                    window = seq[start:end]
                    gc_pct = gc_content_percent(window)
                    r, c = divmod(idx, 10)
                    gc_matrix[r, c] = gc_pct

                heatmap_fig = go.Figure(
                    go.Heatmap(
                        z=gc_matrix,
                        colorscale="Magma",
                        showscale=True,
                        hovertemplate="Segment (%{x}, %{y}): %{z:.1f}% GC<extra></extra>",
                        xgap=2,
                        ygap=2,
                    )
                )
                heatmap_fig.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#fafbff",
                    plot_bgcolor="#fafbff",
                    title="GC Stability Map",
                    xaxis_title="Segment column",
                    yaxis_title="Segment row",
                    font=dict(family='Open Sans, sans-serif', color='#0f3a74'),
                    margin=dict(t=35, l=30, r=30, b=30),
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # BLAST-style conservation summary
        st.markdown("#### BLAST Analysis – Sequence Conservation")
        species_map = CONSERVATION.get(selected_gene["gene"], {})
        if species_map and px is not None:
            # Identity bar chart
            species = list(species_map.keys())
            identities = [round(v * 100, 2) for v in species_map.values()]
            cons_df = pd.DataFrame(
                {"Species": species, "Identity (%)": identities}
            )
            cons_fig = px.bar(
                cons_df,
                x="Species",
                y="Identity (%)",
                title="Pairwise DNA Identity vs Human (CDS fragment)",
                text="Identity (%)",
                template="plotly_dark",
            )
            cons_fig.update_traces(textposition="outside")
            cons_fig.update_yaxes(range=[0, 100])
            st.plotly_chart(cons_fig, use_container_width=True)

            # Simple positional mismatch map given the identity fractions
            if np is not None and len(seq) > 0:
                n_pos = len(seq)
                species_list = list(species_map.keys())
                mismatch_matrix = np.ones((len(species_list), n_pos))
                for i, sp in enumerate(species_list):
                    identity = species_map[sp]
                    n_mismatch = int(round((1.0 - identity) * n_pos))
                    if n_mismatch <= 0:
                        continue
                    # Mark first n_mismatch positions as mismatches (schematic)
                    mismatch_matrix[i, :n_mismatch] = 0.0

                mismatch_fig = go.Figure(
                    go.Heatmap(
                        z=mismatch_matrix,
                        colorscale=[[0.0, "#ffb3b3"], [0.5, "#ff4d4d"], [1.0, "#7f0000"]],
                        zmin=0,
                        zmax=1,
                        opacity=0.6,
                        showscale=False,
                        hoverinfo='z',
                    )
                )
                mismatch_fig.update_layout(
                    title="Sequence Mismatches (glowing mismatch regions)",
                    template="plotly_white",
                    paper_bgcolor="#f5f7fb",
                    plot_bgcolor="#f5f7fb",
                    font=dict(family='Open Sans, sans-serif', color='#0f3a74'),
                    margin=dict(t=35, l=30, r=30, b=30),
                )
                mismatch_fig.update_yaxes(
                    tickmode="array",
                    tickvals=list(range(len(species_list))),
                    ticktext=species_list,
                    title="Species",
                )
                mismatch_fig.update_xaxes(
                    title="Aligned position (schematic)",
                    showticklabels=False,
                )
                st.plotly_chart(mismatch_fig, use_container_width=True)
        else:
            st.caption(
                "Conservation data not configured for this gene or Plotly is unavailable."
            )

        # Gene fact sheet
        st.markdown("##### Gene Fact Sheet")
        st.info(
            f"**{selected_gene['gene']}** – {selected_gene.get('description', 'No description available.')}"
        )

    with tab_trans:
        st.markdown("---")
        st.markdown("<div style='background:#eef4ff;border-left:4px solid #4b82ff;border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.6rem;'><strong style='color:#103e7d;'>3D Visualization and Protein Translation</strong></div>", unsafe_allow_html=True)
        with st.container():
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                if st.button("Translate to Protein"):
                    protein = translate_dna_to_protein(seq)
                    st.code(protein, language="text")
                    st.markdown(f"<div style='font-size:0.85rem; color:#3e4f74; margin-top:0.5rem; padding:0.4rem; background:#f0f4ff; border-radius:8px;'>{interpret_protein(protein)}</div>", unsafe_allow_html=True)
                    
                    aa_counts = amino_acid_composition(protein)
                    st.subheader("Amino Acid Composition")
                    st.bar_chart(aa_counts)
                    
                    hydrophobicity = average_hydrophobicity(protein)
                    st.markdown("<div class='info-card'><strong>Hydrophobicity</strong><br/>Kyte-Doolittle: <span style='font-weight:700;'>" + f"{hydrophobicity:.2f}" + "</span></div>", unsafe_allow_html=True)
                    if px is not None:
                        aa_series = pd.Series(list(protein))
                        aa_freq = (
                            aa_series.value_counts(normalize=True).sort_index() * 100.0
                        ).round(2)
                        freq_df = aa_freq.reset_index()
                        freq_df.columns = ["Amino Acid", "Percent"]
                        freq_fig = px.bar(
                            freq_df,
                            x="Amino Acid",
                            y="Percent",
                            text="Percent",
                            orientation="v",
                            color="Percent",
                            color_continuous_scale="Viridis",
                            title="Amino Acid Frequency (%)",
                            template="plotly_white",
                        )
                        freq_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        freq_fig.update_layout(yaxis_title="Percent", margin=dict(l=10,r=10,t=30,b=10), font=dict(family='Open Sans, sans-serif', color='#0f3a74'))
                        st.plotly_chart(freq_fig, use_container_width=True)

                        counts = aa_series.value_counts()
                        total = counts.sum()
                        donut_fig = px.pie(
                            values=counts.values,
                            names=counts.index,
                            hole=0.55,
                            color_discrete_sequence=px.colors.sequential.Blues,
                            title="Amino Acid Composition",
                            template="plotly_white",
                        )
                        donut_fig.update_traces(textinfo='percent+label')
                        st.plotly_chart(donut_fig, use_container_width=True)
                        st.markdown("<div style='font-size:0.9rem;color:#2f4f7f;'>Legend: Amino acid counts with percent in chart labels.</div>", unsafe_allow_html=True)

                    st.markdown("<div class='viewer-frame'>", unsafe_allow_html=True)
                    target_pdb = pdb_id or "4HHB"
                    show_3d_protein(target_pdb)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.caption("Blue = N-terminus, Red = C-terminus. Hover shows residue and position.")

                    st.markdown("---")
                    st.subheader("Mutation Simulator")

                    position = st.number_input("Position (1-based index)", min_value=1, max_value=len(seq))
                    new_base = st.selectbox("New Nucleotide", ["A", "T", "G", "C"])
                    if st.button("Apply Mutation"):
                        if 1 <= position <= len(seq):
                            original_base = seq[position-1]
                            mutated_seq = seq[:position-1] + new_base + seq[position:]
                            mutated_protein = translate_dna_to_protein(mutated_seq)
                            st.write(f"**Original base at position {position}:** {original_base}")
                            st.write(f"**Mutated base:** {new_base}")
                            st.code(f"Original protein: {protein}", language="text")
                            st.code(f"Mutated protein:  {mutated_protein}", language="text")
                            if protein == mutated_protein:
                                mutation_type = "Silent mutation"
                            elif "*" in mutated_protein and "*" not in protein:
                                mutation_type = "Nonsense mutation"
                            else:
                                mutation_type = "Missense mutation"
                            st.success(f"Mutation type: {mutation_type}")
                        else:
                            st.error("Invalid position.")
                else:
                    st.info("Click 'Translate to Protein' to generate the amino acid sequence and its hydrophobicity.")


if __name__ == "__main__":
    main()

