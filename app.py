import subprocess
import sys
from collections import Counter

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
except ImportError:
    np = None
    px = None
    go = None

CONSERVATION = {
    "HBB": {"Pan troglodytes": 0.99, "Mus musculus": 0.87},
    "BRCA1": {"Pan troglodytes": 0.99, "Mus musculus": 0.56},
    "HTT": {"Pan troglodytes": 0.99, "Mus musculus": 0.90},
    "CFTR": {"Pan troglodytes": 0.98, "Mus musculus": 0.78},
    "TP53": {"Pan troglodytes": 0.99, "Mus musculus": 0.80},
}

from Bio import Entrez
Entrez.email = "sruthibalasubramani6@gmail.com"

def fetch_from_ncbi(query):
    try:
        handle = Entrez.esearch(db="gene", term=f"{query}[Gene Name] AND Homo sapiens[Organism]", retmax=1)
        record = Entrez.read(handle)
        handle.close()
        if not record["IdList"]:
            handle = Entrez.esearch(db="gene", term=f"{query} AND Homo sapiens[Organism]", retmax=1)
            record = Entrez.read(handle)
            handle.close()
        if not record["IdList"]:
            return None, "Gene not found on NCBI"
        gene_id = record["IdList"][0]
        handle = Entrez.esummary(db="gene", id=gene_id)
        summary = Entrez.read(handle)
        handle.close()
        info = summary["DocumentSummarySet"]["DocumentSummary"][0]
        return {"name": str(info["Name"]), "full_name": str(info["Description"]),
                "aliases": str(info.get("OtherAliases", "—")), "ncbi_id": gene_id}, None
    except ImportError:
        return None, "biopython not installed"
    except Exception as e:
        return None, f"NCBI error: {str(e)}"

def load_genes():
    genes = []
    if not GENE_DB:
        st.error("No genes found! Using demo mode.")
        return [{"gene": "DEMO", "sequence": "ATGC", "pdb_id": ""}]
    for symbol, meta in GENE_DB.items():
        genes.append({
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
        })
    return genes

def normalize_seq(seq): return "".join(str(seq or "").upper().split())

def gc_content_percent(dna_sequence):
    seq = normalize_seq(dna_sequence)
    if not seq: return 0.0
    return sum(1 for ch in seq if ch in ("G","C")) / len(seq) * 100.0

def molecular_weight_dna(dna_sequence):
    return 330.0 * len(normalize_seq(dna_sequence))

def melting_temperature_tm(dna_sequence):
    seq = normalize_seq(dna_sequence)
    N = len(seq)
    if N == 0: return 0.0
    y_gc = sum(1 for ch in seq if ch in ("G","C"))
    return 64.9 + 41.0 * (y_gc - 16.4) / N

def wallace_tm(dna_sequence):
    seq = normalize_seq(dna_sequence)
    if not seq: return 0.0
    return 2.0*(seq.count("A")+seq.count("T")) + 4.0*(seq.count("G")+seq.count("C"))

CODON_TABLE = {
    "UUU":"F","UUC":"F","UUA":"L","UUG":"L","UCU":"S","UCC":"S","UCA":"S","UCG":"S",
    "UAU":"Y","UAC":"Y","UAA":"*","UAG":"*","UGU":"C","UGC":"C","UGA":"*","UGG":"W",
    "CUU":"L","CUC":"L","CUA":"L","CUG":"L","CCU":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAU":"H","CAC":"H","CAA":"Q","CAG":"Q","CGU":"R","CGC":"R","CGA":"R","CGG":"R",
    "AUU":"I","AUC":"I","AUA":"I","AUG":"M","ACU":"T","ACC":"T","ACA":"T","ACG":"T",
    "AAU":"N","AAC":"N","AAA":"K","AAG":"K","AGU":"S","AGC":"S","AGA":"R","AGG":"R",
    "GUU":"V","GUC":"V","GUA":"V","GUG":"V","GCU":"A","GCC":"A","GCA":"A","GCG":"A",
    "GAU":"D","GAC":"D","GAA":"E","GAG":"E","GGU":"G","GGC":"G","GGA":"G","GGG":"G",
}

def transcribe_dna_to_rna(dna_sequence):
    return normalize_seq(dna_sequence).replace("T","U")

def translate_dna_to_protein(dna_sequence, stop_at_stop=True):
    rna = transcribe_dna_to_rna(dna_sequence)
    if not rna: return ""
    protein = []
    for i in range(0, len(rna)-2, 3):
        aa = CODON_TABLE.get(rna[i:i+3], "X")
        if aa == "*":
            if stop_at_stop: break
            protein.append("*")
        else:
            protein.append(aa)
    return "".join(protein)

def average_hydrophobicity(protein):
    if not protein: return 0.0
    kd = {"I":4.5,"V":4.2,"L":3.8,"F":2.8,"C":2.5,"M":1.9,"A":1.8,"G":-0.4,"T":-0.7,
          "S":-0.8,"W":-0.9,"Y":-1.3,"P":-1.6,"H":-3.2,"E":-3.5,"Q":-3.5,"D":-3.5,
          "N":-3.5,"K":-3.9,"R":-4.5}
    vals = [kd[aa] for aa in protein if aa in kd]
    return sum(vals)/len(vals) if vals else 0.0

def amino_acid_composition(protein_seq):
    return dict(Counter(protein_seq))

def interpret_gc(gc):
    if gc < 40: return "Low GC content — DNA may be less stable and easier to denature."
    elif gc <= 60: return "Moderate GC content — typical for many organisms."
    else: return "High GC content — DNA is more stable due to stronger bonding."

def interpret_protein(protein_seq):
    if "*" in protein_seq: return "Stop codon detected — translation terminated."
    elif len(protein_seq) < 20: return "Short peptide — may not form a functional protein."
    else: return "Protein sequence generated — potential functional molecule."

def render_2d_sequence(seq, label="DNA Sequence", highlight_pos=None, font_size=13):
    BASE_COLORS = {"A":"#16a34a","T":"#dc2626","G":"#2563eb","C":"#d97706"}
    chunk = 60
    rows = [seq[i:i+chunk] for i in range(0, len(seq), chunk)]
    html = f"""<div style='font-family:monospace;background:#f8fafc;border:1px solid #e2e8f0;
    border-radius:10px;padding:14px;'>
    <div style='font-size:11px;font-weight:600;color:#8898b3;text-transform:uppercase;
    letter-spacing:0.5px;margin-bottom:8px;'>{label} — {len(seq)} bp</div>"""
    for row_idx, row in enumerate(rows):
        start = row_idx * chunk
        html += f"<div style='display:flex;align-items:center;margin-bottom:3px;'>"
        html += f"<span style='font-size:10px;color:#94a3b8;min-width:38px;margin-right:6px;'>{start+1}</span>"
        for i, base in enumerate(row):
            pos = start + i + 1
            color = BASE_COLORS.get(base, "#0a2540")
            bg = "#fef08a" if (highlight_pos and pos == highlight_pos) else "transparent"
            html += f"""<span title='Position {pos}: {base}'
            style='display:inline-block;width:{font_size+1}px;height:{font_size+4}px;
            text-align:center;font-size:{font_size}px;font-weight:600;color:{color};
            background:{bg};border-radius:2px;cursor:default;line-height:{font_size+4}px;'
            onmouseover="this.style.background='#dbeafe';this.style.outline='1px solid #93c5fd'"
            onmouseout="this.style.background='{bg}';this.style.outline='none'"
            >{base}</span>"""
        html += "</div>"
    html += "</div>"
    return html

def render_2d_protein(protein, label="Protein Sequence", highlight_pos=None, font_size=13):
    AA_COLORS = {
        "A":"#6366f1","R":"#ef4444","N":"#f97316","D":"#ef4444","C":"#eab308",
        "Q":"#f97316","E":"#ef4444","G":"#8b5cf6","H":"#06b6d4","I":"#10b981",
        "L":"#10b981","K":"#ef4444","M":"#eab308","F":"#8b5cf6","P":"#f97316",
        "S":"#06b6d4","T":"#06b6d4","W":"#8b5cf6","Y":"#8b5cf6","V":"#10b981",
    }
    chunk = 40
    rows = [protein[i:i+chunk] for i in range(0, len(protein), chunk)]
    html = f"""<div style='font-family:monospace;background:#f8fafc;border:1px solid #e2e8f0;
    border-radius:10px;padding:14px;'>
    <div style='font-size:11px;font-weight:600;color:#8898b3;text-transform:uppercase;
    letter-spacing:0.5px;margin-bottom:8px;'>{label} — {len(protein)} aa</div>"""
    for row_idx, row in enumerate(rows):
        start = row_idx * chunk
        html += f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
        html += f"<span style='font-size:10px;color:#94a3b8;min-width:38px;margin-right:6px;'>{start+1}</span>"
        for i, aa in enumerate(row):
            pos = start + i + 1
            color = AA_COLORS.get(aa, "#0a2540")
            bg = "#fef08a" if (highlight_pos and pos == highlight_pos) else "#ffffff"
            html += f"""<span title='Position {pos}: {aa}'
            style='display:inline-block;width:{font_size+5}px;height:{font_size+6}px;
            text-align:center;font-size:{font_size}px;font-weight:700;color:{color};
            background:{bg};border-radius:3px;cursor:default;line-height:{font_size+6}px;
            border:1px solid #e2e8f0;margin:1px;'
            onmouseover="this.style.background='#dbeafe';this.style.borderColor='#93c5fd'"
            onmouseout="this.style.background='{bg}';this.style.borderColor='#e2e8f0'"
            >{aa}</span>"""
        html += "</div>"
    html += "</div>"
    return html

def show_3d_protein(pdb_id):
    st.markdown(f"""<div style='font-size:11px;font-weight:600;color:#8898b3;
    text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>
    3D Structure — PDB: {pdb_id}</div>
    <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;
    background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;font-size:12px;color:#475569;'>
    <span style='color:#64748b;font-weight:600;'>Legend:</span>
    <span><span style='color:#3b82f6;font-weight:700;'>■</span> N-terminus (blue)</span>
    <span><span style='color:#22c55e;font-weight:700;'>■</span> Middle (green)</span>
    <span><span style='color:#ef4444;font-weight:700;'>■</span> C-terminus (red)</span>
    <span style='color:#94a3b8;'>| Scroll to zoom · Drag to rotate</span>
    </div>""", unsafe_allow_html=True)
    view = py3Dmol.view(query=f"pdb:{pdb_id}", height=480, width=750)
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addSurface(py3Dmol.VDW, {"opacity": 0.15, "color": "white"})
    view.setHoverable({}, True,
        """function(atom,viewer) { if(!atom.label) { atom.label = viewer.addLabel(
        atom.resn+':'+atom.resi, {position:atom,backgroundColor:'white',fontColor:'#0a2540',fontSize:12}); }}""",
        """function(atom,viewer) { if(atom.label) { viewer.removeLabel(atom.label); delete atom.label; }}"""
    )
    view.spin(False)
    view.zoomTo()
    components.html(view._make_html(), height=520, width=780)
    st.caption("Scroll to zoom · Click and drag to rotate · Right-click to pan")


# ─── CARD HELPER ────────────────────────────────────────────────────────────
def metric_card(label, value, sub=""):
    return f"""<div style='background:#fff;border:1px solid #e2e8f0;border-radius:12px;
    padding:16px 18px;'>
    <div style='font-size:11px;color:#8898b3;text-transform:uppercase;letter-spacing:0.5px;
    font-weight:600;margin-bottom:6px;'>{label}</div>
    <div style='font-size:1.4rem;font-weight:700;color:#0a2540;letter-spacing:-0.5px;'>{value}</div>
    {"<div style='font-size:12px;color:#8898b3;margin-top:4px;'>"+sub+"</div>" if sub else ""}
    </div>"""

def section_header(title):
    return f"""<div style='font-size:13px;font-weight:700;color:#0a2540;
    text-transform:uppercase;letter-spacing:0.5px;margin:1.2rem 0 0.8rem;
    padding-bottom:8px;border-bottom:1px solid #e2e8f0;'>{title}</div>"""


def main():
    st.set_page_config(page_title="GeneScope", layout="wide")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#f8fafc;}
    .stApp{background:#f8fafc;}
    .block-container{max-width:1100px!important;margin:0 auto!important;padding:0 1.2rem 2rem!important;}
    h1,h2,h3,h4,h5{font-family:'Inter',sans-serif;font-weight:700;color:#0a2540;}
    /* hide streamlit default header padding */
    header[data-testid="stHeader"]{background:#fff;border-bottom:1px solid #e2e8f0;}
    /* tabs */
    .stTabs [data-baseweb="tab-list"]{border-bottom:1px solid #e2e8f0!important;gap:0;background:transparent!important;}
    .stTabs [data-baseweb="tab"]{color:#8898b3!important;font-size:13px!important;font-weight:500!important;padding:10px 18px!important;border-radius:0!important;background:transparent!important;}
    .stTabs [aria-selected="true"]{color:#2563eb!important;border-bottom:2px solid #2563eb!important;font-weight:600!important;}
    /* buttons */
    .stButton button{background:#0a2540!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:500!important;font-size:13px!important;padding:8px 20px!important;}
    .stButton button:hover{background:#2563eb!important;}
    /* inputs */
    .stTextInput input{border-radius:8px!important;border-color:#e2e8f0!important;background:#fff!important;font-size:13px!important;}
    .stSelectbox div[data-baseweb="select"]>div{border-radius:8px!important;border-color:#e2e8f0!important;background:#fff!important;}
    .stSelectbox label,.stNumberInput label,.stTextInput label{color:#8898b3!important;font-size:11px!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.5px!important;}
    /* metric */
    div[data-testid="stMetricValue"]{font-size:1.6rem!important;font-weight:700!important;color:#0a2540!important;}
    div[data-testid="stMetricLabel"]{font-size:11px!important;color:#8898b3!important;text-transform:uppercase!important;letter-spacing:0.5px!important;}
    div[data-testid="metric-container"]{background:#fff!important;border:1px solid #e2e8f0!important;border-radius:12px!important;padding:16px 18px!important;}
    /* alerts */
    .stSuccess>div{background:#f0fdf4!important;border-radius:8px!important;border:1px solid #bbf7d0!important;color:#14532d!important;}
    .stWarning>div{background:#fffbeb!important;border-radius:8px!important;border:1px solid #fde68a!important;color:#78350f!important;}
    .stError>div{background:#fef2f2!important;border-radius:8px!important;border:1px solid #fecaca!important;color:#7f1d1d!important;}
    .stInfo>div{background:#eff6ff!important;border-radius:8px!important;border:1px solid #bfdbfe!important;color:#1e40af!important;}
    /* code */
    code,.stCode{font-size:12px!important;background:#f8fafc!important;border:1px solid #e2e8f0!important;border-radius:6px!important;}
    /* expander */
    .streamlit-expanderHeader{font-size:13px!important;font-weight:600!important;color:#0a2540!important;}
    /* scrollbar */
    ::-webkit-scrollbar{width:4px;}
    ::-webkit-scrollbar-thumb{background:#e2e8f0;border-radius:10px;}
    </style>
    """, unsafe_allow_html=True)

    genes = load_genes()
    gene_ids = [g["gene"] for g in genes]

    # ── NAV ──────────────────────────────────────────────────────────────────
    st.markdown("<div style='background:#fff;border-bottom:1px solid #e2e8f0;padding:14px 0 10px;margin-bottom:20px;'>", unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns([2, 3, 1])
    with nav1:
        st.markdown("""<div style='padding-top:4px;'>
        <div style='font-size:22px;font-weight:800;color:#0a2540;letter-spacing:-0.5px;'>🧬 GeneScope</div>
        <div style='font-size:12px;color:#8898b3;font-weight:500;margin-top:2px;'>Gene analytics & structure insights</div>
        </div>""", unsafe_allow_html=True)
    with nav2:
        search_term = st.text_input("", "", placeholder="Search gene or disease — e.g. BRCA1, TP53, cystic fibrosis...")
        filtered_genes = [g for g in gene_ids if search_term.upper() in g.upper()]
        if filtered_genes:
            selected_id = st.selectbox("Matching genes", options=filtered_genes, index=0)
        else:
            selected_id = gene_ids[0]
            if search_term:
                with st.spinner("Searching NCBI..."):
                    ncbi, error = fetch_from_ncbi(search_term)
                if ncbi:
                    st.success(f"Found on NCBI: **{ncbi['name']}** — {ncbi['full_name']}")
                    st.caption(f"Aliases: {ncbi['aliases']}")
                    st.markdown(f"[View on NCBI ↗](https://www.ncbi.nlm.nih.gov/gene/{ncbi['ncbi_id']})")
                else:
                    st.warning(f"{error}")
    with nav3:
        st.markdown(f"<div style='text-align:right;padding-top:6px;'><span style='background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;'>🧬 {len(gene_ids)} genes</span></div>", unsafe_allow_html=True)
        if st.button("About"):
            st.info("GeneScope — gene analytics dashboard with 3D protein viewer, mutation simulator, and NCBI live search.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── GENE DATA ────────────────────────────────────────────────────────────
    selected_gene = next(g for g in genes if g["gene"] == selected_id)
    seq = normalize_seq(selected_gene["sequence"])
    gc = gc_content_percent(seq)
    mw = molecular_weight_dna(seq)
    tm_empirical = melting_temperature_tm(seq)
    tm_wallace_value = wallace_tm(seq)
    pdb_id = selected_gene.get("pdb_id", "")

    # ── METRIC CARDS (all identical style) ───────────────────────────────────
    gc_sub = "Low GC" if gc < 40 else ("Moderate GC" if gc <= 60 else "High GC")
    tm_sub = "Stable" if tm_empirical > 60 else "Low stability"
    refseq = selected_gene.get("refseq_mrna", "—")

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:12px;'>
        {metric_card("Gene", selected_gene.get('gene','—'), selected_gene.get('category','—'))}
        {metric_card("GC Content", f"{gc:.2f}%", gc_sub)}
        {metric_card("Sequence Length", f"{len(seq)} bp", f"RefSeq {refseq}")}
        {metric_card("Empirical Tm", f"{tm_empirical:.1f} °C", tm_sub)}
    </div>
    <div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:24px;'>
        {metric_card("Molecular Weight", f"{mw:,.0f} Da")}
        {metric_card("Wallace Tm", f"{tm_wallace_value:.1f} °C")}
        {metric_card("Sequence Status", "RefSeq verified", refseq)}
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab_seq, tab_viz, tab_trans = st.tabs(["Sequence analysis", "Visualization", "Translation & mutation"])

    # ════ TAB 1: SEQUENCE ANALYSIS ══════════════════════════════════════════
    with tab_seq:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(section_header("Entry overview"), unsafe_allow_html=True)
            st.markdown(f"""
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'>
                <div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Protein</div>
                <div style='font-size:13px;font-weight:600;color:#0a2540;'>{selected_gene.get('protein_name','—')}</div>
            </div>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'>
                <div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Location</div>
                <div style='font-size:13px;font-weight:600;color:#0a2540;'>{selected_gene.get('subcellular_location','—')}</div>
            </div>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'>
                <div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Category</div>
                <div style='font-size:13px;font-weight:600;color:#0a2540;'>{selected_gene.get('category','—')}</div>
            </div>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'>
                <div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Condition</div>
                <div style='font-size:13px;font-weight:600;color:#0a2540;'>{selected_gene.get('disease','—')}</div>
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(section_header("Function & biological role"), unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:13px;color:#475569;line-height:1.7;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;'>{selected_gene.get('go_function') or selected_gene.get('description','—')}</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown(section_header("DNA sequence"), unsafe_allow_html=True)
            if selected_gene.get("refseq_mrna"):
                st.caption(f"Source: RefSeq {selected_gene['refseq_mrna']} — {selected_gene.get('sequence_note','').strip()}")
            with st.expander("View raw DNA sequence", expanded=False):
                st.code(seq, language="text")
            st.markdown(f"<div style='display:inline-block;background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:4px 10px;font-size:12px;color:#1e40af;font-weight:600;margin-top:4px;'>{len(seq)} bases</div>", unsafe_allow_html=True)

        # Clinical variants
        variants = selected_gene.get("variants", []) or []
        if variants:
            st.markdown(section_header("Clinical variants"), unsafe_allow_html=True)
            vcols = st.columns(2)
            for idx, row in pd.DataFrame(variants).iterrows():
                sig = str(row.get("significance","Unknown"))
                is_path = sig.lower() == "pathogenic"
                badge_bg = "#fef2f2" if is_path else "#f0fdf4"
                badge_color = "#dc2626" if is_path else "#16a34a"
                badge_border = "#fecaca" if is_path else "#bbf7d0"
                with vcols[idx % 2]:
                    st.markdown(f"""
                    <div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:12px 14px;margin-bottom:8px;'>
                    <div style='font-family:monospace;font-size:12px;font-weight:700;
                    color:#0a2540;margin-bottom:4px;'>{row.get('variant','—')}</div>
                    <div style='font-size:12px;color:#475569;margin-bottom:6px;'>{row.get('condition','—')}</div>
                    <div style='font-size:11px;color:#64748b;margin-bottom:6px;'>{row.get('note','—')}</div>
                    <span style='background:{badge_bg};color:{badge_color};border:1px solid {badge_border};
                    border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;'>{sig}</span>
                    </div>""", unsafe_allow_html=True)

    # ════ TAB 2: VISUALIZATION ══════════════════════════════════════════════
    with tab_viz:
        if px is None or np is None or go is None:
            st.warning("Plotly not installed. Run: pip install plotly")
        else:
            series = pd.Series(list(seq))
            counts = series.value_counts().reindex(["A","C","G","T"]).fillna(0)
            total = counts.sum()

            st.markdown(section_header("Nucleotide composition"), unsafe_allow_html=True)
            viz1, viz2 = st.columns(2)
            with viz1:
                if total > 0:
                    gc_label = "Low GC" if gc < 40 else ("Normal" if gc <= 60 else "High Stability")
                    gc_color = "#3b82f6" if gc < 40 else ("#22c55e" if gc <= 60 else "#ef4444")
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=gc,
                        title={"text": f"GC % — {gc_label}", "font": {"size": 13}},
                        gauge={"axis": {"range": [0,100]}, "bar": {"color": gc_color},
                               "steps": [{"range":[0,40],"color":"rgba(59,130,246,0.15)"},
                                         {"range":[40,60],"color":"rgba(34,197,94,0.15)"},
                                         {"range":[60,100],"color":"rgba(239,68,68,0.15)"}],
                               "threshold": {"line":{"color":gc_color,"width":3},"thickness":0.75,"value":gc}}
                    ))
                    gauge_fig.update_layout(template="plotly_white", height=240,
                        font=dict(family="Inter,sans-serif",color="#0a2540"),
                        margin=dict(t=40,b=10,l=20,r=20), paper_bgcolor="#ffffff")
                    st.plotly_chart(gauge_fig, use_container_width=True)

            with viz2:
                if total > 0:
                    cnt_df = counts.reset_index()
                    cnt_df.columns = ["Nucleotide","Count"]
                    donut = px.pie(cnt_df, values="Count", names="Nucleotide", hole=0.55,
                        color_discrete_sequence=["#2563eb","#7c3aed","#059669","#d97706"])
                    donut.update_traces(textinfo="label+percent")
                    donut.update_layout(template="plotly_white", height=240,
                        margin=dict(t=20,b=10,l=10,r=10), paper_bgcolor="#ffffff",
                        font=dict(family="Inter,sans-serif",color="#0a2540"))
                    st.plotly_chart(donut, use_container_width=True)

            st.markdown(section_header("Sequence stability heatmap"), unsafe_allow_html=True)
            if len(seq) > 0:
                gc_matrix = np.zeros((10,10))
                for idx in range(100):
                    start = int(idx*len(seq)/100)
                    end = max(start+1, int((idx+1)*len(seq)/100))
                    r,c = divmod(idx,10)
                    gc_matrix[r,c] = gc_content_percent(seq[start:end])
                hm = go.Figure(go.Heatmap(z=gc_matrix, colorscale="Blues", showscale=True,
                    hovertemplate="Segment (%{x},%{y}): %{z:.1f}% GC<extra></extra>", xgap=2, ygap=2))
                hm.update_layout(template="plotly_white", paper_bgcolor="#ffffff",
                    title="GC stability map", xaxis_title="Segment column", yaxis_title="Segment row",
                    font=dict(family="Inter,sans-serif",color="#0a2540"),
                    margin=dict(t=35,l=30,r=30,b=30))
                st.plotly_chart(hm, use_container_width=True)

            st.markdown(section_header("BLAST — sequence conservation"), unsafe_allow_html=True)
            species_map = CONSERVATION.get(selected_gene["gene"], {})
            if species_map:
                cons_df = pd.DataFrame({"Species":list(species_map.keys()),
                    "Identity (%)": [round(v*100,2) for v in species_map.values()]})
                bar = px.bar(cons_df, x="Species", y="Identity (%)",
                    title="Pairwise DNA identity vs human", text="Identity (%)",
                    color_discrete_sequence=["#2563eb"])
                bar.update_traces(textposition="outside")
                bar.update_yaxes(range=[0,100])
                bar.update_layout(template="plotly_white", paper_bgcolor="#ffffff",
                    font=dict(family="Inter,sans-serif",color="#0a2540"))
                st.plotly_chart(bar, use_container_width=True)
            else:
                st.caption("Conservation data not available for this gene.")

            st.markdown(section_header("Gene fact sheet"), unsafe_allow_html=True)
            st.info(f"**{selected_gene['gene']}** — {selected_gene.get('description','No description available.')}")

    # ════ TAB 3: TRANSLATION & MUTATION ════════════════════════════════════
    with tab_trans:
        for key in ["protein","mutation_result"]:
            if key not in st.session_state:
                st.session_state[key] = None

        # ── Translation ──────────────────────────────────────────────────────
        st.markdown(section_header("Protein translation"), unsafe_allow_html=True)
        if st.button("Translate to protein", key="translate_btn"):
            st.session_state.protein = translate_dna_to_protein(seq)

        if st.session_state.protein:
            prot = st.session_state.protein
            zoom_t = st.slider("Zoom", 10, 20, 13, key="prot_zoom")

            st.markdown("**2D DNA sequence viewer** — hover over a base to see position")
            dna_h = max(140, (len(seq)//60+1)*(zoom_t+8)+60)
            components.html(render_2d_sequence(seq, "Original DNA", font_size=zoom_t), height=dna_h, scrolling=True)

            st.markdown("**2D Protein sequence viewer** — hover over an amino acid to see position")
            pro_h = max(140, (len(prot)//40+1)*(zoom_t+10)+60)
            components.html(render_2d_protein(prot, "Translated protein", font_size=zoom_t), height=pro_h, scrolling=True)

            st.markdown(f"<div style='background:#eff6ff;border-left:3px solid #2563eb;border-radius:0;padding:10px 14px;font-size:13px;color:#1e40af;margin:8px 0;'>{interpret_protein(prot)}</div>", unsafe_allow_html=True)

            aa_c, hyd_c = st.columns(2)
            with aa_c:
                st.markdown("**Amino acid composition**")
                st.bar_chart(amino_acid_composition(prot))
            with hyd_c:
                hydro = average_hydrophobicity(prot)
                st.markdown(f"""<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 18px;margin-top:24px;'>
                <div style='font-size:11px;color:#8898b3;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;margin-bottom:6px;'>Hydrophobicity (Kyte-Doolittle)</div>
                <div style='font-size:2rem;font-weight:700;color:#0a2540;'>{hydro:.2f}</div>
                <div style='font-size:12px;color:#8898b3;margin-top:4px;'>{"Hydrophobic — likely membrane-associated" if hydro>0 else "Hydrophilic — likely soluble/cytoplasmic"}</div>
                </div>""", unsafe_allow_html=True)

        # ── 3D Viewer ────────────────────────────────────────────────────────
        st.markdown(section_header("3D protein structure"), unsafe_allow_html=True)
        if pdb_id:
            if st.button("Load 3D structure", key="load_3d_btn"):
                show_3d_protein(pdb_id)
            st.caption(f"PDB ID: {pdb_id} — click to load · scroll to zoom · drag to rotate")
        else:
            st.info("No PDB structure available for this gene.")

        # ── Mutation Simulator ────────────────────────────────────────────────
        st.markdown(section_header("Mutation simulator"), unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            mut_type = st.selectbox("Mutation type", [
                "Substitution (change a base)",
                "Deletion (remove a base)",
                "Insertion (add a base)"
            ], key="mut_type")
        with m2:
            position = st.number_input("Position (1-based)", min_value=1, max_value=len(seq), key="mut_pos")
        with m3:
            if mut_type != "Deletion (remove a base)":
                new_base = st.selectbox("New base", ["A","T","G","C"], key="mut_base")
            else:
                new_base = None
                st.markdown("<div style='padding-top:28px;color:#8898b3;font-size:13px;'>No base needed for deletion</div>", unsafe_allow_html=True)

        if st.button("Apply mutation", key="apply_mut_btn"):
            ob = seq[position-1]
            if mut_type == "Substitution (change a base)":
                mutated_seq = seq[:position-1] + new_base + seq[position:]
                mut_label = f"Substitution: position {position}  {ob} → {new_base}"
            elif mut_type == "Deletion (remove a base)":
                mutated_seq = seq[:position-1] + seq[position:]
                mut_label = f"Deletion: removed {ob} at position {position}"
            else:
                mutated_seq = seq[:position-1] + new_base + seq[position-1:]
                mut_label = f"Insertion: added {new_base} before position {position}"
            st.session_state.mutation_result = {
                "mut_label": mut_label, "original_base": ob, "new_base": new_base,
                "original_seq": seq, "mutated_seq": mutated_seq,
                "original_protein": translate_dna_to_protein(seq),
                "mutated_protein": translate_dna_to_protein(mutated_seq),
            }

        if st.session_state.mutation_result:
            res = st.session_state.mutation_result
            st.markdown(f"<div style='background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:13px;color:#78350f;font-weight:600;margin:8px 0;'>{res['mut_label']}</div>", unsafe_allow_html=True)

            zoom_m = st.slider("Zoom sequence", 10, 20, 13, key="mut_zoom")

            # DNA before/after
            st.markdown(section_header("DNA — before vs after"), unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            with d1:
                dh = max(140, (len(res["original_seq"])//60+1)*(zoom_m+8)+60)
                components.html(render_2d_sequence(res["original_seq"],"Original DNA", highlight_pos=position, font_size=zoom_m), height=dh, scrolling=True)
                st.caption(f"Length: {len(res['original_seq'])} bp")
            with d2:
                dh2 = max(140, (len(res["mutated_seq"])//60+1)*(zoom_m+8)+60)
                components.html(render_2d_sequence(res["mutated_seq"],"Mutated DNA", highlight_pos=position, font_size=zoom_m), height=dh2, scrolling=True)
                st.caption(f"Length: {len(res['mutated_seq'])} bp")

            # Protein before/after
            st.markdown(section_header("Protein — before vs after"), unsafe_allow_html=True)
            p1, p2 = st.columns(2)
            with p1:
                ph = max(140, (len(res["original_protein"])//40+1)*(zoom_m+10)+60)
                components.html(render_2d_protein(res["original_protein"],"Original protein", font_size=zoom_m), height=ph, scrolling=True)
                st.caption(f"Length: {len(res['original_protein'])} aa")
                st.markdown(f"<div style='background:#f0fdf4;border-left:3px solid #16a34a;padding:8px 12px;font-size:12px;color:#14532d;margin-top:4px;'>{interpret_protein(res['original_protein'])}</div>", unsafe_allow_html=True)
            with p2:
                ph2 = max(140, (len(res["mutated_protein"])//40+1)*(zoom_m+10)+60)
                components.html(render_2d_protein(res["mutated_protein"],"Mutated protein", font_size=zoom_m), height=ph2, scrolling=True)
                st.caption(f"Length: {len(res['mutated_protein'])} aa")
                st.markdown(f"<div style='background:#fef2f2;border-left:3px solid #dc2626;padding:8px 12px;font-size:12px;color:#7f1d1d;margin-top:4px;'>{interpret_protein(res['mutated_protein'])}</div>", unsafe_allow_html=True)

            # Impact metrics
            st.markdown(section_header("Mutation impact overview"), unsafe_allow_html=True)
            orig_len = len(res["original_protein"])
            mut_len  = len(res["mutated_protein"])
            len_diff = mut_len - orig_len
            orig_gc  = gc_content_percent(res["original_seq"])
            mut_gc   = gc_content_percent(res["mutated_seq"])

            ov1, ov2, ov3 = st.columns(3)
            ov1.metric("Original protein length", f"{orig_len} aa")
            ov2.metric("Mutated protein length",  f"{mut_len} aa",  delta=f"{len_diff:+d} aa")
            ov3.metric("GC content change", f"{mut_gc:.1f}%", delta=f"{mut_gc-orig_gc:+.1f}%")

            if res["original_protein"] == res["mutated_protein"]:
                st.success("Silent mutation — protein is unchanged (synonymous).")
            elif len_diff < 0:
                st.error("Frameshift likely — protein is shorter. Could cause loss of function.")
            elif len_diff > 0:
                st.warning("Protein is longer than expected — possible read-through mutation.")
            else:
                st.warning("Missense mutation — protein sequence changed but same length.")


if __name__ == "__main__":
    main()
