import subprocess
import sys

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


def main():
    st.set_page_config(page_title="GeneScope Biotech Dashboard", layout="wide")

    # Global styling / Biotech dashboard theme
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        /* Metric cards - dark techy cards */
        div[data-testid="metric-container"] {
            background-color: #111827;
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.45);
            border: 1px solid #1f2937;
        }
        div[data-testid="metric-container"] label {
            color: #9ca3af;
        }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #f9fafb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header styled loosely after UniProt entry page
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 0.5rem;">
          <div>
            <h1 style="color:#1f4e79; margin-bottom:0;">GeneScope Biotech Dashboard</h1>
            <p style="color:#6c757d; font-size:0.95rem; margin-top:0.25rem;">
              Explore gene sequences, composition, protein function, and clinical variation.
            </p>
          </div>
          <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="
              display:inline-flex;
              align-items:center;
              gap:0.35rem;
              background:linear-gradient(135deg,#fbbf24,#f59e0b);
              color:#111827;
              padding:0.25rem 0.6rem;
              border-radius:999px;
              font-size:0.8rem;
              font-weight:600;
              box-shadow:0 0 0 1px rgba(180,83,9,0.4);
            ">
              <span style="width:0.55rem;height:0.55rem;border-radius:999px;background:#fef3c7;border:1px solid rgba(120,53,15,0.7);"></span>
              Status: Reviewed
            </span>
          </div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    genes = load_genes()
    gene_ids = [g["gene"] for g in genes]

    # Sidebar: gene selection + project info
    with st.sidebar:
        st.header("Gene Selection")
        selected_id = st.selectbox("Select a gene (searchable)", options=gene_ids, index=0)

        st.markdown("---")
        st.subheader("Project Info")
        st.write(
            "This Biotech Dashboard focuses on GC content, nucleotide composition, "
            "and protein translation for a curated panel of well-known human genes."
        )

    # Loading animation when (re)computing for a selected gene
    with st.spinner("Loading gene profile..."):
        selected_gene = next(g for g in genes if g["gene"] == selected_id)
        seq = normalize_seq(selected_gene["sequence"])
        gc = gc_content_percent(seq)
        mw = molecular_weight_dna(seq)
        tm_empirical = melting_temperature_tm(seq)
        tm_wallace_value = wallace_tm(seq)
        pdb_id = selected_gene.get("pdb_id", "")

    # Summary section with dark "cards"
    st.markdown("### Summary")
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gene ID", selected_gene["gene"])
        with col2:
            st.metric("Category", selected_gene.get("category", "N/A"))
        with col3:
            st.metric("GC Content", f"{gc:.2f} %")
        with col4:
            st.metric("Length (bp)", f"{len(seq)}")

        st.markdown("#### Scientific Metrics")
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Molecular Weight (g/mol)", f"{mw:,.0f}")
        with col6:
            st.metric("Tm (Wallace rule, °C)", f"{tm_wallace_value:.1f}")
        with col7:
            st.metric("Tm (empirical, °C)", f"{tm_empirical:.1f}")

    # Tabs for different views
    tab_seq, tab_viz, tab_trans = st.tabs(
        ["🧬 Sequence Analysis", "📊 Visualization", "🧪 Translation"]
    )

    with tab_seq:
        st.markdown("#### Entry overview")
        if selected_gene.get("protein_name"):
            st.markdown(f"**Protein name:** {selected_gene['protein_name']}")
        if selected_gene.get("disease"):
            st.markdown(f"**Associated disease:** {selected_gene['disease']}")
        if selected_gene.get("subcellular_location"):
            st.markdown(f"**Subcellular location:** {selected_gene['subcellular_location']}")

        st.markdown("#### Function")
        if selected_gene.get("go_function"):
            st.markdown(selected_gene["go_function"])
        else:
            st.markdown(selected_gene.get("description", ""))

        st.markdown("#### DNA Sequence")
        if selected_gene.get("refseq_mrna"):
            st.caption(
                f"Sequence source: RefSeq {selected_gene['refseq_mrna']} ({selected_gene.get('sequence_note','').strip()})"
            )
        st.code(seq, language="text")
        st.write(f"Length: **{len(seq)}** bases")

        # Pathology & Biotech table (variants)
        variants = selected_gene.get("variants", []) or []
        if variants:
            st.markdown("#### Pathology & Biotech")
            df = pd.DataFrame(variants)
            # Reorder columns if present
            cols = [c for c in ["variant", "significance", "condition", "note"] if c in df.columns]
            df = df[cols]
            st.dataframe(df, width='stretch')

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

            # GC gauge chart
            st.markdown("##### GC Content Gauge")
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
                                {"range": [0, 40], "color": "rgba(37, 99, 235, 0.4)"},
                                {"range": [40, 60], "color": "rgba(34, 197, 94, 0.4)"},
                                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.4)"},
                            ],
                            "threshold": {
                                "line": {"color": gauge_color, "width": 4},
                                "thickness": 0.75,
                                "value": gc,
                            },
                        },
                    )
                )
                gauge_fig.update_layout(template="plotly_dark", height=260)
                st.plotly_chart(gauge_fig, width="stretch")

            # Donut chart of nucleotide percentages
            st.markdown("##### Nucleotide Composition Donut")
            if total > 0:
                percentages = (counts / total * 100).round(2)
                donut_fig = px.pie(
                    values=percentages.values,
                    names=percentages.index,
                    hole=0.65,
                    color_discrete_sequence=px.colors.sequential.Viridis,
                )
                donut_fig.update_layout(
                    title="Nucleotide Composition (A / C / G / T)",
                    template="plotly_dark",
                    showlegend=True,
                )
                st.plotly_chart(donut_fig, width="stretch")

            # Optional: retain GC density heatmap for a lab-panel feel
            st.markdown("##### GC Density Heatmap (10 × 10)")
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

                heatmap_fig = px.imshow(
                    gc_matrix,
                    color_continuous_scale="Viridis",
                    origin="upper",
                    labels={"color": "GC %"},
                )
                heatmap_fig.update_layout(
                    title="GC Density Heatmap (higher = more G/C)",
                    xaxis_title="Segment column",
                    yaxis_title="Segment row",
                    template="plotly_dark",
                )
                st.plotly_chart(heatmap_fig, width="stretch")

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
            st.plotly_chart(cons_fig, width="stretch")

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

                mismatch_fig = px.imshow(
                    mismatch_matrix,
                    color_continuous_scale=[(0.0, "red"), (1.0, "#1f2937")],
                    origin="upper",
                    labels={"color": "Match"},
                    aspect="auto",
                )
                mismatch_fig.update_coloraxes(
                    cmin=0,
                    cmax=1,
                    showscale=False,
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
                mismatch_fig.update_layout(
                    title="Sequence Mismatches (red = mismatch, dark = match)",
                    template="plotly_dark",
                )
                st.plotly_chart(mismatch_fig, width="stretch")
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
        st.markdown("#### Protein Translation")
        if st.button("Translate to Protein"):
            protein = translate_dna_to_protein(seq)
            st.code(protein, language="text")
            hydrophobicity = average_hydrophobicity(protein)
            st.metric("Average Hydrophobicity (Kyte-Doolittle)", f"{hydrophobicity:.2f}")
            if px is not None:
                aa_series = pd.Series(list(protein))
                aa_freq = (
                    aa_series.value_counts(normalize=True).sort_index() * 100.0
                ).round(2)
                freq_df = aa_freq.reset_index()
                freq_df.columns = ["Amino Acid", "Percent"]
                freq_fig = px.bar(
                    freq_df,
                    x="Percent",
                    y="Amino Acid",
                    orientation="h",
                    title="Amino Acid Frequency (%)",
                    template="plotly_dark",
                )
                st.plotly_chart(freq_fig, width="stretch")
            st.markdown("#### 3D Structure (Ribbon / Cartoon)")
            target_pdb = pdb_id or "4HHB"
            show_3d_protein(target_pdb)
        else:
            st.info("Click 'Translate to Protein' to generate the amino acid sequence and its hydrophobicity.")


if __name__ == "__main__":
    main()

