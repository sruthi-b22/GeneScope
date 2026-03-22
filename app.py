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
    np = None; px = None; go = None

CONSERVATION = {
    # Your original 5
    "HBB":   {"Pan troglodytes": 0.99, "Mus musculus": 0.87},
    "BRCA1": {"Pan troglodytes": 0.99, "Mus musculus": 0.56},
    "HTT":   {"Pan troglodytes": 0.99, "Mus musculus": 0.90},
    "CFTR":  {"Pan troglodytes": 0.98, "Mus musculus": 0.78},
    "TP53":  {"Pan troglodytes": 0.99, "Mus musculus": 0.80},
    # Disease map genes
    "PAH":    {"Pan troglodytes": 0.99, "Mus musculus": 0.82},
    "DYRK1A": {"Pan troglodytes": 0.99, "Mus musculus": 0.96},
    "APP":    {"Pan troglodytes": 0.99, "Mus musculus": 0.97},
    "PARK2":  {"Pan troglodytes": 0.99, "Mus musculus": 0.84},
    "FBN1":   {"Pan troglodytes": 0.99, "Mus musculus": 0.84},
    "GBA":    {"Pan troglodytes": 0.99, "Mus musculus": 0.88},
    "HEXA":   {"Pan troglodytes": 0.98, "Mus musculus": 0.83},
    "ATP7B":  {"Pan troglodytes": 0.99, "Mus musculus": 0.85},
    "F8":     {"Pan troglodytes": 0.97, "Mus musculus": 0.74},
    "F9":     {"Pan troglodytes": 0.97, "Mus musculus": 0.77},
    "MLH1":   {"Pan troglodytes": 0.99, "Mus musculus": 0.85},
    "RB1":    {"Pan troglodytes": 0.99, "Mus musculus": 0.91},
    "EGFR":   {"Pan troglodytes": 0.99, "Mus musculus": 0.93},
    "INS":    {"Pan troglodytes": 0.97, "Mus musculus": 0.87},
    "DMD":    {"Pan troglodytes": 0.99, "Mus musculus": 0.90},
    "FMR1":   {"Pan troglodytes": 0.99, "Mus musculus": 0.97},
    "MECP2":  {"Pan troglodytes": 0.99, "Mus musculus": 0.96},
    "NF1":    {"Pan troglodytes": 0.99, "Mus musculus": 0.98},
    "TSC1":   {"Pan troglodytes": 0.99, "Mus musculus": 0.91},
    "SMN1":   {"Pan troglodytes": 0.99, "Mus musculus": 0.82},
    "FGFR3":  {"Pan troglodytes": 0.99, "Mus musculus": 0.98},
    "COL5A1": {"Pan troglodytes": 0.99, "Mus musculus": 0.90},
    "COL1A1": {"Pan troglodytes": 0.99, "Mus musculus": 0.90},
    "FANCA":  {"Pan troglodytes": 0.98, "Mus musculus": 0.72},
    "ATM":    {"Pan troglodytes": 0.99, "Mus musculus": 0.84},
    "BLM":    {"Pan troglodytes": 0.99, "Mus musculus": 0.73},
    "XPA":    {"Pan troglodytes": 0.99, "Mus musculus": 0.72},
    "WRN":    {"Pan troglodytes": 0.99, "Mus musculus": 0.75},
    "BRCA2":  {"Pan troglodytes": 0.99, "Mus musculus": 0.59},
    "KRAS":   {"Pan troglodytes": 1.00, "Mus musculus": 1.00},
    "PTEN":   {"Pan troglodytes": 1.00, "Mus musculus": 1.00},
}

DISEASE_GENE_MAP = {
    "down syndrome": "DYRK1A", "down's syndrome": "DYRK1A", "trisomy 21": "DYRK1A",
    "alzheimer": "APP", "alzheimer's": "APP", "alzheimer's disease": "APP", "alzheimers": "APP",
    "parkinson": "PARK2", "parkinson's": "PARK2", "parkinson's disease": "PARK2", "parkinsons": "PARK2",
    "huntington": "HTT", "huntington's": "HTT", "huntington's disease": "HTT",
    "huntington disease": "HTT", "huntingtons": "HTT",
    "marfan syndrome": "FBN1", "marfan's syndrome": "FBN1", "marfans": "FBN1", "marfan": "FBN1",
    "ehlers danlos": "COL5A1", "osteogenesis imperfecta": "COL1A1",
    "cystic fibrosis": "CFTR", "phenylketonuria": "PAH", "pku": "PAH",
    "gaucher disease": "GBA", "gaucher's disease": "GBA",
    "tay sachs": "HEXA", "tay-sachs": "HEXA",
    "wilson disease": "ATP7B", "wilson's disease": "ATP7B",
    "sickle cell": "HBB", "sickle cell anaemia": "HBB", "sickle cell anemia": "HBB",
    "sickle cell disease": "HBB", "hemophilia a": "F8", "haemophilia a": "F8", "hemophilia b": "F9",
    "thalassemia": "HBB", "beta thalassemia": "HBB",
    "breast cancer": "BRCA1", "hereditary breast cancer": "BRCA1", "ovarian cancer": "BRCA1",
    "colon cancer": "MLH1", "colorectal cancer": "MLH1", "lynch syndrome": "MLH1",
    "li-fraumeni": "TP53", "li fraumeni": "TP53",
    "retinoblastoma": "RB1", "lung cancer": "EGFR", "non-small cell lung cancer": "EGFR",
    "diabetes": "INS", "type 1 diabetes": "INS",
    "muscular dystrophy": "DMD", "duchenne": "DMD", "duchenne muscular dystrophy": "DMD",
    "fragile x": "FMR1", "fragile x syndrome": "FMR1",
    "rett syndrome": "MECP2", "neurofibromatosis": "NF1",
    "tuberous sclerosis": "TSC1", "spinal muscular atrophy": "SMN1", "sma": "SMN1",
    "achondroplasia": "FGFR3",
    "fanconi anemia": "FANCA", "fanconi anaemia": "FANCA", "fanconi": "FANCA",
    "ataxia telangiectasia": "ATM", "bloom syndrome": "BLM",
    "xeroderma pigmentosum": "XPA", "cockayne syndrome": "ERCC8",
    "werner syndrome": "WRN", "nijmegen breakage": "NBN",
}

from Bio import Entrez
Entrez.email = "sruthibalasubramani6@gmail.com"

def fetch_from_uniprot(gene_symbol):
    try:
        import urllib.request, json
        url = (f"https://rest.uniprot.org/uniprotkb/search?"
               f"query=gene:{gene_symbol}+AND+organism_id:9606+AND+reviewed:true"
               f"&fields=gene_names,protein_name,cc_subcellular_location,go_f&format=json&size=1")
        req = urllib.request.Request(url, headers={"User-Agent": "GeneScope/1.0 sruthibalasubramani6@gmail.com"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        if not data.get("results"): return None
        result = data["results"][0]
        protein_name = ""
        pn = result.get("proteinDescription", {})
        rec = pn.get("recommendedName", {})
        if rec: protein_name = rec.get("fullName", {}).get("value", "")
        if not protein_name:
            sub = pn.get("submissionNames", [])
            if sub: protein_name = sub[0].get("fullName", {}).get("value", "")
        location = ""
        for c in result.get("comments", []):
            if c.get("commentType") == "SUBCELLULAR LOCATION":
                locs = c.get("subcellularLocations", [])
                if locs: location = ", ".join([l.get("location", {}).get("value", "") for l in locs if l.get("location", {}).get("value")])
                break
        go_list = []
        for ref in result.get("uniProtKBCrossReferences", []):
            if ref.get("database") == "GO":
                for prop in ref.get("properties", []):
                    if prop.get("key") == "GoTerm" and prop.get("value", "").startswith("F:"):
                        go_list.append(prop["value"][2:])
        accession = result.get("primaryAccession", "")
        return {"protein_name": protein_name, "subcellular_location": location,
                "go_function": "; ".join(go_list[:3]), "accession": accession,
                "uniprot_url": f"https://www.uniprot.org/uniprotkb/{accession}"}
    except: return None

def fetch_pdb_for_gene(gene_symbol, uniprot_accession=""):
    import urllib.request, json
    KNOWN_PDB = {
        "TP53": "2OCJ", "BRCA1": "1JM7", "BRCA2": "1MJE", "EGFR": "1IVO",
        "KRAS": "4OBE", "HBB": "4HHB", "CFTR": "5UAK", "HTT": "4FE8",
        "APP": "1AAP", "PTEN": "1D5R", "RB1": "2AZE", "MLH1": "4P7A",
        "PARK2": "3CHR", "INS": "3I40", "PAH": "1PHZ", "FBN1": "2UHX",
        "DYRK1A": "2WO6", "DMD": "1DXX", "FMR1": "3RKU", "NF1": "3PG7",
        "MECP2": "3C2I", "TSC1": "3CH4", "SMN1": "1MFQ", "FGFR3": "4K33",
        "F8": "2R7E", "F9": "1CFH", "GBA": "2NSX", "HEXA": "2GSU",
        "ATP7B": "2ARF", "COL1A1": "1BKV", "COL5A1": "2V53", "ACTA1": "3EKS",
    }
    if gene_symbol.upper() in KNOWN_PDB:
        return KNOWN_PDB[gene_symbol.upper()]
    try:
        if uniprot_accession:
            try:
                url = f"https://data.rcsb.org/rest/v1/core/uniprot/{uniprot_accession}"
                req = urllib.request.Request(url, headers={"User-Agent": "GeneScope/1.0"})
                with urllib.request.urlopen(req, timeout=6) as r:
                    data = json.loads(r.read().decode())
                if data and isinstance(data, list) and len(data) > 0:
                    pdb = data[0].get("rcsb_id", "")
                    if pdb: return pdb.split("_")[0]
            except: pass
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        query = {
            "query": {"type": "group", "logical_operator": "and", "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                    "operator": "exact_match", "value": "Homo sapiens"}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_polymer_entity.pdbx_gene_src_gene",
                    "operator": "contains_words", "value": gene_symbol}}
            ]},
            "return_type": "entry",
            "request_options": {"results_slice": {"start": 0, "limit": 1},
                                 "sort": [{"sort_by": "score", "direction": "descending"}]}
        }
        req = urllib.request.Request(search_url, data=json.dumps(query).encode(),
              headers={"Content-Type": "application/json", "User-Agent": "GeneScope/1.0"}, method="POST")
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        results = data.get("result_set", [])
        if results: return results[0]["identifier"]
    except: pass
    return None

def fetch_from_ncbi(query):
    try:
        query_lower = query.lower().strip()
        if query_lower in DISEASE_GENE_MAP:
            query = DISEASE_GENE_MAP[query_lower]
        handle = Entrez.esearch(db="gene", term=f"{query}[Gene Name] AND Homo sapiens[Organism]", retmax=1)
        record = Entrez.read(handle); handle.close()
        if not record["IdList"]:
            handle = Entrez.esearch(db="gene",
                term=f"{query}[Title] AND Homo sapiens[Organism] AND protein coding[Gene Type]", retmax=1)
            record = Entrez.read(handle); handle.close()
        if not record["IdList"]:
            handle = Entrez.esearch(db="gene", term=f"{query} AND Homo sapiens[Organism]", retmax=1)
            record = Entrez.read(handle); handle.close()
        if not record["IdList"]: return None, "Gene not found on NCBI"
        gene_id = record["IdList"][0]
        handle = Entrez.esummary(db="gene", id=gene_id)
        summary = Entrez.read(handle); handle.close()
        info = summary["DocumentSummarySet"]["DocumentSummary"][0]
        gene_symbol  = str(info["Name"]); full_name = str(info["Description"])
        aliases      = str(info.get("OtherAliases", "—")); summary_text = str(info.get("Summary", "No summary available."))
        chromosome   = str(info.get("Chromosome", "—")); location = str(info.get("MapLocation", "—"))
        seq_handle = Entrez.esearch(db="nucleotide",
            term=f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism] AND mRNA[Filter] AND RefSeq[Filter]", retmax=1)
        seq_record = Entrez.read(seq_handle); seq_handle.close()
        dna_sequence = ""; refseq_id = ""
        if seq_record["IdList"]:
            nuc_id = seq_record["IdList"][0]
            fh = Entrez.efetch(db="nucleotide", id=nuc_id, rettype="fasta", retmode="text")
            fasta = fh.read(); fh.close()
            lines = fasta.strip().split("\n")
            if lines:
                header = lines[0]
                refseq_id = header.split("|")[1] if "|" in header and len(header.split("|")) > 1 else header.split()[0].replace(">", "")
            dna_sequence = "".join(lines[1:])[:500]
        uniprot_data = fetch_from_uniprot(gene_symbol)
        uniprot_acc  = uniprot_data["accession"] if uniprot_data else ""
        pdb          = fetch_pdb_for_gene(gene_symbol, uniprot_acc)
        return {"name": gene_symbol, "full_name": full_name, "aliases": aliases, "summary": summary_text,
                "chromosome": chromosome, "location": location, "ncbi_id": gene_id,
                "sequence": dna_sequence, "refseq_id": refseq_id,
                "protein_name":         uniprot_data["protein_name"]         if uniprot_data else full_name,
                "subcellular_location": uniprot_data["subcellular_location"]  if uniprot_data else "—",
                "go_function":          uniprot_data["go_function"]           if uniprot_data else "—",
                "uniprot_accession":    uniprot_acc,
                "uniprot_url":          uniprot_data["uniprot_url"]           if uniprot_data else "",
                "pdb_id":               pdb or ""}, None
    except ImportError: return None, "biopython not installed — add it to requirements.txt"
    except Exception as e: return None, f"NCBI error: {str(e)}"

def load_genes():
    genes = []
    if not GENE_DB:
        st.error("No genes found! Using demo mode.")
        return [{"gene": "DEMO", "sequence": "ATGC", "pdb_id": ""}]
    for symbol, meta in GENE_DB.items():
        genes.append({"gene": symbol, "category": meta.get("category", "N/A"),
            "disease": meta.get("disease", "N/A"), "description": meta.get("description", ""),
            "sequence": meta.get("sequence", ""), "refseq_mrna": meta.get("refseq_mrna", ""),
            "sequence_note": meta.get("sequence_note", ""), "protein_name": meta.get("protein_name", ""),
            "go_function": meta.get("go_function", ""), "subcellular_location": meta.get("subcellular_location", ""),
            "variants": meta.get("variants", []), "pdb_id": meta.get("pdb_id", "")})
    return genes

def normalize_seq(seq): return "".join(str(seq or "").upper().split())
def gc_content_percent(s):
    seq = normalize_seq(s); return 0.0 if not seq else sum(1 for c in seq if c in "GC") / len(seq) * 100
def molecular_weight_dna(s): return 330.0 * len(normalize_seq(s))
def melting_temperature_tm(s):
    seq = normalize_seq(s); N = len(seq)
    if not N: return 0.0
    return 64.9 + 41.0 * (sum(1 for c in seq if c in "GC") - 16.4) / N
def wallace_tm(s):
    seq = normalize_seq(s)
    return 0.0 if not seq else 2.0 * (seq.count("A") + seq.count("T")) + 4.0 * (seq.count("G") + seq.count("C"))

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
def transcribe_dna_to_rna(s): return normalize_seq(s).replace("T", "U")
def translate_dna_to_protein(s, stop_at_stop=True):
    rna = transcribe_dna_to_rna(s)
    if not rna: return ""
    p = []
    for i in range(0, len(rna) - 2, 3):
        aa = CODON_TABLE.get(rna[i:i+3], "X")
        if aa == "*":
            if stop_at_stop: break
            p.append("*")
        else: p.append(aa)
    return "".join(p)
def average_hydrophobicity(protein):
    if not protein: return 0.0
    kd = {"I":4.5,"V":4.2,"L":3.8,"F":2.8,"C":2.5,"M":1.9,"A":1.8,"G":-0.4,"T":-0.7,
          "S":-0.8,"W":-0.9,"Y":-1.3,"P":-1.6,"H":-3.2,"E":-3.5,"Q":-3.5,"D":-3.5,"N":-3.5,"K":-3.9,"R":-4.5}
    vals = [kd[aa] for aa in protein if aa in kd]
    return sum(vals) / len(vals) if vals else 0.0
def amino_acid_composition(s): return dict(Counter(s))
def interpret_gc(gc):
    if gc < 40: return "Low GC content — DNA may be less stable and easier to denature."
    elif gc <= 60: return "Moderate GC content — typical for many organisms."
    else: return "High GC content — DNA is more stable due to stronger bonding."
def interpret_protein(p):
    if "*" in p: return "Stop codon detected — translation terminated."
    elif len(p) < 20: return "Short peptide — may not form a functional protein."
    else: return "Protein sequence generated — potential functional molecule."

def render_2d_sequence(seq, label="DNA Sequence", highlight_pos=None, font_size=13):
    BC = {"A": "#16a34a", "T": "#dc2626", "G": "#2563eb", "C": "#d97706"}
    chunk = 60; rows = [seq[i:i+chunk] for i in range(0, len(seq), chunk)]
    html = (f"<div style='font-family:monospace;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px;'>"
            f"<div style='font-size:11px;font-weight:600;color:#8898b3;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>{label} — {len(seq)} bp</div>")
    for ri, row in enumerate(rows):
        start = ri * chunk
        html += f"<div style='display:flex;align-items:center;margin-bottom:3px;'><span style='font-size:10px;color:#94a3b8;min-width:38px;margin-right:6px;'>{start+1}</span>"
        for i, base in enumerate(row):
            pos = start + i + 1; color = BC.get(base, "#0a2540")
            bg = "#fef08a" if (highlight_pos and pos == highlight_pos) else "transparent"
            html += (f"<span title='Position {pos}: {base}' style='display:inline-block;width:{font_size+1}px;height:{font_size+4}px;"
                     f"text-align:center;font-size:{font_size}px;font-weight:600;color:{color};background:{bg};border-radius:2px;cursor:default;line-height:{font_size+4}px;' "
                     f"onmouseover=\"this.style.background='#dbeafe';this.style.outline='1px solid #93c5fd'\" "
                     f"onmouseout=\"this.style.background='{bg}';this.style.outline='none'\">{base}</span>")
        html += "</div>"
    html += "</div>"; return html

def render_2d_protein(protein, label="Protein Sequence", highlight_pos=None, font_size=13):
    AC = {"A":"#6366f1","R":"#ef4444","N":"#f97316","D":"#ef4444","C":"#eab308","Q":"#f97316","E":"#ef4444",
          "G":"#8b5cf6","H":"#06b6d4","I":"#10b981","L":"#10b981","K":"#ef4444","M":"#eab308","F":"#8b5cf6",
          "P":"#f97316","S":"#06b6d4","T":"#06b6d4","W":"#8b5cf6","Y":"#8b5cf6","V":"#10b981"}
    chunk = 40; rows = [protein[i:i+chunk] for i in range(0, len(protein), chunk)]
    html = (f"<div style='font-family:monospace;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px;'>"
            f"<div style='font-size:11px;font-weight:600;color:#8898b3;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>{label} — {len(protein)} aa</div>")
    for ri, row in enumerate(rows):
        start = ri * chunk
        html += f"<div style='display:flex;align-items:center;margin-bottom:4px;'><span style='font-size:10px;color:#94a3b8;min-width:38px;margin-right:6px;'>{start+1}</span>"
        for i, aa in enumerate(row):
            pos = start + i + 1; color = AC.get(aa, "#0a2540")
            bg = "#fef08a" if (highlight_pos and pos == highlight_pos) else "#ffffff"
            html += (f"<span title='Position {pos}: {aa}' style='display:inline-block;width:{font_size+5}px;height:{font_size+6}px;"
                     f"text-align:center;font-size:{font_size}px;font-weight:700;color:{color};background:{bg};border-radius:3px;cursor:default;line-height:{font_size+6}px;border:1px solid #e2e8f0;margin:1px;' "
                     f"onmouseover=\"this.style.background='#dbeafe';this.style.borderColor='#93c5fd'\" "
                     f"onmouseout=\"this.style.background='{bg}';this.style.borderColor='#e2e8f0'\">{aa}</span>")
        html += "</div>"
    html += "</div>"; return html

def show_3d_protein(pdb_id):
    st.markdown(
        f"<div style='font-size:11px;font-weight:600;color:#8898b3;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;'>3D Structure — PDB: {pdb_id}</div>"
        f"<div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;font-size:12px;color:#475569;'>"
        f"<span style='color:#64748b;font-weight:600;'>Legend:</span>"
        f"<span><span style='color:#3b82f6;font-weight:700;'>■</span> N-terminus (blue)</span>"
        f"<span><span style='color:#22c55e;font-weight:700;'>■</span> Middle (green)</span>"
        f"<span><span style='color:#ef4444;font-weight:700;'>■</span> C-terminus (red)</span>"
        f"<span style='color:#94a3b8;'>| Scroll to zoom · Drag to rotate</span></div>",
        unsafe_allow_html=True)
    html_content = f"""
    <div id="viewport" style="width:100%;height:500px;background:#1a1a2e;border-radius:10px;position:relative;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
    <script>
        function load3D() {{
            var element = document.getElementById('viewport');
            if (!element) {{ setTimeout(load3D, 200); return; }}
            var viewer = $3Dmol.createViewer(element, {{backgroundColor: '#1a1a2e'}});
            $3Dmol.download("pdb:{pdb_id}", viewer, {{}}, function() {{
                viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.12, color: 'white'}});
                viewer.zoomTo(); viewer.render(); viewer.zoom(0.9);
            }});
            viewer.setHoverable({{}}, true,
                function(atom, viewer) {{
                    if (!atom.label) {{
                        atom.label = viewer.addLabel(atom.resn+":"+atom.resi,
                            {{position:atom,backgroundColor:'white',fontColor:'#0a2540',fontSize:12,backgroundOpacity:0.9}});
                    }}
                }},
                function(atom, viewer) {{
                    if (atom.label) {{ viewer.removeLabel(atom.label); delete atom.label; }}
                }}
            );
        }}
        load3D();
    </script>
    """
    components.html(html_content, height=520, scrolling=False)
    st.caption(f"PDB: {pdb_id} · Scroll to zoom · Drag to rotate · Right-click to pan")
    st.markdown(f"[View on RCSB PDB ↗](https://www.rcsb.org/structure/{pdb_id})")

# ── FIX: no st.rerun() — button click already triggers rerun naturally ────────
def render_3d_button(pdb_id, key_prefix):
    loaded_key = f"{key_prefix}_3d_loaded"
    pdb_key    = f"{key_prefix}_3d_pdb"
    if loaded_key not in st.session_state: st.session_state[loaded_key] = False
    if pdb_key not in st.session_state:    st.session_state[pdb_key] = ""
    if st.session_state[pdb_key] != pdb_id:
        st.session_state[loaded_key] = False
        st.session_state[pdb_key] = pdb_id
    if st.session_state[loaded_key]:
        show_3d_protein(pdb_id)
    else:
        if st.button("Load 3D structure", key=f"{key_prefix}_3d_btn"):
            st.session_state[loaded_key] = True
            # No st.rerun() — the button click itself triggers the rerun
        st.caption(f"PDB ID: {pdb_id} — click to load · scroll to zoom · drag to rotate")

def metric_card(label, value, sub=""):
    s = f"<div style='font-size:12px;color:#8898b3;margin-top:4px;'>{sub}</div>" if sub else ""
    return (f"<div style='background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 18px;'>"
            f"<div style='font-size:11px;color:#8898b3;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;margin-bottom:6px;'>{label}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:#0a2540;letter-spacing:-0.5px;'>{value}</div>{s}</div>")

def section_header(title):
    return (f"<div style='font-size:13px;font-weight:700;color:#0a2540;text-transform:uppercase;"
            f"letter-spacing:0.5px;margin:1.2rem 0 0.8rem;padding-bottom:8px;border-bottom:1px solid #e2e8f0;'>{title}</div>")

def render_mutation_simulator(seq, pdb_id, key_prefix="local"):
    st.markdown(section_header("Mutation simulator"), unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        mut_type = st.selectbox("Mutation type",
            ["Substitution (change a base)", "Deletion (remove a base)", "Insertion (add a base)"],
            key=f"{key_prefix}_mut_type")
    with m2:
        position = st.number_input("Position (1-based)", min_value=1, max_value=len(seq), key=f"{key_prefix}_mut_pos")
    with m3:
        if mut_type != "Deletion (remove a base)":
            new_base = st.selectbox("New base", ["A", "T", "G", "C"], key=f"{key_prefix}_mut_base")
        else:
            new_base = None
            st.markdown("<div style='padding-top:28px;color:#8898b3;font-size:13px;'>No base needed for deletion</div>", unsafe_allow_html=True)
    mr_key = f"{key_prefix}_mutation_result"
    if mr_key not in st.session_state: st.session_state[mr_key] = None
    if st.button("Apply mutation", key=f"{key_prefix}_apply_mut_btn"):
        ob = seq[position - 1]
        if mut_type == "Substitution (change a base)":
            mutated_seq = seq[:position-1] + new_base + seq[position:]
            mut_label = f"Substitution: position {position}  {ob} → {new_base}"
        elif mut_type == "Deletion (remove a base)":
            mutated_seq = seq[:position-1] + seq[position:]
            mut_label = f"Deletion: removed {ob} at position {position}"
        else:
            mutated_seq = seq[:position-1] + new_base + seq[position-1:]
            mut_label = f"Insertion: added {new_base} before position {position}"
        st.session_state[mr_key] = {
            "mut_label": mut_label, "original_seq": seq, "mutated_seq": mutated_seq,
            "original_protein": translate_dna_to_protein(seq),
            "mutated_protein": translate_dna_to_protein(mutated_seq),
        }
    if st.session_state[mr_key]:
        res = st.session_state[mr_key]
        st.markdown(f"<div style='background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:13px;color:#78350f;font-weight:600;margin:8px 0;'>{res['mut_label']}</div>", unsafe_allow_html=True)
        zoom_m = st.slider("Zoom sequence", 10, 20, 13, key=f"{key_prefix}_mut_zoom")
        st.markdown(section_header("DNA — before vs after"), unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            dh = max(140, (len(res["original_seq"]) // 60 + 1) * (zoom_m + 8) + 60)
            components.html(render_2d_sequence(res["original_seq"], "Original DNA", highlight_pos=position, font_size=zoom_m), height=dh, scrolling=True)
            st.caption(f"Length: {len(res['original_seq'])} bp")
        with d2:
            dh2 = max(140, (len(res["mutated_seq"]) // 60 + 1) * (zoom_m + 8) + 60)
            components.html(render_2d_sequence(res["mutated_seq"], "Mutated DNA", highlight_pos=position, font_size=zoom_m), height=dh2, scrolling=True)
            st.caption(f"Length: {len(res['mutated_seq'])} bp")
        st.markdown(section_header("Protein — before vs after"), unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            ph = max(140, (len(res["original_protein"]) // 40 + 1) * (zoom_m + 10) + 60)
            components.html(render_2d_protein(res["original_protein"], "Original protein", font_size=zoom_m), height=ph, scrolling=True)
            st.caption(f"Length: {len(res['original_protein'])} aa")
            st.markdown(f"<div style='background:#f0fdf4;border-left:3px solid #16a34a;padding:8px 12px;font-size:12px;color:#14532d;margin-top:4px;'>{interpret_protein(res['original_protein'])}</div>", unsafe_allow_html=True)
        with p2:
            ph2 = max(140, (len(res["mutated_protein"]) // 40 + 1) * (zoom_m + 10) + 60)
            components.html(render_2d_protein(res["mutated_protein"], "Mutated protein", font_size=zoom_m), height=ph2, scrolling=True)
            st.caption(f"Length: {len(res['mutated_protein'])} aa")
            st.markdown(f"<div style='background:#fef2f2;border-left:3px solid #dc2626;padding:8px 12px;font-size:12px;color:#7f1d1d;margin-top:4px;'>{interpret_protein(res['mutated_protein'])}</div>", unsafe_allow_html=True)
        if pdb_id:
            st.markdown(section_header("3D structure — before vs after mutation"), unsafe_allow_html=True)
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;font-size:12px;color:#14532d;font-weight:600;margin-bottom:8px;'>Original protein — 3D structure (PDB)</div>", unsafe_allow_html=True)
                show_3d_protein(pdb_id)
            with t2:
                st.markdown("<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 12px;font-size:12px;color:#7f1d1d;font-weight:600;margin-bottom:8px;'>Mutated protein — 2D sequence viewer</div>", unsafe_allow_html=True)
                ph_mut = max(200, (len(res["mutated_protein"]) // 40 + 1) * (zoom_m + 10) + 80)
                components.html(render_2d_protein(res["mutated_protein"], "Mutated protein — full sequence", font_size=zoom_m), height=ph_mut, scrolling=True)
                st.markdown("<div style='background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 12px;font-size:12px;color:#78350f;margin-top:8px;'><strong>Why no 3D for mutated?</strong><br>Custom mutations create novel sequences that don't exist in PDB. In real research, tools like AlphaFold2 predict the 3D structure of mutated sequences. The 2D viewer above shows all amino acid changes.</div>", unsafe_allow_html=True)
        st.markdown(section_header("Mutation impact overview"), unsafe_allow_html=True)
        orig_len = len(res["original_protein"]); mut_len = len(res["mutated_protein"]); len_diff = mut_len - orig_len
        orig_gc = gc_content_percent(res["original_seq"]); mut_gc = gc_content_percent(res["mutated_seq"])
        ov1, ov2, ov3 = st.columns(3)
        ov1.metric("Original protein length", f"{orig_len} aa")
        ov2.metric("Mutated protein length", f"{mut_len} aa", delta=f"{len_diff:+d} aa")
        ov3.metric("GC content change", f"{mut_gc:.1f}%", delta=f"{mut_gc - orig_gc:+.1f}%")
        if res["original_protein"] == res["mutated_protein"]:
            st.success("Silent mutation (synonymous) — protein unchanged.")
        elif len_diff < 0:
            st.error(f"Frameshift mutation — protein shorter by {abs(len_diff)} aa. Likely loss of function.")
        elif len_diff > 0:
            st.warning(f"Read-through mutation — protein longer by {len_diff} aa.")
        else:
            diffs = [(i+1, res["original_protein"][i], res["mutated_protein"][i])
                     for i in range(min(len(res["original_protein"]), len(res["mutated_protein"])))
                     if res["original_protein"][i] != res["mutated_protein"][i]]
            diff_str = ", ".join([f"pos {p}: {o}→{m}" for p, o, m in diffs[:5]])
            st.warning(f"Missense mutation — same length, changed amino acids: {diff_str if diff_str else 'see 2D viewer above'}.")

def render_translation(seq, key_prefix="local"):
    st.markdown(section_header("Protein translation"), unsafe_allow_html=True)
    prot_key = f"{key_prefix}_protein"
    if prot_key not in st.session_state: st.session_state[prot_key] = None
    if st.button("Translate to protein", key=f"{key_prefix}_translate_btn"):
        st.session_state[prot_key] = translate_dna_to_protein(seq)
    if st.session_state[prot_key]:
        prot = st.session_state[prot_key]
        zoom_t = st.slider("Zoom", 10, 20, 13, key=f"{key_prefix}_prot_zoom")
        st.markdown("**2D DNA sequence viewer** — hover to see position & base")
        dna_h = max(140, (len(seq) // 60 + 1) * (zoom_t + 8) + 60)
        components.html(render_2d_sequence(seq, "Original DNA", font_size=zoom_t), height=dna_h, scrolling=True)
        st.markdown("**2D Protein sequence viewer** — hover to see position & amino acid")
        pro_h = max(140, (len(prot) // 40 + 1) * (zoom_t + 10) + 60)
        components.html(render_2d_protein(prot, "Translated protein", font_size=zoom_t), height=pro_h, scrolling=True)
        st.markdown(f"<div style='background:#eff6ff;border-left:3px solid #2563eb;border-radius:0;padding:10px 14px;font-size:13px;color:#1e40af;margin:8px 0;'>{interpret_protein(prot)}</div>", unsafe_allow_html=True)
        aa_c, hyd_c = st.columns(2)
        with aa_c:
            st.markdown("**Amino acid composition**")
            st.bar_chart(amino_acid_composition(prot))
        with hyd_c:
            hydro = average_hydrophobicity(prot)
            st.markdown(f"<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 18px;margin-top:24px;'><div style='font-size:11px;color:#8898b3;text-transform:uppercase;letter-spacing:0.5px;font-weight:600;margin-bottom:6px;'>Hydrophobicity (Kyte-Doolittle)</div><div style='font-size:2rem;font-weight:700;color:#0a2540;'>{hydro:.2f}</div><div style='font-size:12px;color:#8898b3;margin-top:4px;'>{'Hydrophobic — likely membrane-associated' if hydro>0 else 'Hydrophilic — likely soluble/cytoplasmic'}</div></div>", unsafe_allow_html=True)

def render_visualization(seq, gc, selected_gene_name, description, species_map=None):
    if px is None or np is None or go is None:
        st.warning("Plotly not installed."); return
    series = pd.Series(list(seq)); counts = series.value_counts().reindex(["A","C","G","T"]).fillna(0); total = counts.sum()
    st.markdown(section_header("Nucleotide composition"), unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        if total > 0:
            gc_label = "Low GC" if gc < 40 else ("Normal" if gc <= 60 else "High Stability")
            gc_color = "#3b82f6" if gc < 40 else ("#22c55e" if gc <= 60 else "#ef4444")
            gf = go.Figure(go.Indicator(mode="gauge+number", value=gc,
                title={"text": f"GC % — {gc_label}", "font": {"size": 13}},
                gauge={"axis": {"range": [0,100]}, "bar": {"color": gc_color},
                       "steps": [{"range":[0,40],"color":"rgba(59,130,246,0.15)"},
                                  {"range":[40,60],"color":"rgba(34,197,94,0.15)"},
                                  {"range":[60,100],"color":"rgba(239,68,68,0.15)"}],
                       "threshold": {"line":{"color":gc_color,"width":3},"thickness":0.75,"value":gc}}))
            gf.update_layout(template="plotly_white", height=240, font=dict(family="Inter,sans-serif",color="#0a2540"),
                margin=dict(t=40,b=10,l=20,r=20), paper_bgcolor="#ffffff")
            st.plotly_chart(gf, use_container_width=True)
    with v2:
        if total > 0:
            cd = counts.reset_index(); cd.columns = ["Nucleotide","Count"]
            df = px.pie(cd, values="Count", names="Nucleotide", hole=0.55,
                color_discrete_sequence=["#2563eb","#7c3aed","#059669","#d97706"])
            df.update_traces(textinfo="label+percent")
            df.update_layout(template="plotly_white", height=240, margin=dict(t=20,b=10,l=10,r=10),
                paper_bgcolor="#ffffff", font=dict(family="Inter,sans-serif",color="#0a2540"))
            st.plotly_chart(df, use_container_width=True)
    st.markdown(section_header("Sequence stability heatmap"), unsafe_allow_html=True)
    if len(seq) > 0:
        gc_matrix = np.zeros((10, 10))
        for idx in range(100):
            s = int(idx*len(seq)/100); e = max(s+1, int((idx+1)*len(seq)/100))
            r, c = divmod(idx, 10); gc_matrix[r,c] = gc_content_percent(seq[s:e])
        hm = go.Figure(go.Heatmap(z=gc_matrix, colorscale="Blues", showscale=True,
            hovertemplate="Segment (%{x},%{y}): %{z:.1f}% GC<extra></extra>", xgap=2, ygap=2))
        hm.update_layout(template="plotly_white", paper_bgcolor="#ffffff", title="GC stability map",
            xaxis_title="Segment column", yaxis_title="Segment row",
            font=dict(family="Inter,sans-serif",color="#0a2540"), margin=dict(t=35,l=30,r=30,b=30))
        st.plotly_chart(hm, use_container_width=True)
    # Auto-lookup conservation if not passed in (for NCBI-searched genes)
    if not species_map and selected_gene_name:
        species_map = CONSERVATION.get(selected_gene_name.upper(), {}) or None

    if species_map:
        st.markdown(section_header("BLAST — sequence conservation"), unsafe_allow_html=True)
        cdf = pd.DataFrame({"Species": list(species_map.keys()),
            "Identity (%)": [round(v*100,2) for v in species_map.values()]})
        bar = px.bar(cdf, x="Species", y="Identity (%)", title="Pairwise DNA identity vs human",
            text="Identity (%)", color_discrete_sequence=["#2563eb"])
        bar.update_traces(textposition="outside"); bar.update_yaxes(range=[0,100])
        bar.update_layout(template="plotly_white", paper_bgcolor="#ffffff", font=dict(family="Inter,sans-serif",color="#0a2540"))
        st.plotly_chart(bar, use_container_width=True)
    else:
        st.markdown(section_header("BLAST — sequence conservation"), unsafe_allow_html=True)
        st.caption("Conservation data not available for this gene.")
    st.markdown(section_header("Gene fact sheet"), unsafe_allow_html=True)
    st.info(f"**{selected_gene_name}** — {description[:400] if description else 'No description available.'}")

def main():
    st.set_page_config(page_title="GeneScope", layout="wide")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#f8fafc;}
    .stApp{background:#f8fafc;}
    .block-container{max-width:1100px!important;margin:0 auto!important;padding:0 1.2rem 2rem!important;}
    h1,h2,h3,h4,h5{font-family:'Inter',sans-serif;font-weight:700;color:#0a2540;}
    header[data-testid="stHeader"]{background:#fff;border-bottom:1px solid #e2e8f0;}
    .stTabs [data-baseweb="tab-list"]{border-bottom:1px solid #e2e8f0!important;gap:0;background:transparent!important;}
    .stTabs [data-baseweb="tab"]{color:#8898b3!important;font-size:13px!important;font-weight:500!important;padding:10px 18px!important;border-radius:0!important;background:transparent!important;}
    .stTabs [aria-selected="true"]{color:#2563eb!important;border-bottom:2px solid #2563eb!important;font-weight:600!important;}
    .stButton button{background:#0a2540!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:500!important;font-size:13px!important;padding:8px 20px!important;}
    .stButton button:hover{background:#2563eb!important;}
    .stTextInput input{border-radius:8px!important;border-color:#e2e8f0!important;background:#fff!important;font-size:13px!important;}
    .stSelectbox div[data-baseweb="select"]>div{border-radius:8px!important;border-color:#e2e8f0!important;background:#fff!important;}
    .stSelectbox label,.stNumberInput label,.stTextInput label{color:#8898b3!important;font-size:11px!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.5px!important;}
    div[data-testid="stMetricValue"]{font-size:1.6rem!important;font-weight:700!important;color:#0a2540!important;}
    div[data-testid="stMetricLabel"]{font-size:11px!important;color:#8898b3!important;text-transform:uppercase!important;letter-spacing:0.5px!important;}
    div[data-testid="metric-container"]{background:#fff!important;border:1px solid #e2e8f0!important;border-radius:12px!important;padding:16px 18px!important;}
    .stSuccess>div{background:#f0fdf4!important;border-radius:8px!important;border:1px solid #bbf7d0!important;color:#14532d!important;}
    .stWarning>div{background:#fffbeb!important;border-radius:8px!important;border:1px solid #fde68a!important;color:#78350f!important;}
    .stError>div{background:#fef2f2!important;border-radius:8px!important;border:1px solid #fecaca!important;color:#7f1d1d!important;}
    .stInfo>div{background:#eff6ff!important;border-radius:8px!important;border:1px solid #bfdbfe!important;color:#1e40af!important;}
    code,.stCode{font-size:12px!important;background:#f8fafc!important;border:1px solid #e2e8f0!important;border-radius:6px!important;}
    .streamlit-expanderHeader{font-size:13px!important;font-weight:600!important;color:#0a2540!important;}
    ::-webkit-scrollbar{width:4px;} ::-webkit-scrollbar-thumb{background:#e2e8f0;border-radius:10px;}
    </style>""", unsafe_allow_html=True)

    genes = load_genes(); gene_ids = [g["gene"] for g in genes]

    # ── Protect all session state keys from being wiped on rerun ─────────────
    for _k in ["local_protein", "ncbi_protein", "local_mutation_result", "ncbi_mutation_result",
               "local_3d_loaded", "ncbi_3d_loaded", "local_3d_pdb", "ncbi_3d_pdb",
               "ncbi_gene", "ncbi_search_term"]:
        if _k not in st.session_state:
            st.session_state[_k] = None if "_loaded" not in _k else False
            if _k in ("ncbi_search_term", "local_3d_pdb", "ncbi_3d_pdb"):
                st.session_state[_k] = ""

    st.markdown("<div style='background:#fff;border-bottom:1px solid #e2e8f0;padding:14px 0 10px;margin-bottom:20px;'>", unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns([2, 3, 1])
    with nav1:
        st.markdown("""<div style='padding-top:4px;'>
        <div style='font-size:22px;font-weight:800;color:#0a2540;letter-spacing:-0.5px;'>🧬 GeneScope</div>
        <div style='font-size:12px;color:#8898b3;font-weight:500;margin-top:2px;'>Gene analytics & structure insights</div>
        </div>""", unsafe_allow_html=True)
    with nav2:
        search_term = st.text_input("", "", placeholder="Search gene or disease — e.g. BRCA1, TP53, Down syndrome, cystic fibrosis...")
        if not search_term:
            st.session_state["ncbi_gene"] = None
            st.session_state["ncbi_search_term"] = ""
        filtered_genes = [g for g in gene_ids if search_term.upper() in g.upper()]
        if filtered_genes:
            selected_id = st.selectbox("Matching genes", options=filtered_genes, index=0)
            st.session_state["ncbi_gene"] = None
            st.session_state["ncbi_search_term"] = ""
        else:
            selected_id = gene_ids[0]
            if search_term:
                already_fetched = (
                    st.session_state.get("ncbi_search_term", "") == search_term
                    and st.session_state.get("ncbi_gene") is not None
                )
                if not already_fetched:
                    for key in ["ncbi_protein", "ncbi_mutation_result",
                                "ncbi_local_protein", "ncbi_local_mutation_result"]:
                        st.session_state.pop(key, None)
                    with st.spinner("Searching NCBI, UniProt & PDB..."):
                        ncbi, error = fetch_from_ncbi(search_term)
                    if ncbi:
                        st.session_state["ncbi_gene"] = ncbi
                        st.session_state["ncbi_search_term"] = search_term
                    else:
                        st.session_state["ncbi_gene"] = None
                        st.warning(f"⚠️ {error}")
                if st.session_state.get("ncbi_gene"):
                    ncbi_cached = st.session_state["ncbi_gene"]
                    st.success(f"🌐 Found: **{ncbi_cached['name']}** — {ncbi_cached['full_name']}")
    with nav3:
        st.markdown(f"<div style='text-align:right;padding-top:6px;'><span style='background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;'>🧬 {len(gene_ids)} genes</span></div>", unsafe_allow_html=True)
        if st.button("About"):
            st.info("GeneScope — gene analytics dashboard with live NCBI, UniProt & PDB integration, 3D protein viewer, mutation simulator, and 2D sequence analysis.")
    st.markdown("</div>", unsafe_allow_html=True)

    import re
    if (search_term and len(search_term.strip()) == 4
            and re.match(r'^[0-9][A-Za-z0-9]{3}$', search_term.strip())):
        pdb_input = search_term.strip().upper()
        st.markdown(
            f"<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;"
            f"padding:14px 20px;margin-bottom:16px;'>"
            f"<div style='font-size:11px;color:#2563eb;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.5px;margin-bottom:4px;'>Direct PDB structure viewer</div>"
            f"<div style='font-size:20px;font-weight:700;color:#0a2540;'>PDB: {pdb_input}</div>"
            f"</div>", unsafe_allow_html=True)
        show_3d_protein(pdb_input)
        st.markdown(f"[View on RCSB PDB ↗](https://www.rcsb.org/structure/{pdb_input})")
        st.stop()

    show_ncbi = (
        bool(search_term) and not filtered_genes
        and st.session_state.get("ncbi_gene") is not None
        and st.session_state.get("ncbi_search_term", "") == search_term
    )

    if show_ncbi:
        ncbi   = st.session_state["ncbi_gene"]
        sq     = st.session_state.get("ncbi_search_term", search_term)
        seq    = normalize_seq(ncbi.get("sequence", ""))
        pdb_id = ncbi.get("pdb_id", "")
        gc     = gc_content_percent(seq) if seq else 0
        mw     = molecular_weight_dna(seq) if seq else 0
        tm_e   = melting_temperature_tm(seq) if seq else 0
        tm_w   = wallace_tm(seq) if seq else 0
        gc_sub = "Low GC" if gc < 40 else ("Moderate" if gc <= 60 else "High GC")
        refseq = ncbi.get("refseq_id", "—")

        if sq.lower().strip() not in ncbi["name"].lower():
            st.info(f"💡 Searched for '{sq}' — showing most associated gene **{ncbi['name']}**. Disease names map to their primary associated gene on NCBI.")

        st.markdown(
            f"<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;'>"
            f"<div><div style='font-size:11px;color:#2563eb;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;'>Live data — NCBI · UniProt · PDB</div>"
            f"<div style='font-size:20px;font-weight:700;color:#0a2540;'>{ncbi['name']} <span style='font-size:13px;color:#64748b;font-weight:400;'>— {ncbi['full_name']}</span></div>"
            f"<div style='font-size:12px;color:#8898b3;margin-top:2px;'>Chr {ncbi['chromosome']} · {ncbi['location']} · Aliases: {ncbi.get('aliases','—')}</div></div>"
            f"<div style='display:flex;gap:8px;flex-wrap:wrap;'>"
            f"<span style='background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;'>NCBI ✓</span>"
            f"{'<span style=\"background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;\">UniProt ✓</span>' if ncbi.get('protein_name') else '<span style=\"background:#fef2f2;color:#dc2626;border:1px solid #fecaca;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;\">UniProt —</span>'}"
            f"{'<span style=\"background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;\">PDB ✓</span>' if pdb_id else '<span style=\"background:#fffbeb;color:#d97706;border:1px solid #fde68a;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;\">PDB not found</span>'}"
            f"</div></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:12px;'>
            {metric_card("Gene", ncbi['name'], "Live · NCBI")}
            {metric_card("GC Content", f"{gc:.2f}%" if seq else "N/A", gc_sub)}
            {metric_card("Sequence Length", f"{len(seq)} nt" if seq else "N/A", f"RefSeq {refseq}")}
            {metric_card("Empirical Tm", f"{tm_e:.1f} °C" if seq else "N/A", "Stable" if tm_e > 60 else "Low stability")}
        </div>
        <div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:24px;'>
            {metric_card("Molecular Weight", f"{mw:,.0f} Da" if seq else "N/A")}
            {metric_card("Wallace Tm", f"{tm_w:.1f} °C" if seq else "N/A")}
            {metric_card("Chromosome", f"Chr {ncbi['chromosome']}", ncbi['location'])}
        </div>
        """, unsafe_allow_html=True)

        tab_seq, tab_viz, tab_trans = st.tabs(["Sequence analysis", "Visualization", "Translation & mutation"])

        with tab_seq:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(section_header("Entry overview"), unsafe_allow_html=True)
                st.markdown(f"""<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
                <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Protein</div><div style='font-size:13px;font-weight:600;color:#0a2540;'>{ncbi.get('protein_name','—')}</div></div>
                <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Location (UniProt)</div><div style='font-size:13px;font-weight:600;color:#0a2540;'>{ncbi.get('subcellular_location','—')}</div></div>
                <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Chromosome</div><div style='font-size:13px;font-weight:600;color:#0a2540;'>Chr {ncbi['chromosome']} · {ncbi['location']}</div></div>
                <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Aliases</div><div style='font-size:13px;font-weight:600;color:#0a2540;'>{ncbi.get('aliases','—')}</div></div>
                </div>""", unsafe_allow_html=True)
                if ncbi.get("go_function") and ncbi["go_function"] != "—":
                    st.markdown(section_header("GO molecular function"), unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:13px;color:#475569;line-height:1.7;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;'>{ncbi['go_function']}</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(section_header("Gene summary"), unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:13px;color:#475569;line-height:1.8;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;'>{ncbi['summary']}</div>", unsafe_allow_html=True)
                st.markdown(section_header("DNA sequence"), unsafe_allow_html=True)
                st.caption(f"Source: RefSeq {refseq} — fetched live from NCBI")
                if seq:
                    with st.expander("View raw DNA sequence", expanded=False):
                        st.code(seq, language="text")
                    st.markdown(f"<div style='display:inline-block;background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:4px 10px;font-size:12px;color:#1e40af;font-weight:600;margin-top:4px;'>{len(seq)} bases</div>", unsafe_allow_html=True)
                else:
                    st.warning("Sequence not available from NCBI for this gene.")
            st.markdown(section_header("External databases"), unsafe_allow_html=True)
            ext1, ext2, ext3 = st.columns(3)
            with ext1:
                ul = ncbi.get("uniprot_url", "")
                st.markdown(f"<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;margin-bottom:6px;'>UniProt</div><div style='font-size:13px;font-weight:600;color:#0a2540;margin-bottom:4px;'>{ncbi.get('uniprot_accession','—')}</div>{'<a href=\"'+ul+'\" target=\"_blank\" style=\"font-size:11px;color:#2563eb;\">View on UniProt ↗</a>' if ul else ''}</div>", unsafe_allow_html=True)
            with ext2:
                ncbi_gene_url = f"https://www.ncbi.nlm.nih.gov/gene/{ncbi['ncbi_id']}"
                st.markdown(f"<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;margin-bottom:6px;'>NCBI Gene</div><div style='font-size:13px;font-weight:600;color:#0a2540;margin-bottom:4px;'>Gene ID: {ncbi['ncbi_id']}</div><a href='{ncbi_gene_url}' target='_blank' style='font-size:11px;color:#2563eb;'>View on NCBI ↗</a></div>", unsafe_allow_html=True)
            with ext3:
                pl = f"<a href='https://www.rcsb.org/structure/{pdb_id}' target='_blank' style='font-size:11px;color:#2563eb;'>View on RCSB ↗</a>" if pdb_id else ""
                st.markdown(f"<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;margin-bottom:6px;'>RCSB PDB</div><div style='font-size:13px;font-weight:600;color:#0a2540;margin-bottom:4px;'>{pdb_id if pdb_id else 'Not found'}</div>{pl}</div>", unsafe_allow_html=True)

        with tab_viz:
            if seq:
                render_visualization(seq, gc, ncbi["name"], ncbi.get("summary", ""),
                                     species_map=CONSERVATION.get(ncbi["name"].upper(), {}) or None)
            else:
                st.warning("No sequence available — visualization requires DNA sequence data.")

        with tab_trans:
            if seq:
                render_translation(seq, key_prefix="ncbi")
                st.markdown(section_header("3D protein structure"), unsafe_allow_html=True)
                if pdb_id:
                    render_3d_button(pdb_id, key_prefix="ncbi")
                else:
                    st.info("No PDB structure found for this gene.")
                render_mutation_simulator(seq, pdb_id, key_prefix="ncbi")
            else:
                st.warning("No sequence available for translation or mutation simulation.")

        st.stop()

    # LOCAL GENE PROFILE
    selected_gene    = next(g for g in genes if g["gene"] == selected_id)
    seq              = normalize_seq(selected_gene["sequence"])
    gc               = gc_content_percent(seq)
    mw               = molecular_weight_dna(seq)
    tm_empirical     = melting_temperature_tm(seq)
    tm_wallace_value = wallace_tm(seq)
    pdb_id           = selected_gene.get("pdb_id", "")
    gc_sub           = "Low GC" if gc < 40 else ("Moderate GC" if gc <= 60 else "High GC")
    tm_sub           = "Stable" if tm_empirical > 60 else "Low stability"
    refseq           = selected_gene.get("refseq_mrna", "—")

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:12px;'>
        {metric_card("Gene",            selected_gene.get('gene','—'), selected_gene.get('category','—'))}
        {metric_card("GC Content",      f"{gc:.2f}%",                  gc_sub)}
        {metric_card("Sequence Length", f"{len(seq)} bp",              f"RefSeq {refseq}")}
        {metric_card("Empirical Tm",    f"{tm_empirical:.1f} °C",      tm_sub)}
    </div>
    <div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:24px;'>
        {metric_card("Molecular Weight", f"{mw:,.0f} Da")}
        {metric_card("Wallace Tm",       f"{tm_wallace_value:.1f} °C")}
        {metric_card("Sequence Status",  "RefSeq verified", refseq)}
    </div>
    """, unsafe_allow_html=True)

    tab_seq, tab_viz, tab_trans = st.tabs(["Sequence analysis", "Visualization", "Translation & mutation"])

    with tab_seq:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(section_header("Entry overview"), unsafe_allow_html=True)
            st.markdown(f"""<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Protein</div><div style='font-size:13px;font-weight:600;color:#0a2540;'>{selected_gene.get('protein_name','—')}</div></div>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Location</div><div style='font-size:13px;font-weight:600;color:#0a2540;'>{selected_gene.get('subcellular_location','—')}</div></div>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Category</div><div style='font-size:13px;font-weight:600;color:#0a2540;'>{selected_gene.get('category','—')}</div></div>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;'><div style='font-size:11px;color:#8898b3;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;'>Condition</div><div style='font-size:13px;font-weight:600;color:#0a2540;'>{selected_gene.get('disease','—')}</div></div>
            </div>""", unsafe_allow_html=True)
            st.markdown(section_header("Function & biological role"), unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:13px;color:#475569;line-height:1.7;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;'>{selected_gene.get('go_function') or selected_gene.get('description','—')}</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown(section_header("DNA sequence"), unsafe_allow_html=True)
            if selected_gene.get("refseq_mrna"):
                st.caption(f"Source: RefSeq {selected_gene['refseq_mrna']} — {selected_gene.get('sequence_note','').strip()}")
            with st.expander("View raw DNA sequence", expanded=False):
                st.code(seq, language="text")
            st.markdown(f"<div style='display:inline-block;background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:4px 10px;font-size:12px;color:#1e40af;font-weight:600;margin-top:4px;'>{len(seq)} bases</div>", unsafe_allow_html=True)
        variants = selected_gene.get("variants", []) or []
        if variants:
            st.markdown(section_header("Clinical variants"), unsafe_allow_html=True)
            vcols = st.columns(2)
            for idx, row in pd.DataFrame(variants).iterrows():
                sig = str(row.get("significance", "Unknown")); is_path = sig.lower() == "pathogenic"
                bb = "#fef2f2" if is_path else "#f0fdf4"; bc = "#dc2626" if is_path else "#16a34a"
                bd = "#fecaca" if is_path else "#bbf7d0"
                with vcols[idx % 2]:
                    st.markdown(f"""<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin-bottom:8px;'>
                    <div style='font-family:monospace;font-size:12px;font-weight:700;color:#0a2540;margin-bottom:4px;'>{row.get('variant','—')}</div>
                    <div style='font-size:12px;color:#475569;margin-bottom:6px;'>{row.get('condition','—')}</div>
                    <div style='font-size:11px;color:#64748b;margin-bottom:6px;'>{row.get('note','—')}</div>
                    <span style='background:{bb};color:{bc};border:1px solid {bd};border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;'>{sig}</span>
                    </div>""", unsafe_allow_html=True)

    with tab_viz:
        species_map = CONSERVATION.get(selected_gene["gene"], {})
        render_visualization(seq, gc, selected_gene["gene"],
                             selected_gene.get("description", ""),
                             species_map=species_map if species_map else None)

    with tab_trans:
        render_translation(seq, key_prefix="local")
        st.markdown(section_header("3D protein structure"), unsafe_allow_html=True)
        if pdb_id:
            render_3d_button(pdb_id, key_prefix="local")
        else:
            st.info("No PDB structure available for this gene.")
        render_mutation_simulator(seq, pdb_id, key_prefix="local")

if __name__ == "__main__":
    main()
