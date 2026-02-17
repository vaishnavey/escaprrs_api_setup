# processor.py
# Place this file at the root of your private repo (Vaishnavey/testprrs).
# The server's main.py will load process_sequence(...) from this file and call it for /analyze.
# This implements the same logic as your Colab UI but returns JSON and base64 plot(s).

import os
import math
import subprocess
import tempfile
import base64
from io import BytesIO, StringIO
from typing import Dict, Any, List

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
import Levenshtein

# Reference sequence (same as the Colab notebook)
REF_AA_SEQ = (
    "MLEKCLTAGYCSQLLFFWCIVPFCFAALVNAASNSSSHLQLIYNLTICELNGTDWLNQKFDWAVETFVIFPVLTHIVSYGALTTSHFLDTAGLITVSTAGYYHGRYVLSSIYAVFALAALICFVIRLTKNCMSWRYSCTRYTNFLLDTKGNLYRWRSPVVIERRGKVEVGDHLIDLKRVVLDGSAATPITKISAEQWGRP"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _read_flevo_score_csv():
    path = os.path.join(BASE_DIR, "prrsv_scaled_flevo_scores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found in repository snapshot.")
    df = pd.read_csv(path)
    df["mutations"] = df["wt"] + df["i"].astype(str) + df["mut"]
    pos = dict(zip(df["mutations"], df["evescape"] - df["evescape"].min()))
    sig = dict(zip(df["mutations"], df["evescape"].apply(lambda x: 1 / (1 + math.exp(-x)))))
    return pos, sig

def _get_mutations_mafft(wt_seq: str, query_seq: str) -> List[str]:
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".fasta") as tmpf:
        tmpf.write(">WT\n")
        tmpf.write(wt_seq + "\n")
        tmpf.write(">QUERY\n")
        tmpf.write(query_seq + "\n")
        tmpname = tmpf.name

    try:
        result = subprocess.run(
            ["mafft", "--globalpair", "--maxiterate", "1000", "--quiet", tmpname],
            capture_output=True, text=True, check=True
        )
        alignment = AlignIO.read(StringIO(result.stdout), "fasta")
        wt_aligned = str(alignment[0].seq)
        query_aligned = str(alignment[1].seq)

        mutations = []
        wt_pos = 0
        for wt_res, q_res in zip(wt_aligned, query_aligned):
            if wt_res != "-":
                wt_pos += 1
            if wt_res == "-" and q_res != "-":
                preceding = wt_aligned[wt_pos - 1] if wt_pos - 1 >= 0 and wt_pos - 1 < len(wt_aligned) else "-"
                mutations.append(f"{preceding}{wt_pos}ins{q_res}")
            elif wt_res != "-" and q_res == "-":
                mutations.append(f"{wt_res}{wt_pos}del")
            elif wt_res != "-" and q_res != "-" and wt_res != q_res:
                mutations.append(f"{wt_res}{wt_pos}{q_res}")
        return mutations
    finally:
        try:
            os.unlink(tmpname)
        except Exception:
            pass

def _find_closest_in_db(query_seq: str):
    db_path = os.path.join(BASE_DIR, "database.fasta")
    if not os.path.exists(db_path):
        return None, None, None
    min_distance = None
    closest_id = None
    for rec in SeqIO.parse(db_path, "fasta"):
        seq_str = str(rec.seq)
        d = Levenshtein.distance(seq_str, query_seq)
        if min_distance is None or d < min_distance:
            min_distance = d
            closest_id = rec.id
    if closest_id is None:
        return None, None, None
    similarity_db = (1 - min_distance / max(len(query_seq), 1)) * 100
    parts = closest_id.split("|")
    seq_id = parts[0]
    date = parts[1] if len(parts) > 1 else "Unknown"
    return seq_id, date, similarity_db

def _plot_score_colorbar(score_pos: float, uncertainty_pct: float = 0.0) -> str:
    minimum = 1.1572728000000003
    maximum = 204.29259040000005
    mean = 92.8164662
    critical_points = [204.29, 92.816, 3.253]
    critical_labels = ['Max \n204.29', 'Median\n92.816', 'Min\n1.153']

    whisker_length = uncertainty_pct / 100.0

    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.subplots_adjust(bottom=0.5)

    colors = [(1, 1, 1), (0, 0, 1), (1, 0, 0)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    norm = mpl.colors.Normalize(vmin=0, vmax=230)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label("EscaPRRS-ORF5 score")

    point_position = (score_pos - 0) / (maximum - 0)
    point_position = max(0.0, min(1.0, point_position))

    ax.plot(point_position, 2.5, marker='o', markersize=10,
            markerfacecolor='black', markeredgecolor='black', transform=ax.transAxes, clip_on=False)
    ax.text(point_position, 2.0, f"{score_pos:.2f}", color='black', ha='center', va='bottom', transform=ax.transAxes)

    ax.hlines(y=2.5,
              xmin=point_position - whisker_length / 2,
              xmax=point_position + whisker_length / 2,
              color='black', linewidth=3,
              transform=ax.transAxes, clip_on=False)

    ax.vlines(x=[point_position - whisker_length / 2, point_position + whisker_length / 2],
              ymin=2.45, ymax=2.55,
              color='black', linewidth=3,
              transform=ax.transAxes, clip_on=False)

    for rp, rl in zip(critical_points, critical_labels):
        xpos = rp / (maximum + 10)
        y_random = 0.5
        ax.plot(xpos, y_random, marker='o', markersize=10,
                markerfacecolor='white', markeredgecolor='black', linewidth=1.5, transform=ax.transAxes)
        ax.text(xpos, y_random + 1.5, rl, ha='center', va='top', fontsize=9, color='navy', transform=ax.transAxes)

    buf = BytesIO()
    fig.savefig(buf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return b64

def process_sequence(sequence: str, options: Dict[str, Any]) -> Dict[str, Any]:
    seq = "".join(str(sequence).split()).replace("-", "").upper()
    if not seq:
        raise ValueError("Empty sequence")

    pos_dict, sig_dict = _read_flevo_score_csv()

    lev_dist = Levenshtein.distance(REF_AA_SEQ, seq)
    similarity = (1 - lev_dist / max(len(REF_AA_SEQ), 1)) * 100
    uncertainty = (lev_dist / max(len(REF_AA_SEQ), 1)) * 100
    if similarity < 30:
        note = "This sequence is not likely that of PRRSV-2 GP5"
    else:
        note = ""

    mutations = _get_mutations_mafft(REF_AA_SEQ, seq)
    filtered = [m for m in mutations if "del" not in m and "ins" not in m]

    score_pos = sum(pos_dict.get(m, 0.0) for m in filtered)
    score_sigmoid = sum(sig_dict.get(m, 0.0) for m in filtered)

    table_rows = []
    for m in filtered:
        table_rows.append({
            "mutation": m,
            "evescape_pos": float(pos_dict.get(m, 0.0)),
            "evescape_sigmoid": float(sig_dict.get(m, 0.0))
        })

    seq_id, date, similarity_db = _find_closest_in_db(seq)

    plot_b64 = _plot_score_colorbar(score_pos, uncertainty_pct=uncertainty)

    summary = (
        f"Levenshtein distance: {lev_dist}; Similarity to WT: {similarity:.2f}%; "
        f"EscaPRRS score: {score_pos:.3f}; Uncertainty: {uncertainty:.3f}%. {note}"
    )

    result: Dict[str, Any] = {
        "summary": summary,
        "table": table_rows,
        "plots": [{"name": "escape_score", "b64": plot_b64}],
    }

    if seq_id:
        result["closest_match"] = {"id": seq_id, "date": date, "similarity": float(similarity_db)}

    return result
