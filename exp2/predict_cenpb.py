from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from evo2 import Evo2
import gzip
import pandas as pd
import torch
import re
from typing import List, Optional

# CENP-B box NTTCGNNNNANNCGGGN (froma luca's slides)
CENP_B_BOX_REGEX = re.compile(r"[ACGT]TTCG[ACGT]{4}A[ACGT]{2}CGGG[ACGT]")

# set start to zero 
def get_chr_section(fasta_file: str, length: int, chromosome_number: str) -> str:
    """
    Scan forward from 'start' until a CENP-B box is found.
    Return a slice of fixed 'length' ending with that CENP-B box.
    """
    with gzip.open(fasta_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            chrom_ids = [f"chr{chromosome_number}", str(chromosome_number)]
            if record.id in chrom_ids:
                full_sequence = str(record.seq).upper()

                # scan forward from 'start' = 0
                match = CENP_B_BOX_REGEX.search(full_sequence, pos=0)

                if not match:
                    raise ValueError(f"No CENP-B box found in chr{chromosome_number}")

                end = match.end()

                if end < length:
                    raise ValueError(f"Not enough bases before first CENP-B box in chr{chromosome_number}")

                # slice of exactly 'length' ending at the CENP-B box
                return full_sequence[end - length:end].upper()


def detect_cenp_b_box(sequence: str) -> int:
    """Return 1 if CENP-B box motif is present in the sequence, else 0."""
    return int(bool(CENP_B_BOX_REGEX.search(sequence)))

def analyze_alignments(generated_seqs: List[str],
                       target_seqs: List[str],
                       names: Optional[List[str]] = None) -> List[dict]:
    """
    Compute pairwise alignment metrics and detect CENP-B boxes.
    """
    metrics = []
    for i, (gen_seq, target_seq) in enumerate(zip(generated_seqs, target_seqs)):
        name = names[i] if names and i < len(names) else f"seq_{i+1}"

        alignments = pairwise2.align.globalms(
            gen_seq, target_seq,
            match=2, mismatch=-1, open=-0.5, extend=-0.1
        )
        best_alignment = alignments[0]

        matches = sum(a == b for a, b in zip(best_alignment[0], best_alignment[1])
                      if a != '-' and b != '-')
        similarity = (matches / len(target_seq)) * 100

        seq_metrics = {
            'similarity': similarity,
            'score': best_alignment[2],
            'length': len(target_seq),
            'gaps': best_alignment[0].count('-') + best_alignment[1].count('-'),
            'name': name,
            'cenp_b_box_detected': detect_cenp_b_box(gen_seq)
        }
        metrics.append(seq_metrics)

    return metrics

def test_chr_slice(fasta_file, model, length, prompt_len, chromosome_number):
    """
    Load a slice of a chromosome (ending in CENP-B), generate continuation,
    evaluate similarity, and detect CENP-B boxes.
    """
    sequence = get_chr_section(fasta_file, length=length, chromosome_number=chromosome_number)
    prompt = sequence[:prompt_len]
    target = sequence[prompt_len:]

    print(f"Prompt length: {len(prompt)}, Target length: {len(target)}")

    output = model.generate(
        prompt_seqs=[prompt],
        n_tokens=len(target),
        temperature=0.5,
        top_k=4
    )
    generated_sequence = output.sequences[0]

    alignment_metrics = analyze_alignments(
        [generated_sequence],
        [target],
        [f"chr{chromosome_number}"]
    )
    return pd.DataFrame(alignment_metrics)

if __name__ == "__main__":

    fasta_file = "../CHM13v2.0.fna.gz"
    model = Evo2("evo2_7b")

    all_results = []
    chromosomes = list(range(1, 23)) + ['X']

    for chrom in chromosomes:
        print(f"\n=== Processing chromosome {chrom} ===")
        try:
            df = test_chr_slice(
                fasta_file,
                model,
                length=5_000,
                prompt_len=2_500,
                chromosome_number=chrom
            )
            df['chromosome'] = chrom
            all_results.append(df)
        except ValueError as e:
            print(f"Skipping chromosome {chrom}: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv("exp2_metrics_genlen2500_temp05.csv", index=False)
        print("\nAll chromosome results saved to exp2_metrics_genlen2500_temp05.csv")
        print(final_df)


