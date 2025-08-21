from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from evo2 import Evo2
import gzip
import pandas as pd
import torch
from typing import List, Optional


def get_chr_section(fasta_file: str, start: int, length: int, chromosome_number: int) -> str:
    """
    Extract a section of a given chromosome from a gzipped FASTA file.
    """
    with gzip.open(fasta_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            chromosome = "chr" + str(chromosome_number)
            if record.id in [chromosome, str(chromosome_number)]:  
                return str(record.seq[start:start+length]).upper()

# taken from evo2 generation notebook
def analyze_alignments(generated_seqs: List[str],
                       target_seqs: List[str],
                       names: Optional[List[str]] = None
                      ) -> List[dict]:
    """
    Compute pairwise alignment metrics between generated and target sequences.
    """
    metrics = []
    print("\nSequence Alignments:")

    for i, (gen_seq, target_seq) in enumerate(zip(generated_seqs, target_seqs)):
        if names and i < len(names):
            print(f"\nAlignment {i+1} ({names[i]}):")
        else:
            print(f"\nAlignment {i+1}:")

        gen_bio_seq = Seq(gen_seq)
        target_bio_seq = Seq(target_seq)

        alignments = pairwise2.align.globalms(
            gen_bio_seq, target_bio_seq,
            match=2,
            mismatch=-1,
            open=-0.5,
            extend=-0.1
        )
        best_alignment = alignments[0]

        matches = sum(a == b for a, b in zip(best_alignment[0], best_alignment[1])
                      if a != '-' and b != '-')
        similarity = (matches / len(target_seq)) * 100

        seq_metrics = {
            'similarity': similarity,
            'score': best_alignment[2],
            'length': len(target_seq),
            'gaps': best_alignment[0].count('-') + best_alignment[1].count('-')
        }

        if names and i < len(names):
            seq_metrics['name'] = names[i]

        metrics.append(seq_metrics)

    return metrics

# main function, 
def test_chr_slice(fasta_file, model, start, length, prompt_len, chromosome_number):
    """
    Load a slice of a given chromosome, generate continuation, and evaluate similarity.
    """
    # Extract sequence slice
    sequence = get_chr_section(fasta_file, start=start, length=length, chromosome_number=chromosome_number)

    # Prompt and target
    prompt = sequence[:prompt_len]
    target = sequence[prompt_len:]

    print(f"Prompt length: {len(prompt)}, Target length: {len(target)}")

    # generate continuation
    output = model.generate(
        prompt_seqs=[prompt],
        n_tokens=len(target), # length - prompt_length
        temperature=0.5, # aim for determinism 
        top_k=4
    )
    generated_sequence = output.sequences[0]

    # Compare
    alignment_metrics = analyze_alignments([generated_sequence], [target], [f"chr{str(chromosome_number)}_{start}"])
    
    return pd.DataFrame(alignment_metrics)


if __name__ == "__main__":
    
    fasta_file = "../CHM13v2.0.fna.gz"

    # Load model
    model = Evo2("evo2_7b")

    all_results = []

    # Loop over chromosomes 1–22 + X
    chromosomes = list(range(1, 23)) + ['X']

    for chrom in chromosomes:
        print(f"\n=== Processing chromosome {chrom} ===")
        try:
            # Run slice test
            df = test_chr_slice(
                fasta_file,
                model,
                start=1_000_000,
                length=5_000, # change to 5_000
                prompt_len=2_500, # change to 2500
                chromosome_number = chrom
            )
            # Update name to include chromosome
            df['chromosome'] = chrom
            all_results.append(df)
        except ValueError as e:
            print(f"Skipping chromosome {chrom}: {e}")

    # combine all results and save
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("exp1_metrics_genlen2500_temp05.csv", index=False)

    print("\nAll chromosome results saved to exp1_metrics_genlen2500_temp05.csv")
    print(final_df)

