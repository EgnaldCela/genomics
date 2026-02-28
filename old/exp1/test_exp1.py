from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from evo2 import Evo2
import gzip
import pandas as pd
import torch
from typing import List, Optional
import random
from tqdm.auto import tqdm

def get_chr_section(fasta_file: str, start: int, length: int, chromosome_number: int) -> str:
    """
    Args: 
        fasta_file, path to the zipped fasta file
        start, the index where you start extracting a sequence from the full DNA sequence
        length, is the length of the sequence you want to extract from the full DNA
        chromosome_number, is the number of the chromosome you extract DNA from
    
    Returns: Extracted section from a starting index of a given chromosome from a gzipped FASTA file
    
    """
    with gzip.open(fasta_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            chromosome = "chr" + str(chromosome_number)
            if record.id in [chromosome, str(chromosome_number)]:  
                return str(record.seq[start:start+length]).upper()



# evaluation metrics taken from evo2 generation notebook
# reference link https://github.com/ArcInstitute/evo2/blob/main/notebooks/generation/generation_notebook.ipynb


def analyze_alignments(generated_seqs: List[str],
                       target_seqs: List[str],
                       names: Optional[List[str]] = None
                      ) -> List[dict]:
    """
    Args:
        generated_seqs, a list of strings of generated sequences by the model
        target_seqs, a list of strings of the true sequences extracted from the DNA
        names, (optional) labels for each sequence

    Returns:
        A list of dicts, where each dictionary is a row of data, easier for pandas conversion
    """
    metrics = []
    # print("\nSequence Alignments:")

    for i, (gen_seq, target_seq) in enumerate(zip(generated_seqs, target_seqs)):
        if names and i < len(names):
            print(f"\nAlignment {i+1} ({names[i]}):")
        # else:
        #      # print(f"\nAlignment {i+1}:")

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

# main function
def test_chr_slice(fasta_file, model, start, length, prompt_len, chromosome_number):
    """
    Loads a slice of a given chromosome, generate continuation, and evaluate similarity.

    Args:
        fasta_file, path to the zipped fasta file
        model, the evo2 loaded model
        start, the index where you start extracting a sequence from the full DNA sequence
        length, is the length of the sequence you want to extract from the full DNA
        prompt_len, the number of base pairs you feed into the evo2 model
        chromosome_number, is the number of the chromosome you extract DNA fro
        
    
    Returns:
        alignment_metrics, a pandas data frame with the eval metrics for the given sequence and generation
        generation_temperature, the temperature of generation needed for csv file naming
    """
    # extract sequence 
    sequence = get_chr_section(fasta_file=fasta_file, start=start, length=length, chromosome_number=chromosome_number)

    # prompt and target
    prompt = sequence[:prompt_len]
    target = sequence[prompt_len:]

    print(f"Prompt length: {len(prompt)}, Target length: {len(target)}")

    # set generation temperature
    generation_temperature = 0.1

    # generate continuation
    output = model.generate(
        prompt_seqs=[prompt],
        n_tokens=len(target), # length - prompt_length
        temperature=generation_temperature, # aim for determinism 
        top_k=4
    )
    
    generated_sequence = output.sequences[0]

    # evaluate metrics 
    alignment_metrics = analyze_alignments([generated_sequence], [target], [f"chr{str(chromosome_number)}_{start}"])
    
    return (pd.DataFrame(alignment_metrics), generation_temperature)


if __name__ == "__main__":

    # def generate_random_dna(length=100):
    #     """Generate a random DNA sequence of given length using ACTG."""
    #     return ''.join(random.choice("ACTG") for _ in range(length))



    # mean = 0
    # for _ in tqdm(range(1000)):
        
    #     rs1 = generate_random_dna()
    #     rs2 = ''.join(random.choice("A") for _ in range(100))
    #     # rs2 = ["A"]
    #     l = (analyze_alignments([rs1], [rs2]))
    #     mean += l[0]["similarity"]
    
    # print(mean / 1000)

    
    # path to genome
    fasta_file = "../CHM13v2.0.fna.gz"

    # load model
    model = Evo2("evo2_7b")

    all_results = []

    # define starting point and length
    start = 1_000_000
    length = 3_000
    prompt_len = 2_500

    # loop over chromosomes 1–22 + X
    chromosomes = [i for i in range(1, 23)] + ['X']

    # temperature
    temperature = None

    for chrom in tqdm(chromosomes):
        print(f"\n=== Processing chromosome {chrom} ===")
        try:
            # run slice test
            df_tuple = test_chr_slice(
                fasta_file,
                model,
                start=start,
                length=length, # change to 5_000
                prompt_len=prompt_len, # change to 2500
                chromosome_number = chrom
            )
            # udate name to include chromosome
            # access the data frame (1st element)
            df_tuple[0]['chromosome'] = chrom
            all_results.append(df_tuple[0])
            # pass the temperature value for the csv generation if it is not passed
            if not temperature:
                temperature = df_tuple[1] 
        except ValueError as e:
            print(f"Skipping chromosome {chrom}: {e}")

    # combine all results and save
    final_df = pd.concat(all_results, ignore_index=True)
    
    # define csv file name
    csv_path = f"exp1_metrics_genlen{length-prompt_len}_temp{temperature}_start{start}_copy03.csv"
    
    # save it as csv
    final_df.to_csv(csv_path, index=False)

    print(f"\nAll results saved in {csv_path}")
    print(final_df)

