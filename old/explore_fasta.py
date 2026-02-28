import gzip

 
"""
DESCRIPTION
A quick script made to navigate easily FASTA (.fna or .fasta) files, format typically used
to store genomes after they're assembled.
Typically they come compressed using the GNU zip format (.gz)

IMPORTANT TIPS
Always read .gz files with "rt" mode to treat them exactly as text files

"""
# set it to your desired file
filepath = "CHM13v2.0.fna.gz"

def read_headers(filepath, make_file=False, file_name=None):
    """
    INPUT: 
            filepath, path to a compressed file
            make_file, if set True it stores the headers in a .txt file
            file_name, set it in the name you want the above .txt file to be saved as
    
    OUTPUT:
            Headers of the FASTA file
            (Optional) File of such headers in .txt
    """
    with gzip.open(filepath, "rt") as f:
        headers = []
        for line in f:
            if line.startswith(">"):
                line = line.strip()
                headers.append(line)
    
    if make_file:
        name = file_name + ".txt" if file_name else "headers.txt"
        with open(name, "w") as f_out:
            for line in headers:
                f_out.write(line + "\n" ) 
    
    return headers


def read_file(filepath, limit=None, make_file=False):
    """
    INPUT:
            filepath, path to fasta.gz file
            limit, the number of bases you want to be read

    OUTPUT:
            out, list of tuples (header, bases)
            (Optional) file with such content written
    """
    out = []
    header = None
    sequence = ""
    
    with gzip.open(filepath, "rt") as f:
        for line in f:
            line = line.strip()
            # handle empty lines
            if not line:
                continue

            # you either encounter the 1st header or consecutive ones
            if line.startswith(">"):
                if header and sequence:
                    out.append((header, sequence))
                if limit and len(sequence)>=limit:
                    break
                header = line
                sequence = ""
            else:
                sequence += line
                if len(sequence) >= limit:
                    out.append((header, sequence[:limit]))
    
    if header and sequence and (not limit or len(sequence) <= limit):
        out.append((header, sequence))
    
    if make_file:
        with open("genome_reading.txt", "w") as f_out:
            for (header, content) in out:
                f_out.write(header + "\n")
                f_out.write(content + "\n")

    return out
                



if __name__ == '__main__':
    print(read_file(filepath, limit=100))

"""
SIDE NOTES:
- I ran read_headers(CHM13v2.0.fna.gz) and got as result
['>chr1', '>chr2', '>chr3', '>chr4', '>chr5', '>chr6', '>chr7', '>chr8', '>chr9', '>chr10', '>chr11', 
'>chr12', '>chr13', '>chr14', '>chr15', '>chr16', '>chr17', '>chr18', '>chr19', '>chr20', '>chr21', 
'>chr22', '>chrX', '>chrY', '>chrM']
What is chromosome M?

"""