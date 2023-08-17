
from termcolor import colored
from typing import List, Optional
from copy import deepcopy
from math import factorial
from itertools import combinations_with_replacement as cwr

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# =============================================================================
def highlight_indices(seq: np.array, indices: np.array, color: str):
    # We use deepcopy to prevent mutation and we cast to object so that we can treat contents as python strings
    # otherwise, it gets messed up as it treats each element as a single character
    newseq = deepcopy(seq).astype('object')
    newseq[indices] = np.vectorize(lambda x: colored(x, color, attrs=["bold"]))(seq[indices])
    return newseq

# =============================================================================
def print_sequence(seq, 
                   header: str=None, 
                   incorrect_indices: Optional[np.array]=None, 
                   correct_indices: Optional[np.array]=None):

    newseq = deepcopy(seq)
    if correct_indices is not None and correct_indices.size != 0:
        newseq = highlight_indices(newseq, correct_indices, "green")
    if incorrect_indices is not None and incorrect_indices.size != 0:
        newseq = highlight_indices(newseq, incorrect_indices, "red")

    line_length = 40
    if header:
        print(header)
    print("=" * line_length)

    i = 0
    while i < len(newseq):
        print(" ".join(newseq[i: i+line_length]))
        i += line_length

# =============================================================================
def get_sequences(fasta_file: str) -> List[np.array]:
    sequences = []
    with open(fasta_file, "r") as input_file:
        sequences = [seq.split("\n") for seq in input_file.read().split(">") if seq]
        sequences = ["".join(parts).strip() for _, *parts in sequences]
        sequences = [np.array([char for char in seq]) for seq in sequences]
    random.shuffle(sequences)
    return sequences

# =============================================================================
def combs_that_sum(totalsum, choose_k, max_pairs=3):

    # Base case
    if choose_k == 1:
        return [(totalsum,)]

    # Recursive case

    # We want to reduce the total values considered to increase speed
    values = list(range(1, int(totalsum/2)))
    random.shuffle(values)
    values = values[:max_pairs]

    pairs = [(x, totalsum - x) for x in values]
    combinations = [(first,) + comb for first, second in pairs for comb in combs_that_sum(second, choose_k - 1)]
    return combinations 

# # =============================================================================
# def noisify_by_random_contigs(seqs, noise_percent, numgaps, mingapsize, mincontigsize):
#     sequences = deepcopy(seqs)
#     for seq in sequences:

#         amino_acid_length = len(seq)
#         amino_acids_to_replace = int(amino_acid_length * noise_percent)
#         amino_acids_to_keep = amino_acid_length - amino_acids_to_replace

#         extra_contigs = random.choice([-1, 0, 1]) if numgaps > 1 else random.choice([0, 1])
#         combinations_of_gaps = combs_that_sum(amino_acids_to_replace, numgaps)
#         combinations_of_contigs = combs_that_sum(amino_acids_to_keep, numgaps + extra_contigs)

#         # Once we've listed all the possible combinations of gaps and contigs, we can just randomly select one from each and then shuffle
#         gaplist = list(random.choice(combinations_of_gaps))
#         contiglist = list(random.choice(combinations_of_contigs))

#         # So gaplist should have fixed length
#         # But contiglist will have a length somewhere between numgaps - 1 and numgaps + 1
#         # There are three possibilities:
#         # 1. numcontigs == numgaps, which means the first thing may be either a gap or a contig
#         # 2. numcontigs == numgaps + 1, which means that the first thing is a contig
#         # 3. numcontigs == numgaps - 1, which means that the first thing is a gap
#         iscontig = extra_contigs == 1 if extra_contigs != 0 else random.choice([True, False])

#         sequence_pointer = 0
#         while sequence_pointer < len(seq):
#             if iscontig:
#                 # Don't do anything for the contig except increment the pointer and pop off the contig queue
#                 sequence_pointer += contiglist.pop(0)
#             else:
#                 length_of_gap = gaplist.pop(0)
#                 seq[sequence_pointer:sequence_pointer+length_of_gap] = '-'
#                 sequence_pointer += length_of_gap

#             # We alternate between contigs and gaps
#             iscontig = not iscontig

#     return sequences

# =============================================================================
def noisify_by_random_contigs(seqs, noise_percent, percent_gaps, numgaps, mingapsize, mincontigsize):

    sequences = deepcopy(seqs)
    for seq in sequences:
        classes = set(c for c in seq)

        amino_acid_length = len(seq)
        amino_acids_to_noisify = int(noise_percent * amino_acid_length)
        amino_acids_to_gapify = int(amino_acids_to_noisify * percent_gaps)
        amino_acids_to_change = amino_acids_to_noisify - amino_acids_to_gapify

        print(f"amino_acid_length: {amino_acid_length}")
        print(f"amino_acids_to_noisify: {amino_acids_to_noisify}")
        print(f"amino_acids_to_gapify: {amino_acids_to_gapify}")
        print(f"amino_acids_to_change: {amino_acids_to_change}")

        nongap_amino_acids = amino_acid_length - amino_acids_to_gapify

        # Randomly decide where to partition the amino acids (i.e. dividers)
        dividers = sorted(random.sample(range(amino_acids_to_gapify), numgaps - 1))

        # Then compute the gap lengths according to that partition
        gaplist = [second - first for first, second in zip([0] + dividers, dividers + [amino_acids_to_gapify])]

        # So gaplist should have fixed length
        # But contiglist will have a length somewhere between numgaps - 1 and numgaps + 1
        # There are three possibilities:

        # Flip a three-sided dice:
        choice = random.sample(range(3), 1)[0]

        # 1. numcontigs == numgaps, which means the first thing may be either a gap or a contig
        if choice == 0:
            start_with_contig = random.sample(range(2), 1)[0] == 0
            numcontigs = numgaps

        # 2. numcontigs == numgaps + 1, which means that the first thing is a contig
        elif choice == 1:
            start_with_contig = True
            numcontigs = numgaps + 1

        # 3. numcontigs == numgaps - 1, which means that the first and last are gaps
        elif choice == 2:
            start_with_contig = False
            numcontigs = numgaps - 1

        # Randomly decide where to partition the contig amino acids (i.e. dividers)
        dividers = sorted(random.sample(range(nongap_amino_acids), numcontigs - 1))

        # Then compute the contig lengths according to that partition
        contiglist = [second - first for first, second in zip([0] + dividers, dividers + [nongap_amino_acids])]

        # Now, we combine gaplist and contiglist to determine which indices need to be replaced
        is_contig = start_with_contig

        seq_ptr = 0
        while gaplist or contiglist:

            if is_contig:
                seq_ptr = seq_ptr + contiglist.pop(0)
            else:
                gap_length = gaplist.pop(0)
                seq[seq_ptr:seq_ptr + gap_length] = "-"
                seq_ptr += gap_length

            is_contig = not is_contig

        # We need to also randomly mess up some portion of the amino acids in the nongaps
        nongap_indices = np.where(seq != "-")[0]
        np.random.shuffle(nongap_indices)
        nongap_indices = nongap_indices[:amino_acids_to_change]

        for i in nongap_indices:
            seq[i] = random.sample(list(classes.difference(seq[i])), 1)[0]

    return sequences

# =============================================================================
def main():

    target_sequence = get_sequences("target_sequence.txt")[0]

    percent_gaps = 0.6 # sixty percent of the noise will be gaps, forty will be wrong amino acids
    percent_missings = np.array([0.20, 0.30, 0.40])
    num_gaps = np.array([2, 4, 6, 8, 10])

    numchars = 60
    numrows = int(len(target_sequence) / numchars) + 1

    for percent_missing in percent_missings:
        for num_gap in num_gaps:
            denovo_sequence = noisify_by_random_contigs([target_sequence], percent_missing, percent_gaps, num_gap, 1, 1)[0]

            # gap_indices = np.where(denovo_sequence == '-')[0]
            # incorrect_indices = np.where(denovo_sequence != target_sequence)[0]
            # correct_indices = np.where(denovo_sequence == target_sequence)[0]
            # incorrect_non_gaps = np.setdiff1d(incorrect_indices, gap_indices)
            # print(f"Length of target: {len(target_sequence)}")
            # print(f"Number of incorrect non-gaps: {len(incorrect_non_gaps)}")
            # print(f"Number of gaps: {len(gap_indices)}")

            # print_sequence(target_sequence, "Protein Scaffold", incorrect_indices, correct_indices)

            filename = f"denovo_{percent_missing:.2f}_{num_gap}.txt"

            filecontents = [f">{filename}"]
            filecontents.extend(["".join(denovo_sequence[row * numchars:(row+1)*numchars]) for row in range(numrows)])
            for line in filecontents:
                print(line)

            with open(filename, "w+") as f:
                f.write("\n".join(filecontents))

if __name__ == '__main__':
    main()

