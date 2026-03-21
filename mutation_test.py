from Bio.Seq import Seq

def calculate_gc(seq):
    gc_count = seq.count("G") + seq.count("C")
    return (gc_count / len(seq)) * 100

print("=== Enhanced Mutation Simulator ===")
sequence = input("Enter DNA sequence (A/T/G/C only): ").upper()

# Validate sequence
if not all(c in "ATGC" for c in sequence):
    print("Error: Invalid DNA sequence.")
    exit()

# Pad sequence to multiple of 3 for translation
remainder = len(sequence) % 3
if remainder != 0:
    sequence += "N" * (3 - remainder)

print(f"Sequence length: {len(sequence)}")
print(f"GC content: {calculate_gc(sequence):.2f}%")

# Get mutation input
try:
    position = int(input(f"Enter position to mutate (1-{len(sequence)}): "))
    if position < 1 or position > len(sequence):
        print("Error: Position out of range.")
        exit()
except ValueError:
    print("Error: Position must be an integer.")
    exit()

new_base = input("Enter new nucleotide (A/T/G/C): ").upper()
if new_base not in "ATGC":
    print("Error: Invalid nucleotide.")
    exit()

# Apply mutation
mut_seq = sequence[:position-1] + new_base + sequence[position:]
mut_protein = str(Seq(mut_seq).translate())
orig_protein = str(Seq(sequence).translate())

# Show results
print("\n--- Results ---")
print(f"Original DNA: {sequence}")
print(f"Mutated DNA:  {mut_seq}")
print(f"Original Protein: {orig_protein}")
print(f"Mutated Protein:  {mut_protein}")

# Classify mutation
if mut_protein == orig_protein:
    mutation_type = "Silent Mutation"
    interpretation = "The mutation did not change the protein sequence."
elif "*" in mut_protein and "*" not in orig_protein:
    mutation_type = "Nonsense Mutation"
    interpretation = "The mutation introduced a stop codon, truncating the protein."
else:
    mutation_type = "Missense Mutation"
    interpretation = "The mutation changed an amino acid, which may alter protein function."

print(f"Mutation Type: {mutation_type}")
print(f"Interpretation: {interpretation}")

# Optional: amino acid summary
aa_counts = {}
for aa in mut_protein:
    if aa not in aa_counts:
        aa_counts[aa] = 0
    aa_counts[aa] += 1

print("\nMutated Protein Amino Acid Composition:")
for aa, count in aa_counts.items():
    print(f"{aa}: {count}")
