# tokenizer.py
import torch
import pickle
import os

class BiologicalTokenizer:
    def __init__(self, df):
        # Sequence vocab
        self.vocab = sorted(set("ACGU"))  # RNA letters
        self.char_to_idx = {c:i+1 for i,c in enumerate(self.vocab)}  # +1 for padding
        self.char_to_idx['<PAD>'] = 0

        # miRNA vocab
        self.mirna_char_to_idx = {mir:i+1 for i, mir in enumerate(df['miRNA'].unique())}

        # Gene vocab
        self.gene_to_idx = {g:i+1 for i,g in enumerate(df['genes'].unique())}

    def tokenize_sequence(self, seq):
        ids = [self.char_to_idx.get(c,0) for c in seq]
        mask = [1]*len(ids)
        return torch.tensor(ids), torch.tensor(mask)

    def tokenize_mirna(self, mirna):
        return torch.tensor(self.mirna_char_to_idx.get(mirna,0))

    def tokenize_gene(self, gene):
        return torch.tensor(self.gene_to_idx.get(gene,0))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
