# model.py
import torch
import torch.nn as nn

class MiRNADataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        seq, mask = self.tokenizer.tokenize_sequence(r['sequence'])
        return {
            "sequence_ids": seq,
            "attention_mask": mask,
            "mirna_ids": self.tokenizer.tokenize_mirna(r['miRNA']),
            "gene_ids": self.tokenizer.tokenize_gene(r['genes']),
            "disease_labels": torch.tensor(r['Disease_label'], dtype=torch.long),
            "context_labels": torch.tensor(r['Context_label'], dtype=torch.long)
        }


class MiRNADiseaseTransformer(nn.Module):
    def __init__(self, vocab_size, mirna_vocab, gene_vocab, num_disease, num_context):
        super().__init__()
        self.embed = nn.Embedding(vocab_size+1, 128, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=2
        )
        self.mirna_emb = nn.Embedding(mirna_vocab+1, 128)
        self.gene_emb = nn.Embedding(gene_vocab+1, 128)
        self.context_emb = nn.Embedding(num_context, 64)

        self.fc_disease = nn.Linear(128*3 + 64, num_disease)
        self.fc_context = nn.Linear(128*3, num_context)

    def forward(self, seq, mask, mirna, gene, ctx):
        # Sequence embedding
        x = self.embed(seq).transpose(0,1)
        x = self.encoder(x).mean(0)  # [batch, 128]

        # miRNA and gene embedding
        m = self.mirna_emb(mirna).unsqueeze(1).mean(1)
        g = self.gene_emb(gene).unsqueeze(1).mean(1)

        # Context embedding
        c = self.context_emb(ctx)

        # Disease prediction
        disease_logits = self.fc_disease(torch.cat([x, m, g, c], dim=1))
        # Context prediction
        context_logits = self.fc_context(torch.cat([x, m, g], dim=1))
        return disease_logits, context_logits
