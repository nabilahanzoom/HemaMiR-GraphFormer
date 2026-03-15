# predict.py
import torch
import pickle

from tokenizer import BiologicalTokenizer
from model import MiRNADiseaseTransformer

# ------------------------------
# Load encoders
# ------------------------------
dis_enc = pickle.load(open("saved_models/disease_encoder.pkl", "rb"))
ctx_enc = pickle.load(open("saved_models/context_encoder.pkl", "rb"))

# ------------------------------
# Load tokenizer
# ------------------------------
tok = BiologicalTokenizer.load("saved_models/tokenizer.pth")

# ------------------------------
# Load model
# ------------------------------
model = MiRNADiseaseTransformer(
    vocab_size=len(tok.char_to_idx),
    mirna_vocab=len(tok.mirna_char_to_idx),
    gene_vocab=len(tok.gene_to_idx),
    num_disease=len(dis_enc.classes_),
    num_context=len(ctx_enc.classes_)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("saved_models/model.pth", map_location=device))
model.to(device)
model.eval()

# ------------------------------
# Prediction function
# ------------------------------
def predict(sequence, mirna, gene):
    seq_ids, mask = tok.tokenize_sequence(sequence)
    mirna_id = tok.tokenize_mirna(mirna)
    gene_id = tok.tokenize_gene(gene)

    # Add batch dimension
    seq_ids = seq_ids.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    mirna_id = mirna_id.unsqueeze(0).to(device)
    gene_id = gene_id.unsqueeze(0).to(device)
    ctx_id = torch.zeros(1, dtype=torch.long).to(device)  # dummy input

    with torch.no_grad():
        dlog, clog = model(seq_ids, mask, mirna_id, gene_id, ctx_id)
        disease_pred = dis_enc.inverse_transform([torch.argmax(dlog, dim=1).item()])[0]
        context_pred = ctx_enc.inverse_transform([torch.argmax(clog, dim=1).item()])[0]

    return disease_pred, context_pred

# ------------------------------
# Example
# ------------------------------
sequence = "UCCCUGAGACCCUUUAACCUGUGA"
mirna = "hsa-mir-125a"
gene = "BMF"

disease, context = predict(sequence, mirna, gene)
print(f"Predicted Disease: {disease}")
print(f"Predicted Context: {context}")
