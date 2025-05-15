import torch
import torch.nn as nn

# Vokabular (Deutsch → Index)
de_vocab = {"Ich": 0, "liebe": 1, "KI": 2, "<PAD>": 3}
# Vokabular (Englisch → Index)
en_vocab = {"I": 0, "love": 1, "AI": 2, "<PAD>": 3}

# Umkehr-Maps für Ausgabe
inv_en_vocab = {v: k for k, v in en_vocab.items()}
word_map = {0: 0, 1: 1, 2: 2}  # Feste Übersetzungsregel (Index)

# Eingabesequenz: "Ich liebe KI"
input_seq = torch.tensor([[0, 1, 2]])  # Batch = 1, Länge = 3

# Encoder: Wandelt Wort-IDs in Vektoren
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)  # shape: (batch, seq_len, embedding_dim)

# Decoder: Übersetzt Vektoren via Mapping (statisch, ohne Training)
class Decoder(nn.Module):
    def forward(self, encoded_ids):
        # Simuliere „Übersetzung“ durch Wort-Mapping
        translated = torch.tensor([[word_map[int(i)] for i in encoded_ids[0]]])
        return translated

# Initialisierung
embedding_dim = 4
encoder = Encoder(len(de_vocab), embedding_dim)
decoder = Decoder()

# Durch Encoder (für Demo: nur Weitergabe der Token-IDs)
encoded = input_seq  # normal: encoder(input_seq), hier: wir übersetzen nur Indexe

# Durch Decoder
output_ids = decoder(encoded)
output_words = [inv_en_vocab[int(i)] for i in output_ids[0]]

# Ergebnis
print("Input (DE):", [list(de_vocab.keys())[int(i)] for i in input_seq[0]])
print("Output (EN):", output_words)
