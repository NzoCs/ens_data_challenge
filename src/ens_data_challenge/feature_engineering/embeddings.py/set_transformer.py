import torch
import torch.nn as nn
import torch.nn.functional as F

# === Encodeur de sets (avec attention) ===
class SetTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, seqs, mask=None):
        z = self.embed(seqs)
        z = self.encoder(z, src_key_padding_mask=mask)
        pooled = z.mean(dim=1)  # attention pooling possible
        return self.readout(pooled)


# === Autoencodeur discriminatif ===
class SetAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, emb_dim, n_persons):
        super().__init__()
        self.encoder = SetTransformer(input_dim, hidden_dim, emb_dim)
        self.decoder = nn.Linear(emb_dim, n_persons)

    def forward(self, seqs, mask=None):
        h = self.encoder(seqs, mask)         # (batch, emb_dim)
        logits = self.decoder(h)             # (batch, n_persons)
        return logits, h


# === Exemple d’entraînement ===
def train_autoencoder(model, dataloader, n_epochs=10, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        total_loss = 0
        for seqs, masks, labels in dataloader:
            seqs, masks, labels = seqs.to(device), masks.to(device), labels.to(device)

            logits, _ = model(seqs, masks)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seqs.size(0)

        print(f"Epoch {epoch+1}: loss = {total_loss / len(dataloader.dataset):.4f}")
