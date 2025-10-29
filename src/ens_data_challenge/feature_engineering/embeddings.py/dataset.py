from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch

class PersonDataset(Dataset):
    def __init__(self, person_sets, labels):
        self.person_sets = person_sets
        self.labels = labels

    def __len__(self):
        return len(self.person_sets)

    def __getitem__(self, idx):
        return self.person_sets[idx], self.labels[idx]

def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = [len(s) for s in seqs]
    padded = pad_sequence(seqs, batch_first=True)
    mask = torch.arange(padded.size(1))[None, :] >= torch.tensor(lengths)[:, None]
    return padded, mask, torch.tensor(labels)

# Exemple
person_sets = [torch.randn(torch.randint(3, 8, (1,)), 16) for _ in range(20)]
labels = torch.arange(20)
loader = DataLoader(PersonDataset(person_sets, labels), 
                    batch_size=8, collate_fn=collate_fn, shuffle=True)
