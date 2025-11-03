import torch
import torch.nn as nn

class DefaultMLP(nn.Module):
    def __init__(self, hidden_layers: list[int]):
        super().__init__()
        layers = []
        for h_dim in hidden_layers:
            layers.append(nn.LazyLinear(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(0.2))

        # couche finale : f_theta(x)
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retourne le log-risk f_theta(x)
        et l'exponentielle exp(f_theta(x)) pour h(t|x) = h0(t) * exp(f_theta(x))
        """
        f_theta = self.model(x)
        h_x = torch.exp(f_theta)  # intensité relative
        return h_x