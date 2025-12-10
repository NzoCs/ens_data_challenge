import torch
import torch.nn as nn

class DefaultMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LeakyReLU(0.1))
            in_dim = h_dim
            

        # couche finale : f_theta(x)
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retourne le log-risk f_theta(x)
        et l'exponentielle exp(f_theta(x)) pour h(t|x) = h0(t) * exp(f_theta(x))
        """
        f_theta = self.model(x)
        return f_theta