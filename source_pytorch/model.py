import torch
import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class ModelArgs:
    input_size: int = 784
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64])
    output_size: int = 10


class NN_Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()            
        initial_list = [args.input_size, *args.hidden_sizes, args.output_size]
        net = []
        for in_features, out_features in zip(initial_list, initial_list[1:]):
            net.append(nn.Linear(in_features, out_features))
            net.append(nn.ReLU())
        net.pop() #no ReLU at the end
        self.network = nn.Sequential(*net)       
    def forward(self, x: torch.Tensor):
        return self.network(x)
