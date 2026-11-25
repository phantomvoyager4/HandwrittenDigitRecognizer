import torch
import torch.nn as nn

class NN_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()            
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
                                    )       
    def forward(self, x: torch.Tensor):
        return self.network(x)


if __name__ == "__main__":
    nn_1 = NN_Model()
    print(nn_1(torch.rand([40,784])).shape)
    print(nn_1.state_dict)
