import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, FeaturesLinear


# sparse feature 사이의 상호작용을 효율적으로 계산합니다.
class FMLayer_Sparse(nn.Module):
    def __init__(self, field_dims:list, factor_dim:int):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, factor_dim)


    def square(self, x):
        return torch.pow(x,2)
    

    def forward(self, x: torch.Tensor):
        # x =               # FILL HERE : Use `self.embedding` #
        # square_of_sum =   # FILL HERE : Use `torch.matmul()` and `self.square()` #
        # sum_of_square =   # FILL HERE : Use `torch.matmul()` and `self.square()` #
        x = self.embedding(x)
        square_of_sum = self.square(torch.sum(x, dim=1))
        sum_of_square = torch.sum(self.square(x), dim=1)
        
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
    


class FactorizationMachine(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim

        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FMLayer_Sparse(self.field_dims, self.factor_dim)


    def forward(self, x: torch.Tensor):
        y = self.linear(x).squeeze(1) + self.fm(x)

        return y


