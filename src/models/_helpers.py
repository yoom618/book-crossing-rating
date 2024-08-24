import torch
import torch.nn as nn
from numpy import cumsum


# factorization을 통해 얻은 feature를 embedding 합니다.
# 사용되는 모델 : FM, CNN-FM, DCN
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims:list, embed_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = [0, *cumsum(field_dims)[:-1]]
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return self.embedding(x)  # (batch_size, num_fields, embed_dim)


# FM 계열 모델에서 활용되는 선형 결합 부분을 정의합니다.
# 사용되는 모델 : FM, FFM, WDN, CNN-FM
class FeaturesLinear(nn.Module):
    def __init__(self, field_dims:list, output_dim:int=1, bias:bool=True):
        super().__init__()
        self.feature_dims = sum(field_dims)
        self.output_dim = output_dim
        self.offsets = [0, *cumsum(field_dims)[:-1]]

        self.fc = nn.Embedding(self.feature_dims, self.output_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty((self.output_dim,)), requires_grad=True)
    
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
            if isinstance(m, nn.Parameter):
                nn.init.constant_(m, 0)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.fc(x), dim=1) + self.bias if hasattr(self, 'bias') \
               else torch.sum(self.fc(x), dim=1)



# sparse feature 사이의 상호작용을 효율적으로 계산합니다.
# 사용되는 모델 : FM
class FMLayer_Sparse(nn.Module):
    def __init__(self, field_dims:list, factor_dim:int):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, factor_dim)


    def square(self, x):
        return torch.pow(x,2)
    

    def forward(self, x: torch.Tensor):
        # x =               # FILL HERE : Use `self.embedding` #
        # square_of_sum =   # FILL HERE : Use `torch.sum()` and `self.square()` #
        # sum_of_square =   # FILL HERE : Use `torch.sum()` and `self.square()` #
        x = self.embedding(x)
        square_of_sum = self.square(torch.sum(x, dim=1))
        sum_of_square = torch.sum(self.square(x), dim=1)
        
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
    

# dense feature 사이의 상호작용을 효율적으로 계산합니다. (first-order도 포함)
# 사용되는 모델 : DeepCoNN, CNN-FM
class FMLayer_Dense(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim))

    def square(self, x:torch.Tensor):
        return torch.pow(x,2)

    def forward(self, x):
        square_of_matmul = self.square(torch.mm(x, self.v))
        matmul_of_square = torch.mm(self.square(x), self.square(self.v))
        

        return 0.5 * torch.sum(square_of_matmul - matmul_of_square, dim=1)
    


# 기본적인 형태의 MLP를 구현합니다.
# 사용되는 모델 : WDN, DCN, NCF
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    


# 기본적인 형태의 CNN을 정의합니다. 이미지 데이터의 특징을 추출하기 위해 사용됩니다.
# 사용되는 모델 : CNN-FM
class CNN_Base(nn.Module):
    def __init__(self, input_size=(3, 64, 64), 
                 channel_list=[8,16,32], kernel_size=3, stride=2, padding=1):
        super().__init__()

        self.cnn = nn.Sequential()
        in_channel_list = [input_size[0]] + channel_list[:-1]
        for idx, (in_channel, out_channel) in enumerate(zip(in_channel_list, channel_list)):
            self.cnn.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
            self.cnn.append(nn.BatchNorm2d(out_channel))
            self.cnn.append(nn.ReLU())
            if idx % 2 == 0:  # Convolutional Layer 2개마다 MaxPooling을 한 번 적용
                self.cnn.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.output_dim = self.compute_output_shape((1, *input_size))


    def compute_output_shape(self, input_shape):
        x = torch.rand(input_shape)
        for layer in self.cnn:
            x = layer(x)

        return x.size()
        

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        return x