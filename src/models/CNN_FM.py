import numpy as np
import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, CNN_Base



# 기존 유저/상품 벡터와 이미지 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
# FM 모델과 유사하게 모델을 작성하되, second-order interaction 부분에 CNN을 통해 임베딩된 이미지 벡터를 추가하여 교호작용을 계산합니다.
class CNN_FM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        # 이미지 feature를 dense하게 임베딩하는 부분
        self.cnn = CNN_Base(
                            input_size=(3, args.img_size, args.img_size),  # default: (3, 224, 224)
                            channel_list=args.channel_list,                # default: [4, 8, 16]
                            kernel_size=args.kernel_size,                  # default: 3
                            stride=args.stride,                            # default: 2
                            padding=args.padding                           # default: 1
                           )
        
        # sparse 및 dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense(
                                input_dim=(args.embed_dim * len(self.field_dims)) + np.prod(self.cnn.output_dim),
                                latent_dim=args.dense_latent_dim
                                )


    def forward(self, x):
        user_book_vector, img_vector = x[0], x[1]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector).squeeze(1)

        # second-order interaction / all feature
        user_book_embedding = self.embedding(user_book_vector)
        user_book_embedding = user_book_embedding.view(-1, len(self.field_dims) * self.embed_dim)
        img_feature = self.cnn(img_vector)
        dense_feature = torch.cat([user_book_embedding, img_feature], dim=1)
        second_order = self.fm(dense_feature)

        return first_order + second_order
