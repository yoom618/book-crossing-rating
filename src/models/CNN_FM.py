import numpy as np
import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, CNN_Base, MLP_Base



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
        
        # 이미지 feature를 cnn을 통해 임베딩하는 부분
        self.cnn = CNN_Base(
                            input_size=(3, args.img_size, args.img_size),  # default: (3, 224, 224)
                            channel_list=args.channel_list,                # default: [4, 8, 16]
                            kernel_size=args.kernel_size,                  # default: 3
                            stride=args.stride,                            # default: 2
                            padding=args.padding,                          # default: 1
                           )
        
        # cnn을 통해 임베딩된 이미지 벡터가 fm에 사용될 수 있도록 embed_dim 크기로 변환하는 부분
        self.cnn_embedding = nn.Linear(np.prod(self.cnn.output_dim), args.embed_dim)
        
        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense(
                                input_dim=(args.embed_dim * len(self.field_dims)) + np.prod(self.cnn.output_dim),
                                latent_dim=args.dense_latent_dim
                                )


    def forward(self, x):
        user_book_vector, img_vector = x[0], x[1]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)

        # image to dense
        img_feature = self.cnn(img_vector)  # (batch_size, out_channels, H, W)
        img_feature = img_feature.view(-1, np.prod(self.cnn.output_dim))  # (batch_size, out_channels * H * W)
        img_feature = self.cnn_embedding(img_feature)  # (batch_size, embed_dim)
        img_feature = img_feature.view(-1, 1, self.embed_dim)  # (batch_size, 1, embed_dim)
        
        # second-order interaction / dense
        dense_feature = torch.cat([user_book_embedding, img_feature], dim=1)  # (batch_size, num_fields + 1, embed_dim)
        second_order = self.fm(dense_feature)  # (batch_size,)

        return first_order.squeeze(1) + second_order




class CNN_DeepFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # 이미지 feature를 cnn을 통해 임베딩하는 부분
        self.cnn = CNN_Base(
                            input_size=(3, args.img_size, args.img_size),  # default: (3, 224, 224)
                            channel_list=args.channel_list,                # default: [4, 8, 16]
                            kernel_size=args.kernel_size,                  # default: 3
                            stride=args.stride,                            # default: 2
                            padding=args.padding                           # default: 1
                           )
        
        # cnn을 통해 임베딩된 이미지 벡터가 fm에 사용될 수 있도록 embed_dim 크기로 변환하는 부분
        self.cnn_embedding = nn.Linear(np.prod(self.cnn.output_dim), args.embed_dim)
        
        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()

        # deep network를 통해 dense feature를 학습하는 부분
        self.deep = MLP_Base(
                             input_dim=(args.embed_dim * (len(self.field_dims) + 1)),
                             embed_dims=args.mlp_dims,
                             dropout=args.dropout
                            )


    def forward(self, x):
        user_book_vector, img_vector = x[0], x[1]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)

        # image to dense
        img_feature = self.cnn(img_vector)  # (batch_size, out_channels, H, W)
        img_feature = img_feature.view(-1, np.prod(self.cnn.output_dim))  # (batch_size, out_channels * H * W)
        img_feature = self.cnn_embedding(img_feature)  # (batch_size, embed_dim)
        img_feature = img_feature.view(-1, 1, self.embed_dim)  # (batch_size, 1, embed_dim)
        
        # second-order interaction / dense feature
        dense_feature = torch.cat([user_book_embedding, img_feature], dim=1)  # (batch_size, num_fields + 1, embed_dim)
        second_order = self.fm(dense_feature)  # (batch_size,)

        # deep network를 통해 feature를 학습하는 부분
        deep_out = self.deep(dense_feature)  # (batch_size, 1)

        return first_order.squeeze(1) + second_order + deep_out.squeeze(1)