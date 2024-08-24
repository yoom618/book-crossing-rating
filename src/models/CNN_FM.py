import numpy as np
import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, FMLayer_Dense, CNN_Base



# 기존 유저/상품 벡터와 이미지 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
# DeepFM 모델과 유사하게 모델을 작성하되, Deep Embedding 부분에 CNN을 통해 임베딩된 이미지 벡터를 추가합니다.
class CNN_FM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.cnn_embed_dim)

        self.cnn = CNN_Base(
                            input_size=(3, 64, 64),
                            channel_list=[6, 12],
                            kernel_size=3,
                            stride=2,
                            padding=1
                           )
        self.fm = FMLayer_Dense(
                                input_dim=(args.cnn_embed_dim * 2) + (12 * 1 * 1),
                                latent_dim=args.cnn_latent_dim
                                )


    def forward(self, x):
        user_book_vector, img_vector = x[0], x[1]
        user_book_embedding = self.embedding(user_book_vector)
        user_book_embedding = user_book_embedding.view(-1, user_book_embedding.size(1) * user_book_embedding.size(2))
        img_feature = self.cnn(img_vector)

        feature_vector = torch.cat([
                                    user_book_embedding,
                                    img_feature
                                    ], dim=1)
        
        output = self.fm(feature_vector)
        return output.squeeze(1)
