import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense



# 텍스트 특징 추출을 위한 기초적인 CNN 1D Layer를 정의합니다.
class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, conv_1d_out_dim=50):
        super(TextCNN, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv1d(
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding
                                        ),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(kernel_size, 1)),
                                nn.Dropout(p=0.5)
                                )
        
        out_dim = (in_channels - kernel_size + 2 * padding) // stride + 1
        self.linear = nn.Sequential(
                                    nn.Linear(int(out_dim/kernel_size), conv_1d_out_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5))


    def forward(self, vec):
        output = self.conv(vec)
        output = self.linear(output.reshape(-1, output.size(1)))
        return output


# 기존 유저/상품 벡터와 유저/상품 리뷰 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
class DeepCoNN(nn.Module):
    def __init__(self, args, data):
        super(DeepCoNN, self).__init__()
        self.field_dims = data.field_dims

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)
        
        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        # 텍스트 feature를 임베딩하는 부분
        self.cnn_u = TextCNN(
                             in_channels=args.word_dim,
                             out_channels=args.conv_1d_out_dim,
                             kernel_size=args.kernel_size,
                             out_dim=args.out_dim,
                            )
        self.cnn_i = TextCNN(
                             in_channels=args.word_dim,
                             out_channels=args.conv_1d_out_dim,
                             kernel_size=args.kernel_size,
                             out_dim=args.out_dim,
                            )
        
        # sparse 및 dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense(
                                input_dim=(args.embed_dim * len(self.field_dims)) + (args.out_dim * 2),
                                latent_dim=args.dense_latent_dim,
                               )


    def forward(self, x):
        user_book_vector, user_text_vector, item_text_vector = x[0], x[1], x[2]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector).squeeze(1)

        # second-order interaction / all feature
        user_book_embedding = self.embedding(user_book_vector)
        user_book_embedding = user_book_embedding.view(-1, user_book_embedding.size(1) * user_book_embedding.size(2))
        user_text_feature = self.cnn_u(user_text_vector)
        item_text_feature = self.cnn_i(item_text_vector)
        dense_feature = torch.cat([user_book_embedding, user_text_feature, item_text_feature], dim=1)
        second_order = self.fm(dense_feature)

        return first_order + second_order
