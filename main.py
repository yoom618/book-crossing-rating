import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load
import src.data as data_module
from src.train import train, test


def main(args):
    Setting.seed_everything(args.seed)

    ######################## LOAD DATA
    if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN'):
        # datatype = 'basic'
        datatype = 'context'
    elif args.model == 'CNN_FM':
        datatype = 'image'
    elif args.model == 'DeepCoNN':
        import nltk
        datatype = 'text'
    else:
        assert False, 'Not Implemented Model'
    
    data_load_fn = getattr(data_module, f'{datatype}_data_load')  # e.g. basic_data_load()
    data_split_fn = getattr(data_module, f'{datatype}_data_split')  # e.g. basic_data_split()
    data_loader_fn = getattr(data_module, f'{datatype}_data_loader')  # e.g. basic_data_loader()

    print(f'--------------- {args.model} Load Data ---------------')
    data = data_load_fn(args)

    print(f'--------------- {args.model} Train/Valid Split ---------------')
    data = data_split_fn(args, data)
    data = data_loader_fn(args, data)


    ####################### Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)

    logger = Logger(args, log_path)
    logger.save_args()


    ######################## Model
    print(f'--------------- INIT {args.model} ---------------')
    model = models_load(args,data)


    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    model = train(args, model, data, logger, setting)


    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    predicts = test(args, model, data, setting)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'):
        submission['rating'] = predicts
    else:
        pass

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다. 만약 0으로 설정하면 모두 Train 데이터로 사용합니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')


    ############### TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE', 'MAE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--metric', type=str, default=['MSE', 'MAE'], help='평가 지표를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')
    arg('--lr_scheduler', type=str, default='', choices=['', 'ReduceLROnPlateau', 'StepLR'], help='Learning Rate Scheduler를 변경할 수 있습니다.')
    
    


    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'], help='학습에 사용할 Device를 조정할 수 있습니다.')


    ############### Common Options for FM, FFM, NCF, WDN, DCN, CNN_FM, DeepCoNN
    arg('--embed_dim', type=int, default=8, help='FM, FFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.')


    ############### NCF, WDN, DCN
    arg('--dropout', type=float, default=0.2, help='NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=list, default=[16, 32], help='NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.')


    ############### DCN
    arg('--num_layers', type=int, default=3, help='에서 Cross Network의 레이어 수를 조정할 수 있습니다.')


    ############### CNN_FM, DeepCoNN
    arg('--kernel_size', type=int, default=3, help='CNN_FM, DEEP_CONN에서 CNN의 kernel 크기를 조정할 수 있습니다.')
    arg('--stride', type=int, default=2, help='CNN_FM, DEEP_CONN에서 CNN의 stride 크기를 조정할 수 있습니다.')
    arg('--padding', type=int, default=1, help='CNN_FM, DEEP_CONN에서 CNN의 padding 크기를 조정할 수 있습니다.')
    arg('--dense_latent_dim', type=int, default=12, help='CNN_FM, DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')


    ############### CNN_FM
    arg('--img_size', type=int, default=224, help='이미지 전처리 시 이미지 크기를 조정할 수 있습니다.')
    arg('--channel_list', type=list, default=[4, 8, 16], help='CNN_FM에서 CNN의 채널 수를 조정할 수 있습니다.')


    ############### DeepCoNN
    arg('--pretained_model', type=str, default='bert-base-uncased', help='책의 요약 정보를 임베딩하기 위한 pretrained model을 설정할 수 있습니다.')
    arg('--vector_create', type=bool, default=False, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--word_dim', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다. (bert-base-uncased 기준 768)')
    arg('--conv_1d_out_dim', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--out_dim', type=int, default=50, help='DEEP_CONN에서 1D conv에 linear를 적용한 뒤 최종적으로 나오는 차원을 조정할 수 있습니다.')


    args = parser.parse_args()
    main(args)
