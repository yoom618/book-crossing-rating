import os
import argparse
from omegaconf import OmegaConf
import pandas as pd
import torch
from src.utils import Logger, Setting
import src.data as data_module
from src.train import train, test
import src.models as model_module


def main(args, wandb=None):
    Setting.seed_everything(args.seed)

    ######################## LOAD DATA
    datatype = args.model_args[args.model].datatype
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

    logger = Logger(args, log_path)
    logger.save_args()


    ######################## Model
    print(f'--------------- INIT {args.model} ---------------')
    # models > __init__.py 에 저장된 모델만 사용 가능
    # model = FM(args.model_args.FM, data).to('cuda')와 동일한 코드
    model = getattr(model_module, args.model)(args.model_args[args.model], data).to(args.device)

    # 만일 기존의 모델을 불러와서 학습을 시작하려면 resume을 true로 설정하고 resume_path에 모델을 지정하면 됨
    if args.train.resume:
        model.load_state_dict(torch.load(args.train.resume_path, weights_only=True))


    ######################## TRAIN
    if not args.predict:
        print(f'--------------- {args.model} TRAINING ---------------')
        model = train(args, model, data, logger, setting)


    ######################## INFERENCE
    if not args.predict:
        print(f'--------------- {args.model} PREDICT ---------------')
        predicts = test(args, model, data, setting)
    else:
        print(f'--------------- {args.checkpoint} PREDICT ---------------')
        predicts = test(args, model, data, setting, args.checkpoint)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')
    submission['rating'] = predicts

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    

    arg = parser.add_argument

    # add basic arguments (no default value)
    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', required=True)
    arg('--predict', '-p', '--p', '--pred', type=bool, 
        help='학습을 생략할지 여부를 설정할 수 있습니다.')
    arg('--checkpoint', '-ckpt', '--ckpt', type=str, 
        help='학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.')
    arg('--model', '-m', '--m', type=str, 
        choices=['FM', 'FFM', 'DeepFM', 'NCF', 'WDN', 'DCN', 'Image_FM', 'Image_DeepFM', 'Text_FM', 'Text_DeepFM'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--wandb', '--w', '-w', type=bool, 
        help='wandb를 사용할지 여부를 설정할 수 있습니다.')
    arg('--wandb_project', '--wp', '-wp', type=str,
        help='wandb 프로젝트 이름을 설정할 수 있습니다.')
    arg('--run_name', '--rn', '-rn', '--r', '-r', type=str,
        help='wandb에서 사용할 run 이름을 설정할 수 있습니다.')
    
    args = parser.parse_args()


    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config)

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]
    config_yaml.model_args = OmegaConf.create({config_yaml.model : config_yaml.model_args[config_yaml.model]})  # 사용되지 않는 model들의 정보 삭제
    # print(OmegaConf.to_yaml(config_yaml))
    
    ######################## W&B
    if args.wandb:
        import wandb
        wandb.require("core")
        # https://docs.wandb.ai/ref/python/init 참고
        wandb.init(project=config_yaml.wandb_project, 
                   config=OmegaConf.to_container(config_yaml, resolve=True),
                   name=config_yaml.run_name if config_yaml.run_name else None,
                   notes=config_yaml.memo,
                   tags=[config_yaml.model],
                   resume="allow")

        wandb.run.log_code("./src")  # src 내의 모든 파일을 업로드. Artifacts에서 확인 가능

    ######################## MAIN
    main(config_yaml)

    if args.wandb:
        wandb.finish()