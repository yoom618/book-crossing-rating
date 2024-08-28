######## 기본 베이스라인 실행 스크립트 ########
python main.py --config config_baseline.yaml --model FM --device mps
python main.py --config config_baseline.yaml --model FFM --device mps
python main.py --config config_baseline.yaml --model DeepFM --device mps
python main.py --config config_baseline.yaml --model WDN --device mps
python main.py --config config_baseline.yaml --model DCN --device mps
python main.py --config config_baseline.yaml --model NCF --device mps
python main.py --config config_baseline.yaml --model Image_FM --device mps
python main.py --config config_baseline.yaml --model Image_DeepFM --device mps
python main.py --config config_baseline.yaml --model Text_FM --device mps
python main.py --config config_baseline.yaml --model Text_DeepFM --device mps

######## 추가 베이스라인 실행 스크립트 ########
# # 텍스트 벡터를 새로 만들고자 할 경우
# python main.py --config config.yaml --model Text_FM --vector_create True

# # 학습 없이 saved_model을 불러와 추론만 하고자 할 경우
# # 예) 저장된 파일명이 20240827_035641_Image_DeepFM.pt이고, configuration이 20240827_035641_Image_DeepFM.yaml에 저장되었다면,
# python main.py --config 20240827_035641_Image_DeepFM.yaml --model Image_DeepFM  --predict True  --checkpoint  20240827_035641_Image_DeepFM.pt

# # 앙상블 실행
# # /submit/20240728_133605_NCF.csv 인 경우, `20240728_133605_NCF`만 작성해주면 됨
# !python ensemble.py --ensemble_files 20240728_133228_Image_FM 20240728_133605_NCF --ensemble_strategy weighted --ensemble_weight 0.5,0.5