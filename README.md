# BEGAN-SVS

BEGAN을 이용한 Singing Voice Synthesis(가창합성) 모델의 성능 향상 연구 프로젝트입니다. 본 연구에서는 이미지 생성에 최적화 되어있던 original BEGAN모델을 오디오 생성모델(SVS모델)에 적용시킬 때 generator loss에 추가된 L1 loss가 과적합문제를 일으켜 본래 감마값의 정의를 변질시킨다는 문제를 제기했고,  generator loss 함수에 L1 loss의 가중치 alpha를 도입하고 조절함으로써 이 문제를 해결하는 방안을 제시하였습니다. 또한 BEGAN SVS모델이 가장 고품질의 오디오를 생성해내는 alpha, gamma 값을 제안합니다. **(본 모델은 한국어로 된 노래만을 지원합니다.)**
(포스터 삽입)

 - 포스터 세션 및 시연 영상: https://www.youtube.com/watch?v=Lfbhz74w3Sw
 - 결과 확인: https://livviee.github.io/BEGAN-Sing/


# 실행순서

### 0. Requirements
  - Python 3.6 ~ 3.8
  - PyTorch 1.5
  - Torchaudio 0.5 
  ```
  pip install -r requirements.txt
  ```

### 1. Preprocess
  - config/default_train.yml 파일의 dataset_path 에 전처리 할 데이터셋의 경로를 설정해야 합니다.
  - 전처리를 위한 모든 데이터셋은 'sample_dataset'폴더 내의 구조를 따라야 합니다.
  - 각 노래는 음정과 박자의 정보를 담고있는 MIDI 파일, 가사를 담고 있는 text 파일, 그리고 노래 하는 목소리의 wav파일 총 3가지 파일로 구성됩니다.
  - 각 노래의 MIDI, text, audio는 모두 정렬된 상태여야 합니다.
  
(참고: https://github.com/emotiontts/emotiontts_open_db/tree/master/Dataset/CSD)
  - 전처리 된 데이터셋은 feature/default 에 저장됩니다(config/default_train.yml 파일의 feature_path에서 경로를 수정 할 수 있습니다).

```
python preprocess.py -c config/default_train.yml
```

### 2. Train
  - 전처리 된 학습데이터셋으로 SVS 모델을 학습시킵니다. config/default_train.yml에서 stop_epoch, save_epoch 그리고 gamma 값을 설정할 수 있습니다.
  - train.py의 lossG = lossL1 + loss_advG 에서 설정하고자 하는 alpha 값을 lossL1에 곱해줍니다. (예: alpha=0.5 일 때, lossG = 0.5*lossL1 + loss_advG)
  - 사용할 gpu device들을 --device 로 설정합니다. (예: gpu 2개 -> --device 0,1)
  - 학습 된 모델은 checkpoint/default에 저장됩니다.  

```
python train.py -c config/default_train.yml --device 0 --batch_size 32
```

### 3. Inference
  - 모델 학습이 완료되면, 가사 text 파일과 MIDI 파일을 통해 가창합성 된 오디오를 생성 할 수 있습니다. (text와 MIDI는 마찬가지로 align된 상태여야 합니다.)
  - config/default_infer.yml의 text_file에 inference 하고자 하는 text 파일의 경로를 설정합니다.(MIDI 파일은 text파일 이름과 같다고 간주됩니다.) 
  - 음성합성결과는 설정한 checkpoint 경로 내에 생성됩니다.
```
python infer.py -c config/default_train.yml config/default_infer.yml --device 0

```

### Results
2시간 38분 분량의 어린이 동요 데이터셋(CSD)으로 alpha,gamma 값을 달리하여 총 9가지 모델을 학습 한 결과입니다. Inference는 아이유의 
 - 결과 확인 : https://livviee.github.io/BEGAN-Sing/
