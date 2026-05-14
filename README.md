### conda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda init

### 환경설치
conda create -n streetscape python=3.8

conda install cudatoolkit=11.3

pip install torch==1.10.1 torchvision==0.11.2  --index-url https://download.pytorch.org/whl/cu113

pip install natten==0.14.4 -f https://shi-labs.com/natten/wheels/cu113/torch1.10.1/index.html

python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth

https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth

https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth

https://shi-labs.com/projects/oneformer/mapillary/250_16_dinat_l_oneformer_mapillary_300k.pth


하나의 데이터에 대해서 돌리기
config, model.weights 파일 받은 후 (https://github.com/SHI-Labs/OneFormer)
python material_demo/inference_with_material.py --config-file configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml --task panoptic --input data/(inputs 폴더)/* --output data/(outputs 폴더) --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS 250_16_dinat_l_oneformer_mapillary_300k.pth

전체 데이터에 대해서 돌리기
python material_demo/inference_all.py --input data/(input 폴더)/* --output data/(outputs 폴더)

python src/inference/inference_all.py --input data/input/AI_hub/* --output data/output/AI_hub

### 데이터 통합
python src/merge_data/make_list.py --output-path ./data/output/AI_hub