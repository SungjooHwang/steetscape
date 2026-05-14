1. Conda Installation
Bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda init
2. Environment Setup
Bash
# Create environment
conda create -n streetscape python=3.8 -y
conda activate streetscape

# Install CUDA toolkit
conda install cudatoolkit=11.3 -c nvidia

# Install PyTorch and Dependencies
pip install torch==1.10.1 torchvision==0.11.2 --index-url https://download.pytorch.org/whl/cu113
pip install natten==0.14.4 -f https://shi-labs.com/natten/wheels/cu113/torch1.10.1/index.html
python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
3. Model Weight Downloads
You can download the pre-trained weights using wget or your browser:

ADE20K Weights

Cityscapes Weights

COCO Weights

Mapillary Vistas Weights

4. Running Inference
Single Data Inference
After downloading the config and weights from the OneFormer Repository:

Bash
python material_demo/inference_with_material.py \
  --config-file configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml \
  --task panoptic \
  --input data/inputs/* \
  --output data/outputs \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS 250_16_dinat_l_oneformer_mapillary_300k.pth
Batch Inference (All Data)
Bash
# General batch inference
python material_demo/inference_all.py --input data/inputs/* --output data/outputs

# AI Hub data specific inference
python src/inference/inference_all.py --input data/input/AI_hub/* --output data/output/AI_hub
5. Data Integration
Bash
python src/merge_data/make_list.py --output-path ./data/output/AI_hub
