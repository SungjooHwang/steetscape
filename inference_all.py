import os
import glob
import argparse

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    # 새로 추가: 경량화 옵션
    parser.add_argument(
        "--mode", type=int, choices=[1, 2], default=1,
        help="1: full mode (all models do all), 2: optimized mode"
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    models = {
        'ade20k': {
            'config': 'configs/ade20k/dinat/oneformer_dinat_large_bs16_160k.yaml',
            'model': '250_16_dinat_l_oneformer_ade20k_160k.pth'
        },
        'cityscapes': {
            'config': 'configs/cityscapes/dinat/oneformer_dinat_large_bs16_90k.yaml',
            'model': '250_16_dinat_l_oneformer_cityscapes_90k.pth'
        },
        'coco': {
            'config': 'configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml',
            'model': '150_16_dinat_l_oneformer_coco_100ep.pth'
        },
        'mapillary_vistas': {
            'config': 'configs/mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml',
            'model': '250_16_dinat_l_oneformer_mapillary_300k.pth'
        }
    }

    if args.mode == 2:
        models = {k: v for k, v in models.items() if 'cityscapes' not in k.lower()}

    for path in tqdm(args.input):
        filename = os.path.basename(path)
        for model_name, model_info in models.items():
            os.system(
                f'python src/inference/inference_with_material_250710.py '
                f'--config-file {model_info["config"]} '
                f'--input {path} '
                f'--output {args.output}/{model_name}/{filename} '
                f'--mode {args.mode} '
                f'--opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ./models/{model_info["model"]}'
            )
            