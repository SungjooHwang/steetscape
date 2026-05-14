# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

## 경량화 버전: --mode {1, 2} 에 따라 다르게 수정
# mode 1: 기존 전체 수행 (현재 동작 그대로)
# mode 2:
#        A. Segment는 4개의 모델을 모두 추출 대신 최적 모델로 추출: 
#           (1)Green-ADE20K (2)Openness & Complexity-Mapillary Vistas, (3)Facility-수행안함, (4)Sidewalk-ADE20K
#        B. Cityscape는 추론하지 않음 (inference_all.py 파일에 코드 삽입)
#        C. YOLO 추론용 224x224 세그먼트 이미지는 저장 생략, 원본사이즈(_로 시작하는 파일)는 계속 저장

import argparse
import multiprocessing as mp
import os
import torch
import random
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import csv
import time
import cv2
import numpy as np
import tqdm
from glob import glob ## 추가 세그먼트 합치기용
from scipy.spatial import KDTree ##Green Segment의 분포 색체분포 등 구하기 용
from scipy.stats import entropy
import re  # 추가 세그먼트 이름 기준 함칠 때 필요
import shutil   # 파일 복사 기능

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

from predictor import VisualizationDemo

from ultralytics import YOLO

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", help="Task type", default='panoptic')
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

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2],
        default=1,
        help="1: full segmentation from all models, 2: optimized mode with selective models",
    )
    return parser

#세그멘트 합친 후 관련 세그먼트들 정보 추출
def process_segment(png_dir, path, keywords, segment_name):
    """
    png_dir : PNG 파일들이 들어있는 폴더
    path : 원본 이미지 파일 경로 (basename과 model_name 추출용)
    keywords : 세그먼트 키워드 리스트 (ex: ['_tree', '_grass', ...])
    segment_name : 최종 segment 이름 (ex: 'green', 'sidewalk')
    """
    # 파일 찾기
    matching_files = []
    all_png_files = glob(os.path.join(png_dir, "*.png"))

    for file in all_png_files:
        basename = os.path.basename(file).lower()
        if any(basename.startswith(k) for k in keywords):
            matching_files.append(file)
        else:
            first_token = re.split(r"[-_]", basename)[0]
            if first_token in keywords:
                matching_files.append(file)

    if not matching_files:
        print(f"[!] 병합할 {segment_name} 계열 PNG가 없습니다.")
            # === 여기서 CSV라도 생성 ===
        input_basename = os.path.splitext(os.path.basename(path))[0]
        model_name = os.path.basename(os.path.dirname(png_dir))
        output_dir = os.path.join(os.path.dirname(os.path.dirname(png_dir)), f"seg_{segment_name}")
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"{input_basename}_{model_name}_{segment_name}.png"
        csv_path = os.path.join(output_dir, output_filename.replace(".png", ".csv"))

        # CSV에 0 또는 의미 있는 기본값으로 채움
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(['filename', f'{segment_name}_pixels', 'total_pixels', 'ratio',
                            'spread_ratio', 'mean_Hue', 'mean_Saturation', 'mean_Value_brt',
                            'Hue_entropy', 'Saturation_entropy',
                            'Spatial_entropy', 'Edge_density'])
            writer.writerow([output_filename, 0, 0, "0.0%",
                            0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0,
                            0.0, 0.0])
        print(f"[✓] 세그먼트 없음 → 빈 CSV 생성 완료: {csv_path}")
        return

    # 이미지 병합
    merged_image = None
    for file in matching_files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[2] < 4:
            continue
        alpha = img[:, :, 3]
        mask = alpha > 0
        if merged_image is None:
            merged_image = np.zeros_like(img)
        merged_image[mask] = img[mask]

    # 병합 이미지 저장
    segment_path = os.path.join(png_dir, f"{segment_name}_segment.png")
    cv2.imwrite(segment_path, merged_image)
    print(f"[✓] Saved {segment_name} segment mask → {segment_path}")

    # output 폴더 및 파일명
    input_basename = os.path.splitext(os.path.basename(path))[0]
    model_name = os.path.basename(os.path.dirname(png_dir))
    output_dir = os.path.join(os.path.dirname(os.path.dirname(png_dir)), f"seg_{segment_name}")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{input_basename}_{model_name}_{segment_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    shutil.copyfile(segment_path, output_path)
    print(f"[✓] Copied {segment_name} segment to → {output_path}")

    # CSV 통계
    img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        print(f"[!] 오류: 알파 채널 없는 이미지거나 열기 실패 → {output_path}")
        return

    alpha_channel = img[:, :, 3]
    segment_pixels = int(np.sum(alpha_channel == 255))
    total_pixels = alpha_channel.shape[0] * alpha_channel.shape[1]
    ratio = (segment_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    rgb_img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    hue = hsv_img[:, :, 0][alpha_channel == 255]
    sat = hsv_img[:, :, 1][alpha_channel == 255]
    val_brt = hsv_img[:, :, 2][alpha_channel == 255]

    hue_mean = float(np.mean(hue)) if len(hue) > 0 else 0.0
    sat_mean = float(np.mean(sat)) if len(sat) > 0 else 0.0
    val_brt_mean = float(np.mean(val_brt)) if len(val_brt) > 0 else 0.0

    hue_entropy = entropy(np.histogram(hue, bins=30, range=(0, 179), density=True)[0], base=2) if len(hue) > 0 else 0.0
    sat_entropy = entropy(np.histogram(sat, bins=30, range=(0, 255), density=True)[0], base=2) if len(sat) > 0 else 0.0

    ys, xs = np.where(alpha_channel == 255)
    spatial_entropy = 0.0
    if len(xs) > 0:
        grid_size = 30
        hist2d, _, _ = np.histogram2d(xs, ys, bins=[grid_size, grid_size], density=True)
        spatial_entropy = entropy(hist2d.flatten(), base=2)

    spread_ratio = 0.0
    if len(xs) > 1:
        coords = np.vstack((xs, ys)).T
        # 바운딩 박스 면적
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        # 유효 픽셀 수
        pixel_area = len(xs)
        spread_ratio = pixel_area / bbox_area if bbox_area > 0 else 0.0

    gray_img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    edge_density = np.sum(edges > 0) / (gray_img.shape[0] * gray_img.shape[1])

    csv_path = os.path.join(output_dir, output_filename.replace(".png", ".csv"))
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['filename', f'{segment_name}_pixels', 'total_pixels', 'ratio',
                         'spread_ratio', 'mean_Hue', 'mean_Saturation', 'mean_Value_brt',
                         'Hue_entropy', 'Saturation_entropy',
                         'Spatial_entropy', 'Edge_density'])
        writer.writerow([output_filename, segment_pixels, total_pixels, f"{ratio:.1f}%",
                         f"{spread_ratio:.2f}", f"{hue_mean:.2f}", f"{sat_mean:.2f}", f"{val_brt_mean:.2f}",
                         f"{hue_entropy:.3f}", f"{sat_entropy:.3f}",
                         f"{spatial_entropy:.3f}", f"{edge_density:.4f}"])

    print(f"[✓] Saved {segment_name} stats CSV → {csv_path}")

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    save_yolo_segment = args.mode != 2
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    
    demo = VisualizationDemo(cfg)

    if args.input:
        executed_seg_all_set = set()  
        material_model = YOLO('models/best.pt')

        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = cv2.imread(path)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Calculate the aspect ratio
            height, width, _ = img.shape
            aspect_ratio = width / height

            # Resize the image while maintaining the aspect ratio
            if height < width:
                new_height = 640
                new_width = int(aspect_ratio * new_height)
            else:
                new_width = 640
                new_height = int(new_width / aspect_ratio)

            img = cv2.resize(img, (new_width, new_height))
            img_array = np.asarray(img)

            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img_array, args.task)
            print("----------------------------------------------") 
            # print(predictions, visualized_output)
            # logger.info(
            #     "{}: {} in {:.2f}s".format(
            #         path,
            #         "detected {} instances".format(len(predictions["instances"]))
            #         if "instances" in predictions
            #         else "finished",
            #         time.time() - start_time,
            #     )
            # )

            if args.output:
                if len(args.input) == 1:
                    for k in visualized_output.keys():
                        if args.task in k:
                            out_filename = os.path.join(args.output)
                            os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                            visualized_output[k].save(out_filename)

                            panoptic_seg, segments_info = predictions['panoptic_seg']

                            panoptic_seg = panoptic_seg.cpu()

                            materials = []

                            if isinstance(panoptic_seg, torch.Tensor):
                                panoptic_seg = panoptic_seg.numpy()

                            labels, areas = np.unique(panoptic_seg, return_counts=True)

                            count = 0
                            
                            png_dir = os.path.join(os.path.dirname(out_filename), f"{os.path.basename(out_filename).replace('.jpg', '')}_png") ## Added by Dudaji
                            os.makedirs(png_dir, exist_ok=True)  ## Added by Dudaji, Saving png of segment

                            for label in filter(lambda l: l < len(demo.metadata.stuff_classes), labels):
                                if label < 1:
                                    continue

                                binary_mask = (panoptic_seg == label).astype(np.uint8)
                                result = cv2.bitwise_and(img, img, mask=binary_mask)
                                
                                result[np.where(binary_mask == 0)] = [0, 0, 0]
                                result_raw = cv2.cvtColor(result.copy(), cv2.COLOR_BGR2RGBA) ##0408 추가 0528 수정정

                                x, y, w, h = cv2.boundingRect(binary_mask) ## image crop based on bounding box
                                result = result[y:y+h, x:x+w]
                                
                                result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA) ## Revised By Dudaji BGR -> BGRA
                                
                                alpha_channel = np.where(binary_mask[y:y+h, x:x+w] > 0, 255, 0).astype(np.uint8)
                                result[:, :, 3] = alpha_channel  ##ADDED by Dudaji Creating Transparent Background
                                
                                alpha_channel_raw = np.where(binary_mask > 0, 255, 0).astype(np.uint8) ##0408 추가
                                result_raw[:, :, 3] = alpha_channel_raw ##0408 추가

                                ### 이거는 기존 코드 by Dudaji 224 X 224로 png 출력
                                result = cv2.resize(result, (224, 224))
                                material = material_model.predict(result[:, :, :3])[0]
                                material_name = material.names[material.probs.top1]
                                materials.append(material_name)

                                category_id = segments_info[label-1]['category_id']
                                png_filename = f"{demo.metadata.stuff_classes[category_id]}_{material_name}.png" ## Added by Dudaji PNG saving
                                png_filepath = os.path.join(png_dir, png_filename) ## Added by Dudaji PNG saving

                                #### 이거는 기존 코드 by Dudaji아래 저장되는 result는 224x224로 함
                                if save_yolo_segment:
                                    cv2.imwrite(png_filepath, result)

                                #  저장은 원래 처음 input 사이즈로 다시 확대 result 그대로!  ### 250402 수정 
                                #result_resized = cv2.resize(result, (img.shape[1], img.shape[0]))
                                #cv2.imwrite(png_filepath, result_resized)

                                # 5. result_raw 저장 (크롭 없이 전체)
                                raw_filename = f"_{demo.metadata.stuff_classes[category_id]}.png"
                                raw_filepath = os.path.join(png_dir, raw_filename)
                                cv2.imwrite(raw_filepath, result_raw)

                            # (1) Green 처리
                            green_keywords = ['_tree', '_grass', '_mountain','_vegetation','_terrain','_field','_plant']
                            if args.mode == 1 or (args.mode == 2 and 'ade20k' in png_dir.lower()):
                                process_segment(png_dir, path, green_keywords, "green")

                            # (2) Sidewalk 처리
                            sidewalk_keywords = ['_sidewalk', '_earth', '_pavement','_pedestrian','_bike','_dirt',]
                            if args.mode == 1 or (args.mode == 2 and 'ade20k' in png_dir.lower()):
                                process_segment(png_dir, path, sidewalk_keywords, "sidewalk")

                            ## (3) Openness 처리 (하늘 + 도로)
                            openness_keywords = ['_sky','_road','_sidewalk', '_earth', '_pavement','_pedestrian','_bike','_dirt','_lane','_curb']
                            if args.mode == 1 or (args.mode == 2 and 'mapillary' in png_dir.lower()):
                                process_segment(png_dir, path, openness_keywords, "openness")

                            ## (4) sky 처리 (하늘)
                            #sky_keywords = ['_sky']
                            #process_segment(png_dir, path, sky_keywords, "sky")

                            ## (5) facilities 처리 (시설물)
                            #facilities_keywords = ['_billboard','_building','_signboard','_traffic','_pole','_house','_street','_traffic','_wall','_booth','_box','_awning','_pot','_fence','_column','_bridge','_stairs','_curtain','_cardboard','_bench','_banner','_utility','_junction']
                            #process_segment(png_dir, path, facilities_keywords, "facility")

                            ## (6) complexity = openness 반전 + 원본이미지 색 유지

                            if args.mode == 1 or (args.mode == 2 and 'mapillary' in png_dir.lower()):
                                openness_path = os.path.join(png_dir, "openness_segment.png")
                                complexity_path = os.path.join(png_dir, "complexity_segment.png")

                                # 1. openness 이미지 열기
                                openness_img = cv2.imread(openness_path, cv2.IMREAD_UNCHANGED)
                                if openness_img is None or openness_img.shape[2] < 4:
                                    print("[!] openness_segment.png 파일 열기 실패 또는 알파 채널 없음.")
                                    input_basename = os.path.splitext(os.path.basename(path))[0]
                                    model_name = os.path.basename(os.path.dirname(png_dir))
                                    output_dir = os.path.join(os.path.dirname(os.path.dirname(png_dir)), "seg_complexity")
                                    os.makedirs(output_dir, exist_ok=True)

                                    output_filename = f"{input_basename}_{model_name}_complexity.png"
                                    csv_path = os.path.join(output_dir, output_filename.replace(".png", ".csv"))

                                    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                                        writer = csv.writer(f, delimiter='|')
                                        writer.writerow(['filename', 'complexity_pixels', 'total_pixels', 'ratio',
                                                        'spread_ratio', 'mean_Hue', 'mean_Saturation', 'mean_Value_brt',
                                                        'Hue_entropy', 'Saturation_entropy',
                                                        'Spatial_entropy', 'Edge_density'])
                                        writer.writerow([output_filename, 0, 0, "0.0%",
                                                        0.0, 0.0, 0.0, 0.0,
                                                        0.0, 0.0,
                                                        0.0, 0.0])
                                    print(f"[✓] complexity 없음 → 빈 CSV 생성 완료: {csv_path}")                              
                                else:
                                    alpha = openness_img[:, :, 3]

                                    # 2. 반전 마스크 (openness가 255인 곳은 complexity에서는 0)
                                    inverted_alpha = (alpha == 0).astype(np.uint8) * 255

                                    # 3. 원본 이미지 불러오기 (RGB 사용)
                                    original_img = cv2.imread(path, cv2.IMREAD_COLOR)
                                    if original_img is None:
                                        print(f"[!] 원본 이미지 열기 실패: {path}")
                                    else:
                                        # 4. 크기 일치 여부 확인
                                        if original_img.shape[:2] != inverted_alpha.shape:
                                            print("[!] 원본과 openness 크기 불일치 → 원본 resize 진행")
                                            original_img = cv2.resize(original_img, (inverted_alpha.shape[1], inverted_alpha.shape[0]))

                                        # 5. complexity 이미지 생성: 원본 RGB + inverted_alpha
                                        complexity_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)
                                        complexity_img[:, :, 3] = inverted_alpha

                                        # 6. 파일 저장
                                        cv2.imwrite(complexity_path, complexity_img)
                                        print(f"[✓] complexity_segment.png 생성 완료 → {complexity_path}")

                                        # 7. 복사 및 CSV 처리
                                        input_basename = os.path.splitext(os.path.basename(path))[0]
                                        model_name = os.path.basename(os.path.dirname(png_dir))
                                        output_dir = os.path.join(os.path.dirname(os.path.dirname(png_dir)), "seg_complexity")
                                        os.makedirs(output_dir, exist_ok=True)

                                        output_filename = f"{input_basename}_{model_name}_complexity.png"
                                        output_path = os.path.join(output_dir, output_filename)
                                        shutil.copyfile(complexity_path, output_path)
                                        print(f"[✓] Copied complexity segment to → {output_path}")

                                        ### CSV 통계 생성 (기존 process_segment와 동일)
                                        img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
                                        alpha_channel = img[:, :, 3]
                                        complexity_pixels = int(np.sum(alpha_channel == 255))
                                        total_pixels = alpha_channel.shape[0] * alpha_channel.shape[1]
                                        ratio = (complexity_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

                                        rgb_img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
                                        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
                                        hue = hsv_img[:, :, 0][alpha_channel == 255]
                                        sat = hsv_img[:, :, 1][alpha_channel == 255]
                                        val_brt = hsv_img[:, :, 2][alpha_channel == 255]

                                        hue_mean = float(np.mean(hue)) if len(hue) > 0 else 0.0
                                        sat_mean = float(np.mean(sat)) if len(sat) > 0 else 0.0
                                        val_brt_mean = float(np.mean(val_brt)) if len(val_brt) > 0 else 0.0

                                        hue_entropy = entropy(np.histogram(hue, bins=30, range=(0, 179), density=True)[0], base=2) if len(hue) > 0 else 0.0
                                        sat_entropy = entropy(np.histogram(sat, bins=30, range=(0, 255), density=True)[0], base=2) if len(sat) > 0 else 0.0

                                        ys, xs = np.where(alpha_channel == 255)
                                        if len(xs) > 0:
                                            grid_size = 30
                                            hist2d, _, _ = np.histogram2d(xs, ys, bins=[grid_size, grid_size], density=True)
                                            spatial_entropy = entropy(hist2d.flatten(), base=2)
                                        else:
                                            spatial_entropy = 0.0

                                        spread_ratio = 0.0
                                        if len(xs) > 1:
                                            coords = np.vstack((xs, ys)).T
                                            # 바운딩 박스 면적
                                            min_x, max_x = xs.min(), xs.max()
                                            min_y, max_y = ys.min(), ys.max()
                                            bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                                            # 유효 픽셀 수
                                            pixel_area = len(xs)
                                            spread_ratio = pixel_area / bbox_area if bbox_area > 0 else 0.0

                                        gray_img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
                                        edges = cv2.Canny(gray_img, 100, 200)
                                        edge_density = np.sum(edges > 0) / (gray_img.shape[0] * gray_img.shape[1])

                                        csv_path = os.path.join(output_dir, output_filename.replace(".png", ".csv"))
                                        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                                            writer = csv.writer(f, delimiter='|')
                                            writer.writerow(['filename', 'complexity_pixels', 'total_pixels', 'ratio',
                                                            'spread_ratio', 'mean_Hue', 'mean_Saturation', 'mean_Value_brt',
                                                            'Hue_entropy', 'Saturation_entropy',
                                                            'Spatial_entropy', 'Edge_density'])
                                            writer.writerow([output_filename, complexity_pixels, total_pixels, f"{ratio:.1f}%",
                                                            f"{spread_ratio:.2f}", f"{hue_mean:.2f}", f"{sat_mean:.2f}", f"{val_brt_mean:.2f}",
                                                            f"{hue_entropy:.3f}", f"{sat_entropy:.3f}",
                                                            f"{spatial_entropy:.3f}", f"{edge_density:.4f}"])

                                        print(f"[✓] Saved complexity stats CSV → {csv_path}")
                         
                            ## (7) seg_all 처리 → 원본 이미지 전체 분석: 4모델 결과 동일하니 ade20k에서만 한번수행
                            if 'ade20k' in png_dir.lower():
                                input_basename = os.path.splitext(os.path.basename(path))[0]
                                model_name = os.path.basename(os.path.dirname(png_dir))
                                output_dir = os.path.join(os.path.dirname(os.path.dirname(png_dir)), "seg_all")
                                os.makedirs(output_dir, exist_ok=True)

                                output_filename = f"{input_basename}_all.png"
                                output_path = os.path.join(output_dir, output_filename)

                                shutil.copyfile(path, output_path)
                                print(f"[✓] Copied original image to → {output_path}")

                                img = cv2.imread(output_path, cv2.IMREAD_COLOR)
                                if img is None:
                                    print(f"[!] 원본 이미지 열기 실패: {output_path}")
                                else:
                                    total_pixels = img.shape[0] * img.shape[1]
                                    segment_pixels = total_pixels

                                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

                                    hue = hsv_img[:, :, 0].flatten()
                                    sat = hsv_img[:, :, 1].flatten()
                                    val_brt = hsv_img[:, :, 2].flatten()

                                    hue_mean = float(np.mean(hue))
                                    sat_mean = float(np.mean(sat))
                                    val_brt_mean = float(np.mean(val_brt))

                                    hue_entropy = entropy(np.histogram(hue, bins=30, range=(0, 179), density=True)[0], base=2)
                                    sat_entropy = entropy(np.histogram(sat, bins=30, range=(0, 255), density=True)[0], base=2)

                                    ys, xs = np.indices((img.shape[0], img.shape[1]))
                                    grid_size = 30
                                    hist2d, _, _ = np.histogram2d(xs.flatten(), ys.flatten(), bins=[grid_size, grid_size], density=True)
                                    spatial_entropy = entropy(hist2d.flatten(), base=2)

                                    spread_ratio = 1.0

                                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                    edges = cv2.Canny(gray_img, 100, 200)
                                    edge_density = np.sum(edges > 0) / total_pixels

                                    csv_path = os.path.join(output_dir, output_filename.replace(".png", ".csv"))
                                    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                                        writer = csv.writer(f, delimiter='|')
                                        writer.writerow(['filename', 'all_pixels', 'total_pixels', 'ratio',
                                                        'spread_ratio', 'mean_Hue', 'mean_Saturation', 'mean_Value_brt',
                                                        'Hue_entropy', 'Saturation_entropy',
                                                        'Spatial_entropy', 'Edge_density'])
                                        writer.writerow([output_filename, segment_pixels, total_pixels, "100.0%",
                                                        f"{spread_ratio:.2f}", f"{hue_mean:.2f}", f"{sat_mean:.2f}", f"{val_brt_mean:.2f}",
                                                        f"{hue_entropy:.3f}", f"{sat_entropy:.3f}",
                                                        f"{spatial_entropy:.3f}", f"{edge_density:.4f}"])

                                    print(f"[✓] Saved seg_all stats CSV → {csv_path}")



                            ## 여기서 부터 다시 전체 파일 처리
                            stuffs = []

                            rows = []

                            for id in range(len(segments_info)):
                                category_id = segments_info[id]['category_id']
                                class_name = demo.metadata.stuff_classes[category_id]
                                stuffs.append(class_name)

                            #for i, stuff in enumerate(stuffs):
                            #    try:
                            #        rows.append([stuff, materials[i]])
                            #    except:
                            #        rows.append([stuff, 'None'])

                            #### # New Code for Pixel Area and Pixel %  #####
                            total_pixels = panoptic_seg.size  # 전체 이미지 픽셀 수

                            for i, stuff in enumerate(stuffs):
                                try:
                                    material_name = materials[i]
                                except:
                                    material_name = 'None'

                                try:
                                    segment_id = segments_info[i]['id']
                                    # panoptic_seg에서 해당 segment ID의 마스크 생성
                                    binary_mask = (panoptic_seg == segment_id)
                                    area = int(binary_mask.sum())  # 해당 세그먼트의 픽셀 수
                                except:
                                    area = 0

                                ratio = (area / total_pixels * 100) if total_pixels > 0 else 0.0

                                # 형식: stuff|material|면적(px)|비율(%)
                                rows.append([f"{stuff}|{material_name}|{area}|{ratio:.1f}%"])
                            ### 여기 까지 Pixel 너비와 면적 추출하는 코드 수정내용

                            with open(out_filename.replace('jpg', 'csv'), 'w', encoding='utf-8') as f:
                                writer = csv.writer(f, delimiter='|')
                                writer.writerows(rows)
                                
                else:
                    for k in visualized_output.keys():
                        opath = os.path.join(args.output, k)    
                        os.makedirs(opath, exist_ok=True)
                        out_filename = os.path.join(opath, os.path.basename(path))
                        visualized_output[k].save(out_filename)    

            else:
                raise ValueError("Please specify an output path!")
    else:
        raise ValueError("No Input Given")


