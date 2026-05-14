# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

## Lightweight Version: Modified according to --mode {1, 2}
# mode 1: Full execution (original behavior)
# mode 2:
#        A. Segments are extracted using optimal models instead of all 4 models: 
#           (1) Green: ADE20K, (2) Openness & Complexity: Mapillary Vistas, (3) Facility: Skipped, (4) Sidewalk: ADE20K
#        B. Cityscapes inference is skipped (code logic inserted in inference_all.py)
#        C. 224x224 segment images for YOLO inference are not saved; original size files (starting with '_') are still saved.

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
from glob import glob  # For merging additional segments
from scipy.spatial import KDTree  # For calculating color distribution/spread of Green segments
from scipy.stats import entropy
import re  # Required for merging based on segment name patterns
import shutil  # File copy functionality

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

# Extract segment information after merging
def process_segment(png_dir, path, keywords, segment_name):
    """
    png_dir : Directory containing PNG files
    path : Original image path (used to extract basename and model_name)
    keywords : List of segment keywords (ex: ['_tree', '_grass', ...])
    segment_name : Final segment name (ex: 'green', 'sidewalk')
    """
    # Find matching files
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
        print(f"[!] No PNG files found for the {segment_name} category to merge.")
        # === Generate empty CSV even if no segments found ===
        input_basename = os.path.splitext(os.path.basename(path))[0]
        model_name = os.path.basename(os.path.dirname(png_dir))
        output_dir = os.path.join(os.path.dirname(os.path.dirname(png_dir)), f"seg_{segment_name}")
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"{input_basename}_{model_name}_{segment_name}.png"
        csv_path = os.path.join(output_dir, output_filename.replace(".png", ".csv"))

        # Fill CSV with zeros or default values
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
        print(f"[✓] No segment found -> Empty CSV created: {csv_path}")
        return

    # Image Merging
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

    # Save Merged Image
    segment_path = os.path.join(png_dir, f"{segment_name}_segment.png")
    cv2.imwrite(segment_path, merged_image)
    print(f"[✓] Saved {segment_name} segment mask -> {segment_path}")

    # Output directory and filename
    input_basename = os.path.splitext(os.path.basename(path))[0]
    model_name = os.path.basename(os.path.dirname(png_dir))
    output_dir = os.path.join(os.path.dirname(os.path.dirname(png_dir)), f"seg_{segment_name}")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{input_basename}_{model_name}_{segment_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    shutil.copyfile(segment_path, output_path)
    print(f"[✓] Copied {segment_name} segment to -> {output_path}")

    # CSV Statistics
    img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] < 4:
        print(f"[!] Error: Image missing alpha channel or failed to open -> {output_path}")
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
        # Bounding box area
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
        # Valid pixel count
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

    print(f"[✓] Saved {segment_name} stats CSV -> {csv_path}")

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
                                result_raw = cv2.cvtColor(result.copy(), cv2.COLOR_BGR2RGBA) ## Added 0408, Modified 0528

                                x, y, w, h = cv2.boundingRect(binary_mask) ## image crop based on bounding box
                                result = result[y:y+h, x:x+w]
                                
                                result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA) ## Revised By Dudaji BGR -> BGRA
                                
                                alpha_channel = np.where(binary_mask[y:y+h, x:x+w] > 0, 255, 0).astype(np.uint8)
                                result[:, :, 3] = alpha_channel  ## ADDED by Dudaji Creating Transparent Background
                                
                                alpha_channel_raw = np.where(binary_mask > 0, 255, 0).astype(np.uint8) ## Added 0408
                                result_raw[:, :, 3] = alpha_channel_raw ## Added 0408

                                ### Existing code by Dudaji: Output PNG as 224x224
                                result = cv2.resize(result, (224, 224))
                                material = material_model.predict(result[:, :, :3])[0]
                                material_name = material.names[material.probs.top1]
                                materials.append(material_name)

                                category_id = segments_info[label-1]['category_id']
                                png_filename = f"{demo.metadata.stuff_classes[category_id]}_{material_name}.png" ## Added by Dudaji PNG saving
                                png_filepath = os.path.join(png_dir, png_filename) ## Added by Dudaji PNG saving

                                #### Existing code by Dudaji: save result as 224x224
                                if save_yolo_segment:
                                    cv2.imwrite(png_filepath, result)

                                # 5. Save result_raw (Full size, no crop)
                                raw_filename = f"_{demo.metadata.stuff_classes[category_id]}.png"
                                raw_filepath = os.path.join(png_dir, raw_filename)
                                cv2.imwrite(raw_filepath, result_raw)

                            # (1) Green Processing
                            green_keywords = ['_tree', '_grass', '_mountain','_vegetation','_terrain','_field','_plant']
                            if args.mode == 1 or (args.mode == 2 and 'ade20k' in png_dir.lower()):
                                process_segment(png_dir, path, green_keywords, "green")

                            # (2) Sidewalk Processing
                            sidewalk_keywords = ['_sidewalk', '_earth', '_pavement','_pedestrian','_bike','_dirt',]
                            if args.mode == 1 or (args.mode == 2 and 'ade20k' in png_dir.lower()):
                                process_segment(png_dir, path, sidewalk_keywords, "sidewalk")

                            ## (3) Openness Processing (Sky + Road)
                            openness_keywords = ['_sky','_road','_sidewalk', '_earth', '_pavement','_pedestrian','_bike','_dirt','_lane','_curb']
                            if args.mode == 1 or (args.mode == 2 and 'mapillary' in png_dir.lower()):
                                process_segment(png_dir, path, openness_keywords, "openness")

                            ## (4) Sky Processing
                            #sky_keywords = ['_sky']
                            #process_segment(png_dir, path, sky_keywords, "sky")

                            ## (5) Facilities Processing
                            #facilities_keywords = ['_billboard','_building','_signboard','_traffic','_pole','_house','_street','_traffic','_wall','_booth','_box','_awning','_pot','_fence','_column','_bridge','_stairs','_curtain','_cardboard','_bench','_banner','_utility','_junction']
                            #process_segment(png_dir, path, facilities_keywords, "facility")

                            ## (6) Complexity = Inverted Openness + Maintain Original Colors
                            if args.mode == 1 or (args.mode == 2 and 'mapillary' in png_dir.lower()):
                                openness_path = os.path.join(png_dir, "openness_segment.png")
                                complexity_path = os.path.join(png_dir, "complexity_segment.png")

                                # 1. Open openness image
                                openness_img = cv2.imread(openness_path, cv2.IMREAD_UNCHANGED)
                                if openness_img is None or openness_img.shape[2] < 4:
                                    print("[!] Failed to open openness_segment.png or alpha channel missing.")
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
                                    print(f"[✓] Complexity not found -> Empty CSV created: {csv_path}")                      
                                else:
                                    alpha = openness_img[:, :, 3]

                                    # 2. Inversion mask (complexity is 255 where openness is 0)
                                    inverted_alpha = (alpha == 0).astype(np.uint8) * 255

                                    # 3. Load original image (RGB)
                                    original_img = cv2.imread(path, cv2.IMREAD_COLOR)
                                    if original_img is None:
                                        print(f"[!] Failed to open original image: {path}")
                                    else:
                                        # 4. Check size consistency
                                        if original_img.shape[:2] != inverted_alpha.shape:
                                            print("[!] Size mismatch between original and openness -> Resizing original image")
                                            original_img = cv2.resize(original_img, (inverted_alpha.shape[1], inverted_alpha.shape[0]))

                                        # 5. Create complexity image: Original BGR + inverted_alpha
                                        complexity_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)
                                        complexity_img[:, :, 3] = inverted_alpha

                                        # 6. Save file
                                        cv2.imwrite(complexity_path, complexity_img)
                                        print(f"[✓] complexity_segment.png generated -> {complexity_path}")

                                        # 7. Copy and handle CSV
                                        input_basename = os.path.splitext(os.path.basename(path))[0]
                                        model_name = os.path.basename(os.path.dirname(png_dir))
                                        output_dir = os.path.join(os.path.dirname(os.path.dirname(png_dir)), "seg_complexity")
                                        os.makedirs(output_dir, exist_ok=True)

                                        output_filename = f"{input_basename}_{model_name}_complexity.png"
                                        output_path = os.path.join(output_dir, output_filename)
                                        shutil.copyfile(complexity_path, output_path)
                                        print(f"[✓] Copied complexity segment to -> {output_path}")

                                        ### Generate CSV stats (same as process_segment)
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
                                            min_x, max_x = xs.min(), xs.max()
                                            min_y, max_y = ys.min(), ys.max()
                                            bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
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

                                        print(f"[✓] Saved complexity stats CSV -> {csv_path}")
                          
                            ## (7) seg_all Processing -> Analyze full original image: 
                            # Results across 4 models are identical, so perform only once for ade20k
                            if 'ade20k' in png_dir.lower():
                                input_basename = os.path.splitext(os.path.basename(path))[0]
                                model_name = os.path.basename(os.path.dirname(png_dir))
                                output_dir = os.path.join(os.path.dirname(os.path.dirname(png_dir)), "seg_all")
                                os.makedirs(output_dir, exist_ok=True)

                                output_filename = f"{input_basename}_all.png"
                                output_path = os.path.join(output_dir, output_filename)

                                shutil.copyfile(path, output_path)
                                print(f"[✓] Copied original image to -> {output_path}")

                                img = cv2.imread(output_path, cv2.IMREAD_COLOR)
                                if img is None:
                                    print(f"[!] Failed to open original image: {output_path}")
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

                                    print(f"[✓] Saved seg_all stats CSV -> {csv_path}")

                            ## Resume full file processing from here
                            stuffs = []
                            rows = []

                            for id in range(len(segments_info)):
                                category_id = segments_info[id]['category_id']
                                class_name = demo.metadata.stuff_classes[category_id]
                                stuffs.append(class_name)

                            #### # New Code for Pixel Area and Pixel %  #####
                            total_pixels = panoptic_seg.size  # Total image pixel count

                            for i, stuff in enumerate(stuffs):
                                try:
                                    material_name = materials[i]
                                except:
                                    material_name = 'None'

                                try:
                                    segment_id = segments_info[i]['id']
                                    # Create mask for current segment ID in panoptic_seg
                                    binary_mask = (panoptic_seg == segment_id)
                                    area = int(binary_mask.sum())  # Pixel count of current segment
                                except:
                                    area = 0

                                ratio = (area / total_pixels * 100) if total_pixels > 0 else 0.0

                                # Format: stuff|material|area(px)|ratio(%)
                                rows.append([f"{stuff}|{material_name}|{area}|{ratio:.1f}%"])
                            ### End of modified code for extracting pixel width and area

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

