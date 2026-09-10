"""
Evaluation script for Mask R-CNN (Detectron2), MASK-ONLY, reported in an
Ultralytics-YOLO-style summary but computed so the numbers are comparable
to YOLO's own segmentation metrics.

Usage:
    python evaluate_test.py \
        --data-dir /path/to/dataset \
        --test-ann test/_annotations_fixed.json \
        --test-img-dir test \
        --weights ./output_detectron2/model_best.pth \
        --num-classes 1 \
        --conf-thresh 0.25 \
        --iou-thresh 0.5
"""

import argparse
import os

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


# --------------------------------------------------------------------------- #
# Matching helpers (real mask-IoU-based TP/FP/FN, not count-based)
# --------------------------------------------------------------------------- #

def greedy_match_flags(iou_matrix, pred_classes, gt_classes, iou_thresh):
    """
    Greedy one-to-one matching between predictions (rows, already sorted by
    descending confidence) and ground truth (columns), matching only within
    the same class and requiring IoU >= iou_thresh. Each GT can be matched
    at most once.
    """
    num_pred, num_gt = iou_matrix.shape
    gt_matched = np.zeros(num_gt, dtype=bool)
    is_tp = np.zeros(num_pred, dtype=bool)

    for i in range(num_pred):
        best_iou = iou_thresh
        best_j = -1
        for j in range(num_gt):
            if gt_matched[j]:
                continue
            if pred_classes[i] != gt_classes[j]:
                continue
            iou = iou_matrix[i, j]
            if iou >= best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            gt_matched[best_j] = True
            is_tp[i] = True

    return is_tp


def smooth(y, f=0.1):
    """
    Box-filter smoothing over a 1D curve, mirroring Ultralytics' logic.
    Uses edge padding to avoid zero-dropoff artifacts at the boundaries,
    while maintaining the exact original length (valid convolution).
    """
    nf = round(len(y) * f * 2) // 2 + 1  # filter width, forced odd
    if nf <= 1 or len(y) < nf:
        return y
    pad = np.ones(nf // 2)
    y_padded = np.concatenate((pad * y[0], y, pad * y[-1]))
    return np.convolve(y_padded, np.ones(nf) / nf, mode="valid")


def polygons_or_rle_to_rle(segm, height, width):
    """Convert a COCO 'segmentation' field (polygon list or RLE dict) to RLE."""
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm["counts"], list):
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        rle = segm
    return rle


def compute_mask_pr_metrics(predictor, dataset_name, conf_thresh=0.25, iou_thresh=0.5):
    """
    Computes global mask Precision/Recall two ways:
      - "fixed_*"   : P/R using predictions with score >= conf_thresh.
      - "best_f1_*" : P/R at confidence maximizing F1 across full curve
                      (comparable to YOLO's val Precision/Recall).
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)

    all_scores = []
    all_tp = []
    total_gt = 0

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        height, width = img.shape[:2]

        # Vyzvednutí Ground Truth (GT)
        gt_anns = [a for a in d.get("annotations", []) if a.get("iscrowd", 0) == 0 and "segmentation" in a]
        if len(gt_anns) > 0:
            gt_classes = np.array([a["category_id"] for a in gt_anns], dtype=np.int64)
            gt_rles = [polygons_or_rle_to_rle(a["segmentation"], height, width) for a in gt_anns]
            total_gt += len(gt_anns)
        else:
            gt_classes = np.array([], dtype=np.int64)
            gt_rles = []

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        order = torch.argsort(instances.scores, descending=True)
        instances = instances[order]
        num_pred = len(instances)

        if num_pred > 0 and instances.has("pred_masks"):
            pred_classes = instances.pred_classes.numpy()
            pred_scores = instances.scores.numpy()
            pred_masks_raw = instances.pred_masks.numpy()
            pred_masks = (pred_masks_raw >= 0.5).astype(np.uint8)
            pred_rles = [maskUtils.encode(np.asfortranarray(m)) for m in pred_masks]

            if len(gt_rles) > 0:
                iou_mat = np.asarray(maskUtils.iou(pred_rles, gt_rles, [0] * len(gt_rles)))
                is_tp = greedy_match_flags(iou_mat, pred_classes, gt_classes, iou_thresh)
            else:
                # Obrázky bez GT: Všechny detekce modelu jsou False Positives
                is_tp = np.zeros(num_pred, dtype=bool)

            all_scores.append(pred_scores)
            all_tp.append(is_tp)

    if not all_scores or total_gt == 0:
        return {
            "fixed_conf": conf_thresh, "fixed_precision": 0.0, "fixed_recall": 0.0,
            "best_f1_conf": 0.0, "best_f1_precision": 0.0, "best_f1_recall": 0.0, "best_f1": 0.0,
        }

    scores = np.concatenate(all_scores)
    tp_flags = np.concatenate(all_tp)

    order = np.argsort(-scores)
    scores = scores[order]
    tp_flags = tp_flags[order]

    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(~tp_flags)

    precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
    recall_curve = tp_cum / total_gt
    f1_curve = 2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-16)

    # --- Fixed operating point: score >= conf_thresh ---
    keep = scores >= conf_thresh
    if keep.any():
        idx_fixed = np.nonzero(keep)[0][-1]
        fixed_precision = float(precision_curve[idx_fixed])
        fixed_recall = float(recall_curve[idx_fixed])
    else:
        fixed_precision, fixed_recall = 0.0, 0.0

    # --- Best-F1 operating point, mirroring Ultralytics ---
    idx_best = int(smooth(f1_curve, 0.1).argmax())
    best_f1_conf = float(scores[idx_best])
    best_f1_precision = float(precision_curve[idx_best])
    best_f1_recall = float(recall_curve[idx_best])
    best_f1 = float(f1_curve[idx_best])

    return {
        "fixed_conf": conf_thresh,
        "fixed_precision": fixed_precision,
        "fixed_recall": fixed_recall,
        "best_f1_conf": best_f1_conf,
        "best_f1_precision": best_f1_precision,
        "best_f1_recall": best_f1_recall,
        "best_f1": best_f1,
    }


# --------------------------------------------------------------------------- #
# Pure GPU inference benchmark
# --------------------------------------------------------------------------- #

def benchmark_gpu_inference(predictor, dataset_name, num_images=100, warmup_iters=10):
    """
    Measures pure GPU execution time of the model forward pass.
    Performs H2D transfers iteratively, avoiding full dataset VRAM allocation.
    Timing events are recorded asynchronously to prevent pipeline stalls.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available. GPU benchmarking skipped.")
        return 0.0, 0.0

    print("\n--- Running Pure GPU Inference Benchmark ---")

    dataset_dicts = DatasetCatalog.get(dataset_name)[:num_images]
    model = predictor.model
    model.eval()

    prepared_cpu_inputs = []
    with torch.no_grad():
        for d in dataset_dicts:
            original_image = cv2.imread(d["file_name"])
            if original_image is None:
                continue
            if predictor.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            prepared_cpu_inputs.append({"image_tensor": image_tensor, "height": height, "width": width})

    if not prepared_cpu_inputs:
        print("Warning: no valid images available for benchmarking.")
        return 0.0, 0.0

    # Warmup pass
    with torch.no_grad():
        for inp in prepared_cpu_inputs[:warmup_iters]:
            gpu_input = [{"image": inp["image_tensor"].to(device), "height": inp["height"], "width": inp["width"]}]
            _ = model(gpu_input)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(len(prepared_cpu_inputs))]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(len(prepared_cpu_inputs))]

    # Záznam událostí bez synchronizace uvnitř smyčky pro maximální propustnost
    with torch.no_grad():
        for i, inp in enumerate(prepared_cpu_inputs):
            gpu_input = [{"image": inp["image_tensor"].to(device), "height": inp["height"], "width": inp["width"]}]
            start_events[i].record()
            _ = model(gpu_input)
            end_events[i].record()

    # Jednorázová synchronizace na konci
    torch.cuda.synchronize()

    latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    avg_latency_ms = float(np.mean(latencies))
    fps = 1000.0 / avg_latency_ms
    print(f"Pure GPU Latency : {avg_latency_ms:.2f} ms / image  (n={len(latencies)})")
    print(f"Pure GPU Speed   : {fps:.1f} FPS")
    return avg_latency_ms, fps


def main():
    parser = argparse.ArgumentParser(description="Evaluate Detectron2 Mask R-CNN masks, YOLO-comparable format")
    parser.add_argument("--data-dir", required=True, help="Root directory containing images")
    parser.add_argument("--test-ann", required=True, help="Path to test COCO JSON")
    parser.add_argument("--test-img-dir", default="test", help="Subdirectory containing test images")
    parser.add_argument("--weights", required=True, help="Path to trained weights (model_best.pth)")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of custom classes (excluding background)")
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="Fixed confidence threshold for the 'fixed' P/R operating point.")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="Mask IoU threshold for TP matching in Precision/Recall (default: 0.5)")
    parser.add_argument("--map-score-thresh", type=float, default=0.001,
                        help="Score threshold used for full PR-curve integration.")
    parser.add_argument("--benchmark-images", type=int, default=100,
                        help="Number of images to use for the pure-GPU latency benchmark")
    args = parser.parse_args()

    test_ds_name = "custom_dataset_test"
    test_img_path = os.path.join(args.data_dir, args.test_img_dir) if not os.path.isabs(
        args.test_img_dir) else args.test_img_dir
    test_json_path = os.path.join(args.data_dir, args.test_ann) if not os.path.isabs(args.test_ann) else args.test_ann

    # 1. Register Test Dataset
    register_coco_instances(test_ds_name, {}, test_json_path, test_img_path)

    # 2. Configure Model
    cfg = get_cfg()
    config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_path))

    cfg.DATASETS.TEST = (test_ds_name,)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.map_score_thresh

    predictor = DefaultPredictor(cfg)

    # 3. Standard COCO mAP Evaluation (MASKS ONLY)
    print("\n==================================================")
    print(" 1. COCO Mask AP Evaluation (AP50-95 & AP50)")
    print("==================================================")

    evaluator = COCOEvaluator(test_ds_name, tasks=("segm",), output_dir="./test_results")
    test_loader = build_detection_test_loader(cfg, test_ds_name)
    coco_results = inference_on_dataset(predictor.model, test_loader, evaluator)

    segm_ap50_95 = coco_results["segm"]["AP"]
    segm_ap50 = coco_results["segm"]["AP50"]

    # 4. Global Mask Precision / Recall
    print("\n==================================================")
    print(f" 2. Mask Precision / Recall (IoU={args.iou_thresh})")
    print("==================================================")
    pr = compute_mask_pr_metrics(
        predictor, test_ds_name, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh
    )
    print(f"Fixed   @ conf={pr['fixed_conf']:.3f}   : P={pr['fixed_precision']:.4f}  R={pr['fixed_recall']:.4f}")
    print(f"Best-F1 @ conf={pr['best_f1_conf']:.3f}   : P={pr['best_f1_precision']:.4f}  "
          f"R={pr['best_f1_recall']:.4f}  F1={pr['best_f1']:.4f}  <- comparable to YOLO's val P/R")

    # 5. Pure GPU Inference Speed
    gpu_latency_ms, gpu_fps = benchmark_gpu_inference(
        predictor, test_ds_name, num_images=args.benchmark_images
    )

    # 6. Summary Output
    print("\n" + "=" * 78)
    print(" ULTRALYTICS-STYLE MASK SUMMARY (TEST SET)")
    print("=" * 78)
    print(f"{'Task':<20} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    print("-" * 78)
    print(f"{'Mask (Seg) fixed':<20} {pr['fixed_precision']:<12.4f} {pr['fixed_recall']:<12.4f} "
          f"{segm_ap50 / 100.0:<12.4f} {segm_ap50_95 / 100.0:<12.4f}")
    print(f"{'Mask (Seg) best-F1':<20} {pr['best_f1_precision']:<12.4f} {pr['best_f1_recall']:<12.4f} "
          f"{segm_ap50 / 100.0:<12.4f} {segm_ap50_95 / 100.0:<12.4f}   <- compare to YOLO")
    print("-" * 78)
    print(f"Pure GPU Inference Latency : {gpu_latency_ms:.2f} ms / image")
    print(f"Pure GPU Inference Speed   : {gpu_fps:.1f} FPS")
    print(f"Fixed P/R conf threshold   : {pr['fixed_conf']}")
    print(f"Best-F1 P/R conf threshold : {pr['best_f1_conf']:.4f} (found automatically, like YOLO)")
    print(f"P/R Mask IoU Threshold     : {args.iou_thresh}")
    print(f"mAP Eval Score Threshold   : {args.map_score_thresh}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
