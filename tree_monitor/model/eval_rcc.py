"""
Evaluation script for Mask R-CNN (Detectron2), MASK-ONLY, reported in an
Ultralytics-YOLO-style summary but computed so the numbers are comparable
to YOLO's own segmentation metrics.

Note: Mask R-CNN's box head and mask head are two separate branches operating
on the same proposals. This script deliberately ignores the box branch
entirely -- both the AP evaluator and the Precision/Recall computation only
ever look at `pred_masks` vs. GT `segmentation` (pixel-wise IoU), never at
`pred_boxes` vs. GT `bbox`.

Computes (masks only):
  - AP50-95 & AP50, via the standard COCOEvaluator restricted to the "segm"
    task, run with a near-zero score threshold so the PR-curve integration
    is not truncated (this is what Ultralytics does internally too).
  - Global Precision & Recall at a fixed confidence threshold (default 0.25,
    matching YOLO's default val threshold), computed via a real greedy
    mask-IoU-matching TP/FP/FN procedure.
  - Pure GPU inference time (CUDA-event based, isolating the model forward
    pass from CPU-side preprocessing / dataloading / H2D transfer). This
    times the whole model (Mask R-CNN always runs both branches together),
    but is reported here alongside the mask metrics since inference cost
    isn't separable per-branch.

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

def greedy_match(iou_matrix, pred_classes, gt_classes, iou_thresh):
    """
    Greedy one-to-one matching between predictions (rows, already sorted by
    descending confidence) and ground truth (columns), matching only within
    the same class and requiring IoU >= iou_thresh. Each GT can be matched
    at most once. This mirrors the logic COCOEvaluator / Ultralytics use for
    computing TP/FP/FN at a fixed threshold.
    """
    num_pred, num_gt = iou_matrix.shape
    gt_matched = np.zeros(num_gt, dtype=bool)
    tp = 0
    fp = 0

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
            tp += 1
        else:
            fp += 1

    fn = int(num_gt - gt_matched.sum())
    return tp, fp, fn


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


def compute_mask_precision_recall(predictor, dataset_name, conf_thresh=0.25, iou_thresh=0.5):
    """
    Computes global mask Precision/Recall at a fixed confidence threshold,
    using real pixel-wise IoU matching (greedy, class-aware). Bounding boxes
    are not used anywhere in this function.
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)

    tp_mask = fp_mask = fn_mask = 0
    skipped_no_gt_masks = 0

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        if img is None:
            continue
        height, width = img.shape[:2]

        gt_anns = [a for a in d["annotations"] if a.get("iscrowd", 0) == 0]
        gt_anns = [a for a in gt_anns if "segmentation" in a]
        if len(gt_anns) == 0:
            skipped_no_gt_masks += 1
            continue

        gt_classes = np.array([a["category_id"] for a in gt_anns], dtype=np.int64)
        gt_rles = [polygons_or_rle_to_rle(a["segmentation"], height, width) for a in gt_anns]

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        keep = instances.scores >= conf_thresh
        instances = instances[keep]
        # sort remaining predictions by descending score for greedy matching
        order = torch.argsort(instances.scores, descending=True)
        instances = instances[order]

        num_pred = len(instances)
        num_gt = len(gt_anns)

        if num_pred > 0 and instances.has("pred_masks"):
            pred_classes = instances.pred_classes.numpy()
            # `pred_masks` SHOULD already be binary here, since predictor()
            # runs detector_postprocess() -> paste_masks_in_image(threshold=0.5)
            # internally. But that binarization happens implicitly inside
            # detectron2 and depends on the exact call path / config, so we
            # threshold explicitly here too rather than relying on it: a plain
            # `.astype(np.uint8)` on genuinely continuous [0,1] probabilities
            # would truncate toward zero (e.g. 0.8 -> 0) and silently corrupt
            # every mask IoU computed below.
            pred_masks_raw = instances.pred_masks.numpy()
            pred_masks = (pred_masks_raw >= 0.5).astype(np.uint8)
            pred_rles = [maskUtils.encode(np.asfortranarray(m)) for m in pred_masks]
            iou_mat = np.asarray(maskUtils.iou(pred_rles, gt_rles, [0] * len(gt_rles)))
            m_tp, m_fp, m_fn = greedy_match(iou_mat, pred_classes, gt_classes, iou_thresh)
        else:
            m_tp, m_fp, m_fn = 0, 0, num_gt

        tp_mask += m_tp
        fp_mask += m_fp
        fn_mask += m_fn

    if skipped_no_gt_masks:
        print(f"Note: skipped {skipped_no_gt_masks} image(s) with no GT segmentation masks.")

    precision = tp_mask / (tp_mask + fp_mask + 1e-16)
    recall = tp_mask / (tp_mask + fn_mask + 1e-16)
    return precision, recall


# --------------------------------------------------------------------------- #
# Pure GPU inference benchmark
# --------------------------------------------------------------------------- #

def benchmark_gpu_inference(predictor, dataset_name, num_images=100, warmup_iters=10):
    """
    Measures pure GPU execution time of the model forward pass, excluding
    disk I/O, CPU-side resize/normalize preprocessing and any postprocessing
    that happens outside the model. All CPU-side work (image resize with the
    predictor's own augmentation, tensor conversion, host-to-device copy) is
    done up front, outside the timed region, so CUDA events measure model
    compute only.

    Note: Mask R-CNN's box and mask branches share the backbone/RPN/ROI
    pooling and run in a single forward pass, so this latency is for the
    whole model -- it cannot be isolated to "just the mask head" without
    modifying the model internals.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available. GPU benchmarking skipped.")
        return 0.0, 0.0

    print("\n--- Running Pure GPU Inference Benchmark ---")

    dataset_dicts = DatasetCatalog.get(dataset_name)[:num_images]
    model = predictor.model
    model.eval()

    prepared_inputs = []
    with torch.no_grad():
        for d in dataset_dicts:
            original_image = cv2.imread(d["file_name"])
            if original_image is None:
                continue
            if predictor.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image = image.to(device)  # do the H2D copy now, outside the timed loop
            prepared_inputs.append({"image": image, "height": height, "width": width})

    if not prepared_inputs:
        print("Warning: no images available for benchmarking.")
        return 0.0, 0.0

    # Warmup (compiles cuDNN kernels, allocates workspace, etc.)
    with torch.no_grad():
        for inp in prepared_inputs[:warmup_iters]:
            _ = model([inp])
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []

    with torch.no_grad():
        for inp in prepared_inputs:
            start_event.record()
            _ = model([inp])
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

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
                        help="Confidence threshold for Precision/Recall, matching YOLO default (0.25)")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="Mask IoU threshold for TP matching in Precision/Recall (default: 0.5)")
    parser.add_argument("--map-score-thresh", type=float, default=0.001,
                        help="Score threshold used ONLY for the COCO mAP eval pass. Must stay low "
                             "(YOLO/COCO convention) so the PR curve is not truncated; do not set this "
                             "to --conf-thresh.")
    parser.add_argument("--benchmark-images", type=int, default=100,
                        help="Number of images to use for the pure-GPU latency benchmark")
    args = parser.parse_args()

    test_ds_name = "custom_dataset_test"
    test_img_path = os.path.join(args.data_dir, args.test_img_dir) if not os.path.isabs(
        args.test_img_dir) else args.test_img_dir
    test_json_path = os.path.join(args.data_dir, args.test_ann) if not os.path.isabs(args.test_ann) else args.test_ann

    # 1. Register Test Dataset
    register_coco_instances(test_ds_name, {}, test_json_path, test_img_path)

    # 2. Configure Model.
    # IMPORTANT: SCORE_THRESH_TEST is kept low here on purpose. COCO AP is an
    # integral over the full precision/recall curve across all confidence
    # levels; truncating predictions at 0.25 before the evaluator sees them
    # would silently drop low-confidence-but-correct detections and make
    # AP50 / AP50-95 incomparable to YOLO's own AP, which is computed the
    # same low-threshold way internally. The 0.25 cutoff is only applied
    # afterwards, in compute_mask_precision_recall, to match YOLO's
    # *displayed* P/R (which YOLO also reports at conf=0.25 by default).
    cfg = get_cfg()
    config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_path))

    cfg.DATASETS.TEST = (test_ds_name,)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.map_score_thresh

    predictor = DefaultPredictor(cfg)

    # 3. Standard COCO mAP Evaluation, MASKS ONLY (AP50-95 and AP50).
    # tasks=("segm",) tells COCOEvaluator to skip box AP entirely -- it
    # never even computes bbox precision/recall internally.
    print("\n==================================================")
    print(" 1. COCO Mask AP Evaluation (AP50-95 & AP50)")
    print("==================================================")

    evaluator = COCOEvaluator(test_ds_name, tasks=("segm",), output_dir="./test_results")
    test_loader = build_detection_test_loader(cfg, test_ds_name)
    coco_results = inference_on_dataset(predictor.model, test_loader, evaluator)

    segm_ap50_95 = coco_results["segm"]["AP"]
    segm_ap50 = coco_results["segm"]["AP50"]

    # 4. Global mask Precision / Recall at conf = args.conf_thresh, via real
    # pixel-wise mask IoU matching. Boxes play no role here.
    print("\n==================================================")
    print(f" 2. Mask Precision / Recall @ conf={args.conf_thresh}, IoU={args.iou_thresh}")
    print("==================================================")
    precision, recall = compute_mask_precision_recall(
        predictor, test_ds_name, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh
    )

    # 5. Pure GPU Inference Speed (whole model -- see note in the function)
    gpu_latency_ms, gpu_fps = benchmark_gpu_inference(
        predictor, test_ds_name, num_images=args.benchmark_images
    )

    # ==================================================
    # 6. ULTRALYTICS-STYLE SUMMARY TABLE OUTPUT (masks only)
    # ==================================================
    print("\n" + "=" * 65)
    print(" ULTRALYTICS-STYLE MASK SUMMARY (TEST SET)")
    print("=" * 65)
    print(f"{'Task':<12} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    print("-" * 65)
    print(f"{'Mask (Seg)':<12} {precision:<12.4f} {recall:<12.4f} "
          f"{segm_ap50 / 100.0:<12.4f} {segm_ap50_95 / 100.0:<12.4f}")
    print("-" * 65)
    print(f"Pure GPU Inference Latency : {gpu_latency_ms:.2f} ms / image")
    print(f"Pure GPU Inference Speed   : {gpu_fps:.1f} FPS")
    print(f"P/R Confidence Threshold   : {args.conf_thresh}")
    print(f"P/R Mask IoU Threshold     : {args.iou_thresh}")
    print(f"mAP Eval Score Threshold   : {args.map_score_thresh} (kept low; see comment in main())")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
