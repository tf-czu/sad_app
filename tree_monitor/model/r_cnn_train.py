"""
Training script for Mask R-CNN using Detectron2 with Early Stopping.

Dependencies:
    pip install torch torchvision opencv-python tensorboard pycocotools
    python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

Usage:
    python train_detectron2.py \
        --data-dir /path/to/dataset \
        --train-ann annotations/instances_train.json \
        --val-ann annotations/instances_val.json \
        --num-classes 1 \
        --max-iter 7000 \
        --batch-size 4 \
        --output-dir ./output_detectron2
"""

import argparse
import os
import cv2
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode


# --------------------------------------------------------------------------
# Custom Early Stopping Hook
# --------------------------------------------------------------------------
class EarlyStoppingHook(HookBase):
    """
    Hook for Early Stopping based on validation metrics.
    Stops training when the tracked metric stops improving after a given patience.
    """

    def __init__(self, patience=4, metric_name="segm/AP50", min_delta=0.001):
        """
        Args:
            patience (int): Number of evaluations without improvement to wait before stopping.
            metric_name (str): Metric key in trainer.storage to monitor (e.g., 'segm/AP50', 'bbox/AP').
            min_delta (float): Minimum change to qualify as an improvement.
        """
        super().__init__()
        self.patience = patience
        self.metric_name = metric_name
        self.min_delta = min_delta
        self.best_metric = -float("inf")
        self.patience_counter = 0

    def after_step(self):
        # Trigger check only when evaluation has just been executed
        eval_period = self.trainer.cfg.TEST.EVAL_PERIOD
        if eval_period <= 0 or (self.trainer.iter + 1) % eval_period != 0:
            return

        # Read the latest evaluated metric from Storage
        latest_metrics = self.trainer.storage.latest()
        if self.metric_name not in latest_metrics:
            print(f"\n[EarlyStopping] Warning: Metric '{self.metric_name}' not found in storage. Skipping check.")
            return

        current_val, _ = latest_metrics[self.metric_name]

        if current_val > self.best_metric + self.min_delta:
            print(
                f"\n[EarlyStopping] Metric '{self.metric_name}' improved from {self.best_metric:.4f} to {current_val:.4f}. Resetting counter.")
            self.best_metric = current_val
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(
                f"\n[EarlyStopping] No improvement in '{self.metric_name}' ({current_val:.4f} vs best {self.best_metric:.4f}). "
                f"Patience: {self.patience_counter}/{self.patience}")

            if self.patience_counter >= self.patience:
                print(f"\n[EarlyStopping] Early stopping triggered at iteration {self.trainer.iter + 1}!")
                # Force loop termination by setting current iteration to max_iter
                self.trainer.iter = self.trainer.max_iter


# --------------------------------------------------------------------------
# Custom Trainer with COCO Evaluation and Early Stopping Support
# --------------------------------------------------------------------------
class CustomTrainer(DefaultTrainer):
    """
    Extends DefaultTrainer to automatically evaluate (mAP)
    on the validation dataset and apply Early Stopping.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "validation")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        # Build default hooks (checkpointing, LR scheduler, evaluation hook, etc.)
        hooks = super().build_hooks()

        # Add Early Stopping Hook (patience=4 * EVAL_PERIOD=500 -> 2000 steps patience)
        early_stopping_hook = EarlyStoppingHook(
            patience=4,
            metric_name="segm/AP50",
            min_delta=0.001
        )
        hooks.append(early_stopping_hook)
        return hooks


# --------------------------------------------------------------------------
# Main Execution Pipeline
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN in Detectron2 with Early Stopping")
    parser.add_argument("--data-dir", required=True, help="Root directory containing images")
    parser.add_argument("--train-ann", required=True, help="Path to training COCO JSON")
    parser.add_argument("--val-ann", default=None, help="Path to validation COCO JSON")
    parser.add_argument("--num-classes", type=int,default=1,
                        help="Number of CUSTOM classes (EXCLUDING background)")
    parser.add_argument("--max-iter", type=int, default=7000,
                        help="Total number of training iterations (~20-25 epochs for 1200 imgs)")
    parser.add_argument("--batch-size", type=int, default=4, help="Total batch size per iteration")
    parser.add_argument("--lr", type=float, default=0.00025, help="Base learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--output-dir", default="./output_detectron2", help="Output directory for checkpoints and logs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(args.data_dir, p)

    # 1. Register datasets in COCO format
    train_ds_name = "custom_dataset_train"
    train_img_dir = os.path.join(args.data_dir, "train")
    val_img_dir = os.path.join(args.data_dir, "valid")
    register_coco_instances(train_ds_name, {}, resolve(args.train_ann), train_img_dir)

    val_ds_name = None
    if args.val_ann:
        val_ds_name = "custom_dataset_val"
        register_coco_instances(val_ds_name, {}, resolve(args.val_ann), val_img_dir)

    # 2. Configure model and hyperparameters
    cfg = get_cfg()

    # Load default architecture config (ResNet-50-FPN)
    config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_path))

    cfg.DATASETS.TRAIN = (train_ds_name,)
    cfg.DATASETS.TEST = (val_ds_name,) if val_ds_name else ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    # Pretrained weights from COCO
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)

    # Freeze stem and first res-stage to prevent overfitting on small datasets
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    # Optimization parameters tailored for ~1200 images
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (5000, 6500)  # Decay learning rate towards the end
    cfg.SOLVER.GAMMA = 0.1

    # Checkpoint saving and evaluation frequency
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    if val_ds_name:
        cfg.TEST.EVAL_PERIOD = 500  # Run COCO mAP evaluation every 500 steps

    # Model heads (class count EXCLUDES background, Detectron2 handles it internally)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = args.output_dir

    # 3. Start Training
    print("--- Starting Detectron2 Training ---")
    trainer = CustomTrainer(cfg) if val_ds_name else DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 4. Final Evaluation and Sample Inference
    if val_ds_name:
        print("\n--- Running Final Evaluation ---")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for inference

        evaluator = COCOEvaluator(val_ds_name, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, val_ds_name)
        predictor = DefaultPredictor(cfg)

        results = inference_on_dataset(predictor.model, val_loader, evaluator)
        print("Evaluation Results (COCO mAP):", results)

        # Generate preview on a sample validation image
        print("\n--- Generating Inference Sample ---")
        val_metadata = MetadataCatalog.get(val_ds_name)

        # Read the first image found in the data directory
        sample_file = os.listdir(val_img_dir)[0]
        img = cv2.imread(os.path.join(val_img_dir, sample_file))

        outputs = predictor(img)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=val_metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        preview_file = os.path.join(args.output_dir, "prediction_sample.jpg")
        cv2.imwrite(preview_file, out.get_image()[:, :, ::-1])
        print(f"Visualization saved to: {preview_file}")


if __name__ == "__main__":
    main()
