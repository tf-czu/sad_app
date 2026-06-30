"""
    Detect of columnar apple trees in postprocessing
"""
import os
import cv2
import json

import matplotlib.pyplot as plt

from ast import literal_eval

from sympy.physics.optics import medium

from tree_analyse import TreeAnalyse


class TreeDetection:
    def __init__(self, images_path, model_path, out_dir,
                 margin = 0.01, spacing = 300, min_area = 0.1, verbose = False):
        self.images_path = images_path
        self.tree_analyse = TreeAnalyse((1080, 1920), model_path, margin=margin,
                                        min_tree_spacing = spacing, min_area_limit = min_area, verbose = verbose)
        self.result_dir = os.path.join(self.images_path, "tmp", out_dir)
        os.makedirs(self.result_dir)
        self.annotations = {}
        self.verbose = verbose

    def process_data(self, im_name):
        im = cv2.imread(os.path.join(self.images_path, im_name))
        assert im.shape == (1080, 1920, 3)
        tree_data, debug_img = self.tree_analyse.process(im)
        self.draw(debug_img, im_name)
        return tree_data


    def draw(self, img, im_name):
        cv2.imwrite(os.path.join(self.result_dir, im_name), img)

    def run_detection(self):
        for im_name in sorted(os.listdir(self.images_path)):
            if im_name.lower().endswith((".jpg", ".jpeg")):
                tree_data = self.process_data(im_name)
                if tree_data:
                    canopy_list = []
                    for __, canopy in tree_data:
                        canopy_list.append([ar.tolist() for ar in canopy])  # arr to list due json

                    self.annotations[im_name] = canopy_list

        with open(os.path.join(self.result_dir, 'annotation.json'), 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=4)

        if self.verbose:
            return self.tree_analyse.debug_area_ratio, self.tree_analyse.debug_spacing

def process_debug_data(debug_area_ratio, debug_spacing):
    print("debug_area_ratio")
    print(debug_area_ratio)
    print("debug_spacing")
    print(debug_spacing)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    ax1.hist(debug_area_ratio, bins=30, color='dimgrey', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Area ratio (-)', fontsize=12)
    ax1.set_ylabel('Frequency (-)', fontsize=12)

    distances = [d*0.97 for d in debug_spacing]
    ax2.hist(distances, bins=30, color='dimgrey', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Detection distance (mm)', fontsize=12)
    ax2.set_ylabel('Frequency (-)', fontsize=12)

    # Auto label format (avoid overlap)
    plt.tight_layout()

    plt.savefig('histograms.png', dpi=1200)
    # plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', help='path to image dir')
    parser.add_argument('--models', help='Path to models, or string "both" (use both models)', default="tree_monitor/model/my_models/medium/")
    parser.add_argument('--out_dir', help='Output directory for results', default="results")
    parser.add_argument('--margin', help='Ignored edge distance, float or list', default="0.01")
    parser.add_argument('--spacing', help='Spacing between trees (px), int or list of ints', default="300")
    parser.add_argument('--min-area', help='Required canopy area in bbox, float or list', default="0.1")
    args = parser.parse_args()

    margins = literal_eval(args.margin)
    spacings = literal_eval(args.spacing)
    min_areas = literal_eval(args.min_area)

    if isinstance(margins, list):
        for margin in margins:
            assert isinstance(spacings, int)
            assert isinstance(min_areas, float)
            out_dir = f"results_mar_{margin:.3f}"
            detect = TreeDetection(args.images, args.models, out_dir, margin=margin)
            detect.run_detection()

    elif isinstance(spacings, list):
        for spa in spacings:
            assert isinstance(min_areas, list)
            for area in min_areas:
                out_dir = f"results_mar_{spa:03d}_{area:02f}"
                print(out_dir)
                detect = TreeDetection(args.images, args.models, out_dir, spacing = spa, min_area = area)
                detect.run_detection()
    else:
        if args.models == "both":
            models = [
                "tree_monitor/model/my_models/nano/",
                "tree_monitor/model/my_models/medium/"
            ]
            ratios = []
            spacings = []
            for model, out_dir in zip(models, ["margin_res_nano", "margin_res_medium"]):
                print(f"Process data with model: {model}")
                detect = TreeDetection(args.images, model, out_dir, verbose = True)
                debug_area_ratio, debug_spacing = detect.run_detection()
                ratios.extend(debug_area_ratio)
                spacings.extend(debug_spacing)
            process_debug_data(ratios, spacings)

        else:
            detect = TreeDetection(args.images, args.models, args.out_dir)
            detect.run_detection()
