import json

def fix_coco_json(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['categories'] = [
        {"id": 1, "name": "canopy", "supercategory": "none"}
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

fix_coco_json("coco-seg-dataset/train/_annotations.coco.json", "coco-seg-dataset/train/_annotations_fixed.json")
fix_coco_json("coco-seg-dataset/valid/_annotations.coco.json", "coco-seg-dataset/valid/_annotations_fixed.json")
fix_coco_json("coco-seg-dataset/test/_annotations.coco.json", "coco-seg-dataset/test/_annotations_fixed.json")
