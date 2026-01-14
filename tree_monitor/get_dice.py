import os
import json
import csv
import argparse
import numpy as np
import cv2


def det_contour_to_xy(contour):
    # detection point format: [[x,y]] (OpenCV-like) -> (x,y)
    pts = []
    for p in contour:
        if isinstance(p, (list, tuple)) and len(p) == 1 and isinstance(p[0], (list, tuple)) and len(p[0]) == 2:
            x, y = p[0]
        else:
            # allow also [x,y]
            x, y = p
        pts.append((float(x), float(y)))
    return pts


def coco_poly_to_xy(poly_flat):
    # COCO polygon: [x1,y1,x2,y2,...]
    assert len(poly_flat) % 2 == 0
    pts = []
    for i in range(0, len(poly_flat), 2):
        pts.append((float(poly_flat[i]), float(poly_flat[i + 1])))
    return pts


def pts_to_cv(pts, w, h):
    arr = np.round(np.array(pts, dtype=np.float32)).astype(np.int32)
    if arr.size == 0:
        return None
    arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
    return arr.reshape((-1, 1, 2))


def tree_y_from_det(tree_contours):
    ys = []
    for c in tree_contours:
        pts = det_contour_to_xy(c)
        if len(pts) >= 3:
            ys.extend([p[1] for p in pts])
    return float(np.mean(ys)) if ys else float("inf")


def tree_y_from_ann(polys_xy):
    ys = []
    for pts in polys_xy:
        if len(pts) >= 3:
            ys.extend([p[1] for p in pts])
    return float(np.mean(ys)) if ys else float("inf")


def mask_from_det(tree_contours, w, h):
    m = np.zeros((h, w), dtype=np.uint8)
    for c in tree_contours:
        pts = det_contour_to_xy(c)
        if len(pts) < 3:
            continue
        cv_pts = pts_to_cv(pts, w, h)
        if cv_pts is None:
            continue
        cv2.fillPoly(m, [cv_pts], 1)
    return m


def mask_from_ann(polys_xy, w, h):
    m = np.zeros((h, w), dtype=np.uint8)
    for pts in polys_xy:
        if len(pts) < 3:
            continue
        cv_pts = pts_to_cv(pts, w, h)
        if cv_pts is None:
            continue
        cv2.fillPoly(m, [cv_pts], 1)
    return m


def dice(m1, m2):
    a = int(m1.sum())
    b = int(m2.sum())
    if a + b == 0:
        return 0.0
    inter = int((m1 & m2).sum())
    return (2.0 * inter) / (a + b)


def match_by_y(det_ys, ann_ys):
    # returns list of ann_index or None for each det_index
    nd, na = len(det_ys), len(ann_ys)
    if nd == 0:
        return []
    if na == 0:
        return [None] * nd

    det_idx = list(range(nd))
    ann_idx = list(range(na))

    # Typical case: 1-2 trees -> simplest: sort and pair
    if nd == na:
        det_sorted = sorted(det_idx, key=lambda i: det_ys[i])
        ann_sorted = sorted(ann_idx, key=lambda j: ann_ys[j])
        out = [None] * nd
        for i, j in zip(det_sorted, ann_sorted):
            out[i] = j
        return out

    # Greedy unique matching by |y diff|
    pairs = []
    for i in det_idx:
        for j in ann_idx:
            pairs.append((abs(det_ys[i] - ann_ys[j]), i, j))
    pairs.sort(key=lambda t: t[0])

    used_det = set()
    used_ann = set()
    out = [None] * nd
    for _, i, j in pairs:
        if i in used_det or j in used_ann:
            continue
        out[i] = j
        used_det.add(i)
        used_ann.add(j)

    return out


def draw_debug(out_path, w, h, det_tree, ann_polys_xy, title):
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # detection = green
    for c in det_tree:
        pts = det_contour_to_xy(c)
        if len(pts) < 3:
            continue
        cv_pts = pts_to_cv(pts, w, h)
        if cv_pts is None:
            continue
        cv2.polylines(img, [cv_pts], True, (0, 200, 0), 2)

    # annotation = red
    if ann_polys_xy is not None:
        for pts in ann_polys_xy:
            if len(pts) < 3:
                continue
            cv_pts = pts_to_cv(pts, w, h)
            if cv_pts is None:
                continue
            cv2.polylines(img, [cv_pts], True, (0, 0, 255), 2)

    y = 30
    for line in title.split("\n"):
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        y += 30

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out_csv", default="results.csv")
    ap.add_argument("--debug_dir", default="tmp")
    args = ap.parse_args()

    det = json.load(open(args.detections, "r", encoding="utf-8"))
    coco = json.load(open(args.annotations, "r", encoding="utf-8"))

    # COCO lookups
    img_by_name = {im["file_name"]: im for im in coco["images"]}
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    # We expect c1/c2 categories exist
    assert any(v == "c1" for v in cat_name.values())
    assert any(v == "c2" for v in cat_name.values())

    os.makedirs(args.debug_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "image_name",
            "det_index",
            "ann_label",
            "ann_ids",
            "det_y",
            "ann_y",
            "dice",
            "debug_image"
        ])

        for image_name, det_trees in det.items():
            assert isinstance(det_trees, list)

            # Must exist in COCO
            assert image_name in img_by_name, f"Image {image_name} not found in COCO images"
            im = img_by_name[image_name]
            img_id = im["id"]
            W = int(im["width"])
            H = int(im["height"])
            assert W > 0 and H > 0

            # ---- group annotations into trees by label c1/c2 ----
            ann_groups = {}  # label -> {"ids": [...], "polys": [pts,...]}
            for a in anns_by_img.get(img_id, []):
                label = cat_name.get(a["category_id"], "")
                if label not in ("c1", "c2"):
                    continue
                seg = a.get("segmentation", [])
                # COCO polygon format: list of polygons
                assert isinstance(seg, list) and len(seg) > 0
                # seg is typically [ [x1,y1,...] ] ; but allow multiple polygons
                polys = []
                for poly_flat in seg:
                    pts = coco_poly_to_xy(poly_flat)
                    if len(pts) >= 3:
                        polys.append(pts)

                if not polys:
                    continue

                g = ann_groups.setdefault(label, {"ids": [], "polys": []})
                g["ids"].append(int(a.get("id", -1)))
                g["polys"].extend(polys)

            ann_labels = sorted(ann_groups.keys())  # stable order
            ann_trees = [ann_groups[l]["polys"] for l in ann_labels]
            ann_ids = [ann_groups[l]["ids"] for l in ann_labels]

            # Compute Y positions
            det_ys = [tree_y_from_det(t) for t in det_trees]
            ann_ys = [tree_y_from_ann(polys) for polys in ann_trees]

            # Match
            det_to_ann = match_by_y(det_ys, ann_ys)

            # Precompute masks for annotations (small count -> ok)
            ann_masks = []
            for polys in ann_trees:
                ann_masks.append(mask_from_ann(polys, W, H))

            # Iterate detections
            for di, det_tree in enumerate(det_trees):
                det_mask = mask_from_det(det_tree, W, H)
                ai = det_to_ann[di]

                if ai is None:
                    d = 0.0
                    dbg = os.path.join(args.debug_dir, f"{os.path.splitext(image_name)[0]}_det{di}_unmatched.png")
                    title = f"{image_name}\ndet={di} ann=NONE\nDICE=0.0000"
                    draw_debug(dbg, W, H, det_tree, None, title)

                    wcsv.writerow([image_name, di, "NONE", "", f"{det_ys[di]:.3f}", "", f"{d:.6f}", dbg])
                    continue

                d = dice(det_mask, ann_masks[ai])
                dbg = os.path.join(args.debug_dir, f"{os.path.splitext(image_name)[0]}_det{di}_{ann_labels[ai]}_dice{d:.3f}.png")
                title = (
                    f"{image_name}\n"
                    f"det={di} ann={ann_labels[ai]} ids={len(ann_ids[ai])}\n"
                    f"det_y={det_ys[di]:.1f} ann_y={ann_ys[ai]:.1f}  DICE={d:.4f}"
                )
                draw_debug(dbg, W, H, det_tree, ann_trees[ai], title)

                wcsv.writerow([
                    image_name,
                    di,
                    ann_labels[ai],
                    ";".join(map(str, ann_ids[ai])),
                    f"{det_ys[di]:.3f}",
                    f"{ann_ys[ai]:.3f}",
                    f"{d:.6f}",
                    dbg
                ])

    print("OK")
    print("CSV:", args.out_csv)
    print("Debug dir:", args.debug_dir)


if __name__ == "__main__":
    main()
