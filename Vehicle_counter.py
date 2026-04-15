import cv2
import numpy as np
import argparse
from collections import defaultdict


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLOR = {
    "car":        (0,   220,   0),
    "motorcycle": (255, 165,   0),
    "bus":        (0,   165, 255),
    "truck":      (0,    60, 255),
}
DEFAULT_COLOR = (200, 200, 200)

LANE_COLOR = {
    "left":  (255, 200,  50),
    "right": (50,  200, 255),
}


class CentroidTracker:
    def __init__(self, max_dist: int = 80, max_age: int = 20):
        self.next_id = 0
        self.objects: dict[int, dict] = {}
        self.max_dist = max_dist
        self.max_age  = max_age

    def update(self, detections):
        for tid in list(self.objects):
            self.objects[tid]["age"] += 1
            if self.objects[tid]["age"] > self.max_age:
                del self.objects[tid]

        updated  = []
        used_ids = set()

        for (cx, cy, x1, y1, x2, y2, cls_name) in detections:
            best_id, best_d = None, self.max_dist

            for tid, t in self.objects.items():
                if tid in used_ids:
                    continue
                d = np.hypot(cx - t["cx"], cy - t["cy"])
                if d < best_d:
                    best_d, best_id = d, tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.objects[best_id] = {
                    "cx": cx, "cy": cy,
                    "counted": False,
                    "age": 0,
                    "cls": cls_name,
                }
            else:
                self.objects[best_id]["cx"]  = cx
                self.objects[best_id]["cy"]  = cy
                self.objects[best_id]["age"] = 0

            used_ids.add(best_id)
            updated.append((best_id, cx, cy, x1, y1, x2, y2, cls_name))

        return updated


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=20):
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy  = x2 - x1, y2 - y1
    dist    = int(np.hypot(dx, dy))
    if dist == 0:
        return
    for i in range(0, dist, dash_len * 2):
        s = (int(x1 + dx * i / dist), int(y1 + dy * i / dist))
        e = (int(x1 + dx * min(i + dash_len, dist) / dist),
             int(y1 + dy * min(i + dash_len, dist) / dist))
        cv2.line(img, s, e, color, thickness)


def overlay_hud(frame, W, H, SPLIT_X, lane_counts, lane_class_counts):
    """
    Draw:
      - vertical lane divider
      - LEFT lane HUD  (top-left)
      - RIGHT lane HUD (top-right)
    (Counting line đã được bỏ)
    """
    draw_dashed_line(frame, (SPLIT_X, 0), (SPLIT_X, H),
                     (180, 180, 180), 1, dash_len=15)
    cv2.putText(frame, "L", (SPLIT_X - 18, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, LANE_COLOR["left"],  2)
    cv2.putText(frame, "R", (SPLIT_X +  6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, LANE_COLOR["right"], 2)

    for lane, anchor_x, align in [("left", 0, "left"), ("right", W, "right")]:
        total      = lane_counts[lane]
        cls_counts = lane_class_counts[lane]
        n_rows     = len(cls_counts) + 2
        panel_h    = 14 + 24 * n_rows
        panel_w    = 200

        x0 = anchor_x if align == "left" else anchor_x - panel_w

        cv2.rectangle(frame, (x0, 0), (x0 + panel_w, panel_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (x0, 0), (x0 + panel_w, panel_h), (80, 80, 80),  1)

        lc    = LANE_COLOR[lane]
        label = "LEFT LANE" if lane == "left" else "RIGHT LANE"
        cv2.putText(frame, label,
                    (x0 + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, lc, 2)
        cv2.putText(frame, f"Total: {total}",
                    (x0 + 6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)

        y = 64
        for cls_name, cnt in sorted(cls_counts.items()):
            color = CLASS_COLOR.get(cls_name, DEFAULT_COLOR)
            cv2.putText(frame, f"  {cls_name.capitalize()}: {cnt}",
                        (x0 + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
            y += 22


def run(video_path: str,
        output_path: str,
        split_ratio: float = 0.50,
        conf_thres:  float = 0.35,
        model_name:  str   = "yolov8n",
        show:        bool  = False):

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not found.\nInstall with:  pip install ultralytics")

    print(f"[INFO] Loading model: {model_name}.pt ...")
    model = YOLO(f"{model_name}.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    SPLIT_X      = int(W * split_ratio)

    print(f"[INFO] Video  : {W}x{H} @ {fps:.1f} fps | {total_frames} frames")
    print(f"[INFO] Split X: {SPLIT_X} ({split_ratio:.0%} from left)  →  Left lane | Right lane")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    tracker = CentroidTracker(max_dist=90, max_age=25)

    lane_counts       = {"left": 0, "right": 0}
    lane_class_counts = {"left": defaultdict(int), "right": defaultdict(int)}

    frame_idx = 0
    print("[INFO] Processing ...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(
            frame,
            conf=conf_thres,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id   = int(box.cls[0])
                cls_name = VEHICLE_CLASSES.get(cls_id, "vehicle")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, x1, y1, x2, y2, cls_name))

        tracks = tracker.update(detections)

        for (tid, cx, cy, x1, y1, x2, y2, cls_name) in tracks:
            t     = tracker.objects[tid]
            color = CLASS_COLOR.get(cls_name, DEFAULT_COLOR)

            lane = "left" if cx < SPLIT_X else "right"

            if not t["counted"]:
                t["counted"] = True
                lane_counts[lane]                 += 1
                lane_class_counts[lane][cls_name] += 1

            lane_tint = LANE_COLOR[lane]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
          

            label = f"{cls_name} #{tid} {'L' if lane=='left' else 'R'}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label,
                        (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        

        overlay_hud(frame, W, H, SPLIT_X, lane_counts, lane_class_counts)

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  Frame {frame_idx}/{total_frames} ({pct:.1f}%)"
                  f"  |  Left: {lane_counts['left']}  Right: {lane_counts['right']}"
                  f"  |  Total: {sum(lane_counts.values())}")

        out.write(frame)

        if show:
            cv2.imshow("Vehicle Counter — 2 Lanes [YOLOv8]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Stopped by user.")
                break

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

    total = sum(lane_counts.values())
    print("\n" + "=" * 55)
    print(f"  TOTAL VEHICLES COUNTED : {total}")
    for lane in ("left", "right"):
        print(f"\n  {'LEFT' if lane=='left' else 'RIGHT'} LANE  ({lane_counts[lane]} vehicles)")
        for cls_name, cnt in sorted(lane_class_counts[lane].items()):
            print(f"    {cls_name.capitalize():<15}: {cnt}")
    print("=" * 55)
    print(f"[INFO] Output saved → {output_path}")
    return lane_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle counter — 2 lanes (left / right) using YOLOv8"
    )
    parser.add_argument("--input",  required=True,  help="Path to input video")
    parser.add_argument("--output", required=True,  help="Path to output annotated video")
    parser.add_argument("--split",  type=float, default=0.50,
                        help="Lane divider position 0.0–1.0 from left (default: 0.50)")
    parser.add_argument("--conf",   type=float, default=0.35,
                        help="YOLO confidence threshold (default: 0.35)")
    parser.add_argument("--model",  default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
                        help="YOLOv8 model variant (default: yolov8n)")
    parser.add_argument("--show",   action="store_true",
                        help="Show live preview window while processing")

    args = parser.parse_args()
    run(
        video_path  = args.input,
        output_path = args.output,
        split_ratio = args.split,
        conf_thres  = args.conf,
        model_name  = args.model,
        show        = args.show,
    )