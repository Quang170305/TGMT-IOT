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

class CentroidTracker:
    def __init__(self, max_dist: int = 110, max_age: int = 35):
        self.next_id = 0
        self.objects: dict[int, dict] = {}
        self.max_dist = max_dist
        self.max_age  = max_age

    def update(self, detections):
        
        for tid in list(self.objects):
            self.objects[tid]["age"] += 1
            if self.objects[tid]["age"] > self.max_age:
                del self.objects[tid]

        updated = []
        used_ids = set()

        for (cx, cy, x1, y1, x2, y2, cls_name) in detections:
            best_id = None
            best_d = self.max_dist
            best_match_cls = None

            
            for tid, t in self.objects.items():
                if tid in used_ids:
                    continue
                d = np.hypot(cx - t["cx"], cy - t["cy"])
                if d < best_d and (t["cls"] == cls_name or best_match_cls is None):
                    best_d = d
                    best_id = tid
                    best_match_cls = t["cls"]

            if best_id is None or best_d >= self.max_dist:
               
                best_id = self.next_id
                self.next_id += 1
                self.objects[best_id] = {
                    "cx": cx, "cy": cy,
                    "counted": False,
                    "age": 0,
                    "cls": cls_name,
                    "area": (x2 - x1) * (y2 - y1)  
                }
            else:
                
                self.objects[best_id]["cx"] = cx
                self.objects[best_id]["cy"] = cy
                self.objects[best_id]["age"] = 0
               
                if best_match_cls != cls_name:
                    self.objects[best_id]["cls"] = cls_name  

            used_ids.add(best_id)
            updated.append((best_id, cx, cy, x1, y1, x2, y2, cls_name))

        return updated


def overlay_hud(frame, W, H, total_count, class_counts):
    panel_h = 14 + 26 * (len(class_counts) + 2)
    panel_w = 240

    cv2.rectangle(frame, (8, 8), (panel_w, panel_h), (25, 25, 25), -1)
    cv2.rectangle(frame, (8, 8), (panel_w, panel_h), (110, 110, 110), 2)

    cv2.putText(frame, "VEHICLE COUNTER", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)
    cv2.putText(frame, f"Total: {total_count}", (18, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 120), 2)

    y = 88
    for cls_name, cnt in sorted(class_counts.items()):
        color = CLASS_COLOR.get(cls_name, DEFAULT_COLOR)
        cv2.putText(frame, f"   {cls_name.capitalize():<12}: {cnt}",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 26


def run(video_path: str, output_path: str,
        conf_thres: float = 0.50,
        model_name: str = "yolov8n",
        show: bool = False):

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("pip install ultralytics")

    model = YOLO(f"{model_name}.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {W}x{H} | Conf: {conf_thres} | Model: {model_name}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    tracker = CentroidTracker(max_dist=110, max_age=35)

    total_count = 0
    class_counts = defaultdict(int)

    frame_idx = 0
    print("[INFO] Processing...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(frame, conf=conf_thres, iou=0.45,
                                classes=list(VEHICLE_CLASSES.keys()),
                                max_det=60, verbose=False)

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = VEHICLE_CLASSES.get(cls_id)
                if not cls_name:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area < 900:         
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, x1, y1, x2, y2, cls_name))

        tracks = tracker.update(detections)

        for (tid, cx, cy, x1, y1, x2, y2, cls_name) in tracks:
            t = tracker.objects[tid]
            color = CLASS_COLOR.get(cls_name, DEFAULT_COLOR)

           
            if not t["counted"]:
               
                if t.get("area", 0) > 0 and abs(t.get("area", 0) - (x2-x1)*(y2-y1)) < 8000:
                    t["counted"] = True
                    total_count += 1
                    class_counts[cls_name] += 1
                    print(f"[COUNT] #{tid} | {cls_name} | Frame {frame_idx}")  # debug

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} #{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

        overlay_hud(frame, W, H, total_count, class_counts)

        if frame_idx % 40 == 0 or frame_idx == total_frames:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"Frame {frame_idx}/{total_frames} ({pct:.1f}%)  |  Total counted: {total_count}")

        out.write(frame)

        if show:
            cv2.imshow("Vehicle Counter - Fixed Double Counting", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

    print("\n" + "="*70)
    print(f"TOTAL VEHICLES COUNTED: {total_count}")
    print("="*70)
    for cls_name, cnt in sorted(class_counts.items()):
        print(f"    {cls_name.capitalize():<12}: {cnt}")
    print("="*70)
    print(f"Output saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Counter - Fixed Double Counting")
    parser.add_argument("--input",  required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--conf",   type=float, default=0.52, help="Confidence threshold")
    parser.add_argument("--model",  default="yolov8n", choices=["yolov8n", "yolov8s", "yolov8m"])
    parser.add_argument("--show",   action="store_true")

    args = parser.parse_args()
    run(args.input, args.output, args.conf, args.model, args.show)