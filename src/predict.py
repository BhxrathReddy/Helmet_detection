import os
import cv2
import pytesseract
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH    = os.path.join(PROJECT_ROOT, "data", "videos", "hd.mp4")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "best.pt")  # Use your trained model here
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output", "violations")
CONF_THRESHOLD = 0.3

# If Tesseract is not on your PATH, set its exe location here:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─── PREPARE ───────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Video capture
cap = cv2.VideoCapture(VIDEO_PATH)

# Map class indices → names
# 0: with helmet, 1: without helmet, 2: rider, 3: number plate
class_map = {0: 'with helmet', 1: 'without helmet', 
             2: 'rider',        3: 'number plate'}

seen_vehicles = set()
frame_index   = 0

# ─── PROCESS FRAMES ────────────────────────────────────────────────────────

# Stream inference (no display, CPU/GPU auto)
results = model.predict(
    show=True,
    source=VIDEO_PATH,
    stream=True,
    conf=CONF_THRESHOLD
)

for result in results:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract all detected boxes
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        frame_index += 1
        continue

    # Lists to hold each class's boxes
    wh_boxes = []   # without helmet
    rider_boxes = []
    plate_boxes = []

    # Separate detections by class
    for i in range(len(boxes.cls)):
        cls_id = int(boxes.cls[i])
        coords = boxes.xyxy[i].cpu().numpy().astype(int)  # [x1,y1,x2,y2]
        if cls_id == 1:
            wh_boxes.append(coords)
        elif cls_id == 2:
            rider_boxes.append(coords)
        elif cls_id == 3:
            plate_boxes.append(coords)

    # For each helmet violation, find its rider & plate
    for wh in wh_boxes:
        x1,y1,x2,y2 = wh
        center_wh   = ((x1+x2)//2, (y1+y2)//2)

        # 1) find nearest rider that overlaps significantly with the "without helmet" detection
        best_rider = None
        min_dr     = float('inf')
        for r in rider_boxes:
            rx1,ry1,rx2,ry2 = r
            c = ((rx1+rx2)//2, (ry1+ry2)//2)
            d = (c[0]-center_wh[0])**2 + (c[1]-center_wh[1])**2
            
            # Check for significant overlap between "without helmet" and "rider" boxes
            overlap_x1 = max(x1, rx1)
            overlap_y1 = max(y1, ry1)
            overlap_x2 = min(x2, rx2)
            overlap_y2 = min(y2, ry2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                wh_area = (x2 - x1) * (y2 - y1)
                overlap_ratio = overlap_area / wh_area if wh_area > 0 else 0
                
                # Only consider riders with significant overlap (>30%)
                if overlap_ratio > 0.3 and d < min_dr:
                    min_dr, best_rider = d, r

        # 2) find nearest number plate
        best_plate = None
        min_dp     = float('inf')
        for p in plate_boxes:
            px1,py1,px2,py2 = p
            c = ((px1+px2)//2, (py1+py2)//2)
            d = (c[0]-center_wh[0])**2 + (c[1]-center_wh[1])**2
            if d < min_dp:
                min_dp, best_plate = d, p

        # proceed only if both found
        if best_rider is not None and best_plate is not None:
            # ─── OCR PREPROCESS ───────────────────────────────
            px1,py1,px2,py2 = best_plate
            plate_crop = frame[py1:py2, px1:px2]

            # grayscale → equalize → threshold → resize
            gray   = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            eq     = cv2.equalizeHist(gray)
            _, bin_ = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h,w    = bin_.shape
            proc   = cv2.resize(bin_, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

            # ─── RUN TESSERACT ────────────────────────────────
            config = ("--oem 3 --psm 7 "
                      "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            raw    = pytesseract.image_to_string(proc, config=config)
            plate_text = "".join(ch for ch in raw if ch.isalnum()).upper() or None

            # ─── SAVE CROPPED RIDER ────────────────────────────
            if plate_text and plate_text not in seen_vehicles:
                seen_vehicles.add(plate_text)
                rx1,ry1,rx2,ry2 = best_rider
                rider_crop = frame[ry1:ry2, rx1:rx2]
                fname = f"{plate_text}_frame{frame_index}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, fname), rider_crop)

    frame_index += 1

cap.release()
print(f"Done! Saved {len(seen_vehicles)} violation images to '{OUTPUT_FOLDER}/'")
