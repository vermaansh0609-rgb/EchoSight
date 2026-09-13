import base64
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import easyocr

app = FastAPI(title="EchoSight Assistive System")

# Initialize models (falls back to CPU automatically in free Hugging Face containers)
seg_model = YOLO("yolov8n-seg.pt")
ocr_reader = easyocr.Reader(["en"], gpu=False)

def detect_floor_dropoff(cv_image):
    height, width = cv_image.shape[:2]
    floor_roi = cv_image[int(height * 0.70):height, :]
    gray = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=15)

    if lines is not None:
        horizontal_line_count = 0
        for line in lines:
            coords = line.ravel()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = map(int, coords[:4])
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            if angle < 20 or angle > 160:
                horizontal_line_count += 1

        if horizontal_line_count >= 3:
            return True, "Warning: drop-off or steps ahead."
            
    return False, ""

def compute_walkable_corridor(results, cv_image):
    height, width = cv_image.shape[:2]
    lower_half_y = int(height * 0.5)
    obstacle_mask = np.zeros((height - lower_half_y, width), dtype=np.uint8)

    if results[0].masks is not None:
        for mask_tensor in results[0].masks.data:
            m = mask_tensor.cpu().numpy()
            m_resized = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
            obstacle_mask = np.bitwise_or(obstacle_mask, (m_resized[lower_half_y:, :] > 0.5).astype(np.uint8))

    free_space = 1 - obstacle_mask
    col_density = np.sum(free_space, axis=0)
    total_free = np.sum(col_density)

    if total_free == 0:
        return "Warning: path fully blocked ahead."

    free_x_center = np.sum(np.arange(width) * col_density) / total_free
    if free_x_center < 110:
        return "Steer slightly left."
    elif free_x_center > 210:
        return "Steer slightly right."
    return ""

def analyze_navigation(results, cv_image):
    is_dropoff, dropoff_msg = detect_floor_dropoff(cv_image)
    if is_dropoff:
        return dropoff_msg, 0.0, True

    boxes = results[0].boxes
    names = results[0].names
    obstacles = []
    FRAME_HEIGHT, FRAME_WIDTH = 240.0, 320.0

    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            if label in ['person', 'chair', 'couch', 'bottle', 'backpack', 'table', 'door', 'bed']:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_height = y2 - y1
                x_center = (x1 + x2) / 2.0

                height_ratio = box_height / FRAME_HEIGHT
                if height_ratio > 0.60:
                    distance_band, hazard = "under 1 meter", "CRITICAL"
                elif height_ratio > 0.30:
                    distance_band, hazard = "1 to 2 meters", "WARNING"
                else:
                    distance_band, hazard = "over 2 meters", "INFO"

                pan_val = ((x_center / FRAME_WIDTH) * 2.0) - 1.0
                pos_text = "on your left" if x_center < 105 else ("on your right" if x_center > 215 else "directly ahead")

                obstacles.append({
                    "label": label, "pos_text": pos_text, "distance": distance_band,
                    "hazard": hazard, "pan": pan_val, "height_ratio": height_ratio
                })

    if obstacles:
        obstacles.sort(key=lambda item: item["height_ratio"], reverse=True)
        primary = obstacles[0]
        if primary["hazard"] == "CRITICAL":
            return f"Stop! {primary['label']} {primary['pos_text']}, {primary['distance']}.", primary["pan"], True

    steering_advice = compute_walkable_corridor(results, cv_image)
    if steering_advice:
        return steering_advice, 0.0, False

    if obstacles:
        primary = obstacles[0]
        return f"{primary['label']} {primary['pos_text']}, {primary['distance']}.", primary["pan"], True

    return "Path clear straight ahead.", 0.0, False

def identify_currency(cv_img):
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h_mean, s_mean = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1])

    ocr_results = ocr_reader.readtext(cv_img)
    text_corpus = " ".join([entry[1].strip() for entry in ocr_results])
    for denom in ["500", "200", "100", "50", "20", "10"]:
        if denom in text_corpus or f"₹{denom}" in text_corpus:
            return f"{denom} Rupees note detected."

    if s_mean > 35:
        if 15 <= h_mean <= 30: return "Likely 200 Rupees note."
        elif 80 <= h_mean <= 100: return "Likely 50 Rupees note."
        elif 110 <= h_mean <= 145: return "Likely 100 Rupees note."
        elif 5 <= h_mean <= 18: return "Likely 10 or 20 Rupees note."
    elif 30 <= h_mean <= 75 and s_mean < 40:
        return "Likely 500 Rupees note."
    return "No clear banknote detected. Hold closer."

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...), mode: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if mode == "NAVIGATION":
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        results = seg_model(pil_img, verbose=False)
        alert_text, pan_val, beep = analyze_navigation(results, cv_img)
        return JSONResponse({"alert": alert_text, "pan": pan_val, "beep": beep})

    elif mode == "OCR":
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
        ocr_results = ocr_reader.readtext(clahe)
        extracted = [e[1].strip() for e in ocr_results if e[2] > 0.25 and len(e[1].strip()) > 1]
        msg = f"Text reads: {' '.join(extracted)}" if extracted else "No text detected."
        return JSONResponse({"alert": msg, "pan": 0.0, "beep": False})

    elif mode == "CURRENCY":
        cash_msg = identify_currency(cv_img)
        return JSONResponse({"alert": cash_msg, "pan": 0.0, "beep": False})

# Mount the static frontend directory to serve the webpage
app.mount("/", StaticFiles(directory="static", html=True), name="static")
