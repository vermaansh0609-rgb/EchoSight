import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="EchoSight Assistive Vision", layout="centered")
st.title("EchoSight Assistive Vision")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def get_yolo_model():
    from ultralytics import YOLO
    return YOLO("yolov8n-seg.pt")

@st.cache_resource
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)

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

def analyze_navigation(results, cv_image):
    is_dropoff, dropoff_msg = detect_floor_dropoff(cv_image)
    if is_dropoff:
        return dropoff_msg

    boxes = results[0].boxes
    names = results[0].names
    obstacles = []
    FRAME_HEIGHT, FRAME_WIDTH = cv_image.shape[:2]

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

                pos_text = "on left" if x_center < (FRAME_WIDTH * 0.35) else ("on right" if x_center > (FRAME_WIDTH * 0.65) else "ahead")
                obstacles.append({
                    "label": label, "pos_text": pos_text, "distance": distance_band,
                    "hazard": hazard, "height_ratio": height_ratio
                })

    if obstacles:
        obstacles.sort(key=lambda item: item["height_ratio"], reverse=True)
        primary = obstacles[0]
        return f"{primary['label']} {primary['pos_text']} ({primary['distance']})"

    return "Path clear"

def identify_currency(cv_img, ocr_reader):
    ocr_results = ocr_reader.readtext(cv_img)
    text_corpus = " ".join([entry[1].strip() for entry in ocr_results])
    for denom in ["500", "200", "100", "50", "20", "10"]:
        if denom in text_corpus or f"₹{denom}" in text_corpus:
            return f"{denom} Rupees detected"

    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    h_mean, s_mean = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1])
    if s_mean > 35:
        if 15 <= h_mean <= 30: return "Likely 200 Rupees"
        elif 80 <= h_mean <= 100: return "Likely 50 Rupees"
        elif 110 <= h_mean <= 145: return "Likely 100 Rupees"
        elif 5 <= h_mean <= 18: return "Likely 10 or 20 Rupees"
    elif 30 <= h_mean <= 75 and s_mean < 40:
        return "Likely 500 Rupees"
    return "Scanning currency..."

mode = st.radio("Select Operating Mode:", ["🧭 Navigation", "📖 Text Reader (OCR)", "💵 Currency Identifier"], horizontal=True)

# Shared state between WebRTC frame callbacks
class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.latest_text = "Starting..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Run inference once every 6 frames to keep CPU usage minimal
        if self.frame_count % 6 == 0:
            if mode == "🧭 Navigation":
                seg_model = get_yolo_model()
                results = seg_model(img, verbose=False, imgsz=320)
                self.latest_text = analyze_navigation(results, img)

            elif mode == "📖 Text Reader (OCR)":
                ocr = get_ocr_reader()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ocr_results = ocr.readtext(gray)
                extracted = [e[1].strip() for e in ocr_results if e[2] > 0.35 and len(e[1].strip()) > 1]
                self.latest_text = " ".join(extracted[:4]) if extracted else "No text"

            elif mode == "💵 Currency Identifier":
                ocr = get_ocr_reader()
                self.latest_text = identify_currency(img, ocr)

        # Draw overlay text directly on live video stream
        cv2.rectangle(img, (10, 10), (img.shape[1] - 10, 60), (0, 0, 0), -1)
        cv2.putText(img, self.latest_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="echosight-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
