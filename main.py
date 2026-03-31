from fastapi import FastAPI, UploadFile, File
import shutil
import cv2
import os
import numpy as np

app = FastAPI()

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print("⚠️ Running WITHOUT AI model (Render safe mode)")


# ✅ Home API
@app.get("/")
def home():
    return {"message": "AI Video Detector Running 🚀"}


# 🎥 Extract frames
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Take every 10th frame
        if count % 10 == 0:
            frames.append(frame)

        count += 1

    cap.release()

    print(f"[DEBUG] Frames extracted: {len(frames)}")
    return frames


# 🔍 Blur detection
def get_blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# 👤 Face detection
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces)


# 🧠 SMART LOGIC (no AI model)
def analyze_video(video_path):
    frames = extract_frames(video_path)

    if len(frames) == 0:
        return "Error", 0.0

    brightness_list = []
    blur_list = []
    face_counts = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = gray.mean()
        blur = get_blur_score(frame)
        faces = detect_faces(frame)

        brightness_list.append(brightness)
        blur_list.append(blur)
        face_counts.append(faces)

    avg_brightness = np.mean(brightness_list)
    avg_blur = np.mean(blur_list)
    avg_faces = np.mean(face_counts)

    score = 0

    # Face presence
    if avg_faces > 0.5:
        score += 0.4

    # Blur check
    if avg_blur < 50:
        score -= 0.3
    else:
        score += 0.3

    # Brightness check
    if avg_brightness < 60 or avg_brightness > 200:
        score -= 0.2
    else:
        score += 0.2

    print(f"[DEBUG] Score: {score:.2f}")

    if score > 0:
        return "Likely Real", round(min(score, 1.0), 2)
    else:
        return "Likely Fake", round(abs(score), 2)


# 🚀 Upload API
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"

    print(f"[DEBUG] Uploading: {file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("[DEBUG] File saved")

    result, score = analyze_video(file_path)

    print(f"[DEBUG] Result: {result}, Confidence: {score}")

    return {
        "filename": file.filename,
        "result": result,
        "confidence": round(score, 2)
    }