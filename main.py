from fastapi import FastAPI, UploadFile, File
import shutil
import cv2
import os
import numpy as np
import uuid

app = FastAPI()

# ==============================
# FOLDERS
# ==============================

os.makedirs("uploads", exist_ok=True)

# ==============================
# FACE DETECTOR
# ==============================

CASCADE_PATH = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("❌ ERROR: Haar Cascade could not be loaded")
else:
    print("✅ Haar Cascade loaded successfully")

print("🚀 AI Video Detector started")


# ==============================
# HOME
# ==============================

@app.get("/")
def home():
    return {
        "message": "AI Video Detector Running 🚀",
        "status": "online"
    }


# ==============================
# FRAME ANALYSIS
# ==============================

def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Could not open video")
        return "Error", 0.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    print(
        f"🎥 Total frames: {total_frames}, FPS: {fps}"
    )

    # Analyze maximum 30 frames
    max_frames = 30

    if total_frames <= 0:
        cap.release()
        return "Error", 0.0

    step = max(
        1,
        total_frames // max_frames
    )

    brightness_values = []
    blur_values = []
    face_values = []

    frame_number = 0
    analyzed = 0

    while cap.isOpened() and analyzed < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % step == 0:

            try:

                # Resize frame for faster processing
                frame = cv2.resize(
                    frame,
                    (640, 360)
                )

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                # --------------------------
                # Brightness
                # --------------------------

                brightness = float(
                    gray.mean()
                )

                brightness_values.append(
                    brightness
                )

                # --------------------------
                # Blur
                # --------------------------

                blur = cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                ).var()

                blur_values.append(
                    float(blur)
                )

                # --------------------------
                # Face detection
                # --------------------------

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5
                )

                face_values.append(
                    len(faces)
                )

                analyzed += 1

            except Exception as e:

                print(
                    f"⚠️ Frame analysis error: {e}"
                )

        frame_number += 1

    cap.release()

    print(
        f"✅ Analyzed {analyzed} frames"
    )

    if analyzed == 0:
        return "Error", 0.0

    # ==============================
    # AVERAGES
    # ==============================

    avg_brightness = np.mean(
        brightness_values
    )

    avg_blur = np.mean(
        blur_values
    )

    avg_faces = np.mean(
        face_values
    )

    print(
        f"Brightness: {avg_brightness:.2f}"
    )

    print(
        f"Blur: {avg_blur:.2f}"
    )

    print(
        f"Faces: {avg_faces:.2f}"
    )

    # ==============================
    # SCORING
    # ==============================

    score = 0.0

    # Face presence
    if avg_faces > 0.5:
        score += 0.4

    # Blur
    if avg_blur < 50:
        score -= 0.3
    else:
        score += 0.3

    # Brightness
    if 60 <= avg_brightness <= 200:
        score += 0.2
    else:
        score -= 0.2

    print(
        f"Final score: {score:.2f}"
    )

    # ==============================
    # RESULT
    # ==============================

    if score > 0:

        confidence = min(
            score,
            1.0
        )

        return (
            "Likely Real",
            round(confidence, 2)
        )

    else:

        confidence = min(
            abs(score),
            1.0
        )

        return (
            "Likely Fake",
            round(confidence, 2)
        )


# ==============================
# UPLOAD API
# ==============================

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):

    # Unique filename
    filename = (
        str(uuid.uuid4())
        + "_"
        + file.filename
    )

    file_path = os.path.join(
        "uploads",
        filename
    )

    print(
        f"📥 Upload received: {file.filename}"
    )

    try:

        # Save uploaded file
        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"✅ File saved: {file_path}"
        )

        # Analyze
        result, confidence = analyze_video(
            file_path
        )

        print(
            f"🎯 Result: {result}"
        )

        print(
            f"📊 Confidence: {confidence}"
        )

        return {
            "filename": file.filename,
            "result": result,
            "confidence": confidence
        }

    except Exception as e:

        print(
            f"❌ Upload processing error: {e}"
        )

        return {
            "filename": file.filename,
            "result": "Error",
            "confidence": 0.0,
            "message": str(e)
        }

    finally:

        # Delete uploaded video
        if os.path.exists(file_path):

            try:
                os.remove(file_path)

                print(
                    "🗑️ Temporary video deleted"
                )

            except Exception as e:

                print(
                    f"⚠️ Could not delete file: {e}"
                )