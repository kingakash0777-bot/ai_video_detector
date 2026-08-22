from fastapi import FastAPI, UploadFile, File
import shutil
import cv2
import os
import numpy as np

app = FastAPI()

# Create upload folder
os.makedirs("uploads", exist_ok=True)

# Get Haar cascade path safely
CASCADE_PATH = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("WARNING: Face cascade could not be loaded")
else:
    print("Face detector loaded successfully")

print("Running WITHOUT AI model - Render safe mode")


@app.get("/")
def home():
    return {
        "message": "AI Video Detector Running 🚀",
        "status": "online"
    }


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if count % 10 == 0:
            frames.append(frame)

        count += 1

        # Prevent extremely large processing
        if len(frames) >= 30:
            break

    cap.release()

    print(f"Frames extracted: {len(frames)}")

    return frames


def get_blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def detect_faces(frame):

    # If cascade is unavailable, simply return 0
    if face_cascade.empty():
        return 0

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    return len(faces)


def analyze_video(video_path):

    frames = extract_frames(video_path)

    if len(frames) == 0:
        return "Error", 0.0

    brightness_list = []
    blur_list = []
    face_counts = []

    for frame in frames:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        brightness = gray.mean()

        blur = get_blur_score(frame)

        faces = detect_faces(frame)

        brightness_list.append(brightness)
        blur_list.append(blur)
        face_counts.append(faces)

    avg_brightness = float(
        np.mean(brightness_list)
    )

    avg_blur = float(
        np.mean(blur_list)
    )

    avg_faces = float(
        np.mean(face_counts)
    )

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
    if avg_brightness < 60 or avg_brightness > 200:
        score -= 0.2
    else:
        score += 0.2

    print(
        f"Brightness: {avg_brightness:.2f}"
    )

    print(
        f"Blur: {avg_blur:.2f}"
    )

    print(
        f"Faces: {avg_faces:.2f}"
    )

    print(
        f"Score: {score:.2f}"
    )

    if score > 0:

        return (
            "Likely Real",
            round(min(score, 1.0), 2)
        )

    else:

        return (
            "Likely Fake",
            round(abs(score), 2)
        )


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):

    # Protect filename
    filename = os.path.basename(file.filename)

    file_path = os.path.join(
        "uploads",
        filename
    )

    print(
        f"Uploading: {filename}"
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    print("File saved")

    try:

        result, score = analyze_video(
            file_path
        )

        print(
            f"Result: {result}"
        )

        print(
            f"Confidence: {score}"
        )

        return {
            "filename": filename,
            "result": result,
            "confidence": round(
                score,
                2
            )
        }

    except Exception as e:

        print(
            f"Analysis error: {e}"
        )

        return {
            "filename": filename,
            "result": "Error",
            "confidence": 0.0,
            "error": str(e)
        }

    finally:

        # Delete uploaded video after processing
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass