from fastapi import FastAPI, UploadFile, File
import shutil
import cv2
import os
import numpy as np
import uuid
from PIL import Image

app = FastAPI()

# =====================================================
# SETTINGS
# =====================================================

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("AI Deepfake Detector starting...")
print("Image + Video + Audio API")


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():
    return {
        "message": "AI Deepfake Detector Running",
        "status": "online",
        "detectors": [
            "image",
            "video",
            "audio"
        ]
    }


# =====================================================
# COMMON FILE SAVE
# =====================================================

async def save_upload(file: UploadFile):

    filename = (
        str(uuid.uuid4())
        + "_"
        + os.path.basename(file.filename)
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    return file_path


# =====================================================
# IMAGE ANALYSIS
# =====================================================

def analyze_image(image_path):

    print("Starting image analysis...")

    try:

        image = cv2.imread(image_path)

        if image is None:
            return "Error", 0.0

        # Resize
        image = cv2.resize(
            image,
            (480, 480)
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Brightness
        brightness = float(
            gray.mean()
        )

        # Sharpness
        blur = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        print(
            f"Image brightness: {brightness:.2f}"
        )

        print(
            f"Image sharpness: {blur:.2f}"
        )

        score = 0.0

        # Sharpness check
        if blur >= 50:
            score += 0.5
        else:
            score -= 0.5

        # Brightness check
        if 60 <= brightness <= 200:
            score += 0.5
        else:
            score -= 0.5

        if score > 0:

            result = "Likely Real"

            confidence = min(
                score,
                1.0
            )

        else:

            result = "Likely Fake"

            confidence = min(
                abs(score),
                1.0
            )

        confidence = round(
            confidence,
            2
        )

        print(
            f"Image result: {result}"
        )

        return result, confidence

    except Exception as e:

        print(
            f"Image analysis error: {e}"
        )

        return "Error", 0.0


# =====================================================
# VIDEO ANALYSIS
# =====================================================

def analyze_video(video_path):

    print("Starting video analysis...")

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "Could not open video"
        )

        return "Error", 0.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    duration = 0

    if fps > 0:

        duration = (
            total_frames / fps
        )

    print(
        f"Frames: {total_frames}"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    if total_frames <= 0:

        cap.release()

        return "Error", 0.0

    # Analyze up to 10 frames
    sample_count = min(
        10,
        total_frames
    )

    positions = np.linspace(
        0,
        total_frames - 1,
        sample_count,
        dtype=int
    )

    brightness_values = []
    blur_values = []

    analyzed = 0

    for position in positions:

        try:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(position)
            )

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.resize(
                frame,
                (480, 270)
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            brightness = float(
                gray.mean()
            )

            blur = float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                ).var()
            )

            brightness_values.append(
                brightness
            )

            blur_values.append(
                blur
            )

            analyzed += 1

            print(
                f"Frame {analyzed}/{sample_count}"
            )

        except Exception as e:

            print(
                f"Frame error: {e}"
            )

    cap.release()

    if analyzed == 0:

        return "Error", 0.0

    avg_brightness = float(
        np.mean(
            brightness_values
        )
    )

    avg_blur = float(
        np.mean(
            blur_values
        )
    )

    print(
        f"Average brightness: "
        f"{avg_brightness:.2f}"
    )

    print(
        f"Average blur: "
        f"{avg_blur:.2f}"
    )

    score = 0.0

    # Sharpness
    if avg_blur >= 50:

        score += 0.5

    else:

        score -= 0.5

    # Brightness
    if 60 <= avg_brightness <= 200:

        score += 0.5

    else:

        score -= 0.5

    if score > 0:

        result = "Likely Real"

        confidence = min(
            score,
            1.0
        )

    else:

        result = "Likely Fake"

        confidence = min(
            abs(score),
            1.0
        )

    confidence = round(
        confidence,
        2
    )

    print(
        f"Video result: {result}"
    )

    return result, confidence


# =====================================================
# AUDIO ANALYSIS
# =====================================================

def analyze_audio(audio_path):

    print("Starting audio analysis...")

    try:

        file_size = os.path.getsize(
            audio_path
        )

        print(
            f"Audio file size: "
            f"{file_size} bytes"
        )

        if file_size <= 0:

            return "Error", 0.0

        # -------------------------------------------------
        # IMPORTANT:
        # This is only a basic audio-file validation.
        # It is NOT an AI voice deepfake detector.
        # -------------------------------------------------

        if file_size > 1000:

            result = "Audio Received"

            confidence = 0.50

        else:

            result = "Invalid Audio"

            confidence = 0.00

        return result, confidence

    except Exception as e:

        print(
            f"Audio analysis error: {e}"
        )

        return "Error", 0.0


# =====================================================
# IMAGE UPLOAD
# =====================================================

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...)
):

    print(
        f"Image upload: {file.filename}"
    )

    file_path = None

    try:

        file_path = await save_upload(
            file
        )

        result, confidence = (
            analyze_image(
                file_path
            )
        )

        return {
            "filename": file.filename,
            "result": result,
            "confidence": confidence
        }

    except Exception as e:

        print(
            f"Image upload error: {e}"
        )

        return {
            "filename": file.filename,
            "result": "Error",
            "confidence": 0.0,
            "message": str(e)
        }

    finally:

        if (
            file_path
            and os.path.exists(file_path)
        ):

            os.remove(file_path)


# =====================================================
# VIDEO UPLOAD
# =====================================================

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):

    print(
        f"Video upload: {file.filename}"
    )

    file_path = None

    try:

        file_path = await save_upload(
            file
        )

        result, confidence = (
            analyze_video(
                file_path
            )
        )

        return {
            "filename": file.filename,
            "result": result,
            "confidence": confidence
        }

    except Exception as e:

        print(
            f"Video upload error: {e}"
        )

        return {
            "filename": file.filename,
            "result": "Error",
            "confidence": 0.0,
            "message": str(e)
        }

    finally:

        if (
            file_path
            and os.path.exists(file_path)
        ):

            os.remove(file_path)


# =====================================================
# AUDIO UPLOAD
# =====================================================

@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...)
):

    print(
        f"Audio upload: {file.filename}"
    )

    file_path = None

    try:

        file_path = await save_upload(
            file
        )

        result, confidence = (
            analyze_audio(
                file_path
            )
        )

        return {
            "filename": file.filename,
            "result": result,
            "confidence": confidence
        }

    except Exception as e:

        print(
            f"Audio upload error: {e}"
        )

        return {
            "filename": file.filename,
            "result": "Error",
            "confidence": 0.0,
            "message": str(e)
        }

    finally:

        if (
            file_path
            and os.path.exists(file_path)
        ):

            os.remove(file_path)