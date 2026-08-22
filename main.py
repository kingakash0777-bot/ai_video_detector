from fastapi import FastAPI, UploadFile, File
import shutil
import cv2
import os
import numpy as np
import uuid

app = FastAPI()

# =========================================================
# FOLDERS
# =========================================================

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================================
# FACE DETECTOR
# =========================================================

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
print("⚠️ Running in lightweight detection mode")


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "message": "AI Video Detector Running 🚀",
        "status": "online"
    }


# =========================================================
# VIDEO ANALYSIS
# =========================================================

def analyze_video(video_path):

    print("🎥 Starting video analysis...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Could not open video")
        return "Error", 0.0

    # Get video information
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    duration = 0

    if fps > 0:
        duration = total_frames / fps

    print(
        f"🎬 Total frames: {total_frames}"
    )

    print(
        f"⏱️ Video duration: {duration:.2f} seconds"
    )

    if total_frames <= 0:
        cap.release()
        return "Error", 0.0

    # =====================================================
    # SAMPLE ONLY 10 FRAMES
    # =====================================================

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
    face_values = []

    analyzed_frames = 0

    # =====================================================
    # ANALYZE SELECTED FRAMES
    # =====================================================

    for position in positions:

        try:

            # Jump directly to frame
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(position)
            )

            ret, frame = cap.read()

            if not ret:
                print(
                    f"⚠️ Could not read frame {position}"
                )
                continue

            # Resize for faster processing
            frame = cv2.resize(
                frame,
                (480, 270)
            )

            # Convert to grayscale
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # =================================================
            # BRIGHTNESS
            # =================================================

            brightness = float(
                gray.mean()
            )

            brightness_values.append(
                brightness
            )

            # =================================================
            # BLUR
            # =================================================

            blur = cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()

            blur_values.append(
                float(blur)
            )

            # =================================================
            # FACE DETECTION
            # =================================================

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            face_count = len(faces)

            face_values.append(
                face_count
            )

            analyzed_frames += 1

            print(
                f"✅ Frame {analyzed_frames}/{sample_count} "
                f"| Faces: {face_count} "
                f"| Brightness: {brightness:.1f} "
                f"| Blur: {blur:.1f}"
            )

        except Exception as e:

            print(
                f"⚠️ Frame analysis error: {e}"
            )

    cap.release()

    # =====================================================
    # CHECK ANALYSIS
    # =====================================================

    if analyzed_frames == 0:

        print(
            "❌ No frames could be analyzed"
        )

        return "Error", 0.0

    # =====================================================
    # CALCULATE AVERAGES
    # =====================================================

    avg_brightness = float(
        np.mean(brightness_values)
    )

    avg_blur = float(
        np.mean(blur_values)
    )

    avg_faces = float(
        np.mean(face_values)
    )

    print(
        f"📊 Average brightness: "
        f"{avg_brightness:.2f}"
    )

    print(
        f"📊 Average blur: "
        f"{avg_blur:.2f}"
    )

    print(
        f"📊 Average faces: "
        f"{avg_faces:.2f}"
    )

    # =====================================================
    # SCORING
    # =====================================================

    score = 0.0

    # -----------------------------------------------------
    # Face presence
    # -----------------------------------------------------

    if avg_faces > 0.5:

        score += 0.4

        print(
            "✅ Face detected"
        )

    else:

        print(
            "⚠️ No consistent face detected"
        )

    # -----------------------------------------------------
    # Blur
    # -----------------------------------------------------

    if avg_blur < 50:

        score -= 0.3

        print(
            "⚠️ Video appears blurry"
        )

    else:

        score += 0.3

        print(
            "✅ Video sharpness acceptable"
        )

    # -----------------------------------------------------
    # Brightness
    # -----------------------------------------------------

    if 60 <= avg_brightness <= 200:

        score += 0.2

        print(
            "✅ Brightness acceptable"
        )

    else:

        score -= 0.2

        print(
            "⚠️ Unusual brightness"
        )

    print(
        f"🧠 Final score: {score:.2f}"
    )

    # =====================================================
    # RESULT
    # =====================================================

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
        f"🎯 Result: {result}"
    )

    print(
        f"📈 Confidence: {confidence}"
    )

    return result, confidence


# =========================================================
# UPLOAD API
# =========================================================

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):

    print(
        f"📥 Upload received: {file.filename}"
    )

    # Create unique filename
    safe_name = (
        str(uuid.uuid4())
        + "_"
        + os.path.basename(file.filename)
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        safe_name
    )

    try:

        # =================================================
        # SAVE VIDEO
        # =================================================

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            f"✅ Video saved: {file_path}"
        )

        # =================================================
        # ANALYZE
        # =================================================

        result, confidence = analyze_video(
            file_path
        )

        # =================================================
        # RESPONSE
        # =================================================

        response = {
            "filename": file.filename,
            "result": result,
            "confidence": confidence
        }

        print(
            f"📤 Sending response: {response}"
        )

        return response

    except Exception as e:

        print(
            f"❌ ERROR: {e}"
        )

        return {
            "filename": file.filename,
            "result": "Error",
            "confidence": 0.0,
            "message": str(e)
        }

    finally:

        # =================================================
        # DELETE TEMPORARY VIDEO
        # =================================================

        if os.path.exists(file_path):

            try:

                os.remove(
                    file_path
                )

                print(
                    "🗑️ Temporary video deleted"
                )

            except Exception as e:

                print(
                    f"⚠️ Could not delete video: {e}"
                )