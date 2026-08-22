from fastapi import FastAPI, UploadFile, File
import shutil
import cv2
import os
import numpy as np
import uuid

app = FastAPI()

# =====================================================
# UPLOAD DIRECTORY
# =====================================================

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("🚀 AI Video Detector starting...")
print("⚠️ Running in Render lightweight mode")


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():
    return {
        "message": "AI Video Detector Running 🚀",
        "status": "online"
    }


# =====================================================
# VIDEO ANALYSIS
# =====================================================

def analyze_video(video_path):

    print("🎥 Starting video analysis...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Could not open video")
        return "Error", 0.0

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
        f"🎬 Frames: {total_frames}"
    )

    print(
        f"⏱️ Duration: {duration:.2f} seconds"
    )

    if total_frames <= 0:
        cap.release()
        return "Error", 0.0

    # Analyze only 10 frames
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

    # =================================================
    # SAMPLE FRAMES
    # =================================================

    for position in positions:

        try:

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

            # Resize
            frame = cv2.resize(
                frame,
                (480, 270)
            )

            # Grayscale
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # -----------------------------------------
            # BRIGHTNESS
            # -----------------------------------------

            brightness = float(
                gray.mean()
            )

            brightness_values.append(
                brightness
            )

            # -----------------------------------------
            # BLUR
            # -----------------------------------------

            blur = cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()

            blur_values.append(
                float(blur)
            )

            analyzed += 1

            print(
                f"✅ Frame {analyzed}/{sample_count} "
                f"| Brightness: {brightness:.1f} "
                f"| Blur: {blur:.1f}"
            )

        except Exception as e:

            print(
                f"⚠️ Frame error: {e}"
            )

    cap.release()

    # =================================================
    # CHECK
    # =================================================

    if analyzed == 0:

        print(
            "❌ No frames analyzed"
        )

        return "Error", 0.0

    # =================================================
    # AVERAGES
    # =================================================

    avg_brightness = float(
        np.mean(brightness_values)
    )

    avg_blur = float(
        np.mean(blur_values)
    )

    print(
        f"💡 Average brightness: "
        f"{avg_brightness:.2f}"
    )

    print(
        f"🔍 Average blur: "
        f"{avg_blur:.2f}"
    )

    # =================================================
    # SCORING
    # =================================================

    score = 0.0

    # -----------------------------------------
    # BLUR
    # -----------------------------------------

    if avg_blur < 50:

        score -= 0.5

        print(
            "⚠️ Video is relatively blurry"
        )

    else:

        score += 0.5

        print(
            "✅ Video sharpness acceptable"
        )

    # -----------------------------------------
    # BRIGHTNESS
    # -----------------------------------------

    if 60 <= avg_brightness <= 200:

        score += 0.5

        print(
            "✅ Brightness acceptable"
        )

    else:

        score -= 0.5

        print(
            "⚠️ Unusual brightness"
        )

    # =================================================
    # RESULT
    # =================================================

    print(
        f"🧠 Final score: {score:.2f}"
    )

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
        f"📊 Confidence: {confidence}"
    )

    return result, confidence


# =====================================================
# UPLOAD
# =====================================================

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):

    print(
        f"📥 Upload received: {file.filename}"
    )

    # Unique filename
    filename = (
        str(uuid.uuid4())
        + "_"
        + os.path.basename(file.filename)
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    try:

        # ---------------------------------------------
        # SAVE VIDEO
        # ---------------------------------------------

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(
            "✅ Video saved successfully"
        )

        # ---------------------------------------------
        # ANALYZE
        # ---------------------------------------------

        result, confidence = analyze_video(
            file_path
        )

        # ---------------------------------------------
        # RESPONSE
        # ---------------------------------------------

        response = {
            "filename": file.filename,
            "result": result,
            "confidence": confidence
        }

        print(
            f"📤 Response: {response}"
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

        # ---------------------------------------------
        # DELETE VIDEO
        # ---------------------------------------------

        if os.path.exists(file_path):

            try:

                os.remove(file_path)

                print(
                    "🗑️ Temporary video deleted"
                )

            except Exception as e:

                print(
                    f"⚠️ Delete error: {e}"
                )