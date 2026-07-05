import os
import sys
import base64
import math
import random

# Windows consoles often default to cp1252, which crashes on emoji prints
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np
import joblib
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "models/tennis_sequence_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Tennis pose model loaded")
else:
    print("⚠️ Model not found — falling back to rule-based analysis only")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Pose estimation via the modern MediaPipe Tasks API ──────────────
# (the legacy mp.solutions API was removed in newer mediapipe releases)
POSE_TASK_PATH = "models/pose_landmarker_lite.task"
POSE_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

if not os.path.exists(POSE_TASK_PATH):
    print("⬇️  Downloading BlazePose model (one-time, ~5 MB)…")
    os.makedirs("models", exist_ok=True)
    r = requests.get(POSE_TASK_URL, timeout=120)
    r.raise_for_status()
    with open(POSE_TASK_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Pose model downloaded")

pose_detector = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5))

# Skeleton connections (subset of BlazePose topology) for drawing overlays
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),          # arms
    (11, 23), (12, 24), (23, 24),                               # torso
    (23, 25), (25, 27), (27, 29), (27, 31),                     # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),                     # right leg
    (0, 11), (0, 12),                                           # neck-ish
]

SHOT_NAMES = {0: "Forehand", 1: "Backhand", 2: "Serve"}
SPIN_BY_SHOT = {"Forehand": "Topspin", "Backhand": "Slice", "Serve": "Kick/Flat"}

# Landmarks that must be visible for us to accept the picture as a real player
KEY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
MIN_VISIBILITY = 0.55


def _angle(a, b, c):
    """Angle at point b (degrees) formed by points a-b-c."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 180.0
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def detect_player(frame_bgr):
    """Run pose estimation. Returns (landmarks, landmarks) or (None, None)
    when no convincingly-visible human athlete is in the frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = pose_detector.detect(mp_img)
    if not res.pose_landmarks:
        return None, None
    lm = res.pose_landmarks[0]
    vis = [(lm[i].visibility if lm[i].visibility is not None else 1.0)
           for i in KEY_LANDMARKS]
    if float(np.mean(vis)) < MIN_VISIBILITY:
        return None, None
    return lm, lm


def draw_skeleton(frame_bgr, lm):
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in POSE_CONNECTIONS:
        cv2.line(annotated, pts[a], pts[b], (176, 227, 123), 3, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(annotated, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
    return annotated


def frame_to_data_url(frame_bgr, max_width=520):
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def classify_shot(lm):
    """Blend the trained RandomForest with a heuristic fallback."""
    heuristic = "Serve" if lm[16].y < lm[0].y else ("Backhand" if lm[16].x < lm[11].x else "Forehand")
    if model is None:
        return heuristic, 0.0
    feats = np.array([[lm[12].x, lm[12].y, lm[14].x, lm[14].y, lm[16].x, lm[16].y]])
    try:
        proba = model.predict_proba(feats)[0]
        cid = int(np.argmax(proba))
        conf = float(proba[cid])
        # The model was trained on raw coords (not scale-invariant), so only
        # trust it when it is confident; otherwise use the heuristic
        if conf >= 0.5:
            return SHOT_NAMES.get(cid, heuristic), conf
        return heuristic, conf
    except Exception:
        return heuristic, 0.0


def biomech_report(lm):
    """Rule-based joint-angle analysis: score, feedback, mistakes, injury flags."""
    elbow = _angle(lm[12], lm[14], lm[16])          # right elbow
    knee = _angle(lm[24], lm[26], lm[28])           # right knee
    shoulder_tilt = abs(lm[11].y - lm[12].y)
    hip_shoulder_sep = abs((lm[11].x - lm[12].x) - (lm[23].x - lm[24].x))
    contact_height = lm[0].y - lm[16].y             # wrist above head → positive
    stance_width = abs(lm[27].x - lm[28].x)         # ankle spread
    hip_mid_x = (lm[23].x + lm[24].x) / 2
    sho_mid_x = (lm[11].x + lm[12].x) / 2
    torso_lean = math.degrees(math.atan2(abs(sho_mid_x - hip_mid_x),
                                         abs(((lm[11].y + lm[12].y) / 2) - ((lm[23].y + lm[24].y) / 2)) or 1e-3))
    balance = max(0, 100 - int(abs(sho_mid_x - hip_mid_x) * 400))

    score = 100
    feedback, mistakes, injuries = [], [], []

    if elbow < 70:
        score -= 12
        mistakes.append("Elbow too tucked at contact — extend through the ball")
        injuries.append("⚠️ Cramped elbow angles under load can strain the medial elbow (tennis elbow risk)")
    elif elbow > 165:
        score -= 6
        mistakes.append("Arm fully locked — keep a slight elbow bend for control")
    else:
        feedback.append(f"✅ Healthy elbow extension ({int(elbow)}°)")

    if knee > 165:
        score -= 10
        mistakes.append("Legs too straight — bend the knees to load power from the ground")
        injuries.append("⚠️ Stiff-legged strokes transfer shock to knees and lower back")
    elif knee < 95:
        feedback.append("💪 Deep knee load — great leg drive")
    else:
        feedback.append(f"✅ Good knee bend ({int(knee)}°)")

    if shoulder_tilt > 0.09:
        feedback.append("✅ Strong shoulder tilt — good kinetic chain rotation")
    else:
        score -= 6
        mistakes.append("Shoulders too level — rotate the trunk more into the shot")

    if hip_shoulder_sep > 0.04:
        feedback.append("✅ Hip–shoulder separation detected — power position")
    else:
        score -= 4
        feedback.append("→ Work on separating hips from shoulders during preparation")

    if contact_height > 0.05:
        feedback.append("✅ High contact point — ideal for serves and kick topspin")

    score = int(max(55, min(98, score + random.randint(-2, 2))))
    arm_extension = int(min(elbow / 180 * 100, 100))
    power_index = int(max(0, min(100, 100 - abs(elbow - 115) * 0.45
                                 - max(0, knee - 140) * 0.5
                                 + (15 if shoulder_tilt > 0.09 else 0))))
    risk_level = "HIGH" if len(injuries) >= 2 else ("MEDIUM" if injuries else "LOW")
    return {
        "score": score,
        "elbow_angle": round(elbow, 1),
        "knee_angle": round(knee, 1),
        "shoulder_tilt_deg": round(math.degrees(math.asin(min(shoulder_tilt * 2, 1))), 1),
        "torso_lean_deg": round(torso_lean, 1),
        "stance_width": round(stance_width, 3),
        "balance": balance,
        "wrist_height": round(contact_height, 2),
        "arm_extension": arm_extension,
        "power_index": power_index,
        "hip_shoulder_sep": round(hip_shoulder_sep, 3),
        "risk_level": risk_level,
        "feedback": feedback,
        "mistakes": mistakes,
        "injury_flags": injuries,
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/health')
def health():
    return jsonify({"ok": True, "model_loaded": model is not None, "groq": bool(GROQ_API_KEY)})


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file"}), 400
    file = request.files['file']
    safe_name = secure_filename(file.filename) or 'upload.bin'
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)
    return jsonify({
        "success": True,
        "filepath": filepath,
        "type": (file.content_type or 'application/octet-stream').split('/')[0],
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    file_type = data.get('type')
    filepath = data.get('filepath', '')

    # Never trust a client-supplied path outside the uploads folder
    if not filepath or os.path.basename(os.path.dirname(filepath)) != 'uploads':
        filepath = os.path.join(UPLOAD_FOLDER, os.path.basename(filepath))
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "File not found on server"}), 404

    if file_type == 'image':
        return analyze_image(filepath)
    return analyze_video(filepath)


def analyze_image(filepath):
    frame = cv2.imread(filepath)
    if frame is None:
        return jsonify({"success": False, "error": "Could not read the image file"}), 400

    lm, res = detect_player(frame)
    if lm is None:
        return jsonify({
            "success": False,
            "no_player": True,
            "error": "No tennis player detected in this image. Please upload a clear photo of a person mid-stroke (full body visible works best).",
        })

    shot, conf = classify_shot(lm)
    report = biomech_report(lm)
    annotated = frame_to_data_url(draw_skeleton(frame, res))

    return jsonify({
        "success": True,
        "type": "image",
        "form_score": report["score"],
        "metrics": {k: report[k] for k in
                    ("elbow_angle", "knee_angle", "shoulder_tilt_deg",
                     "torso_lean_deg", "stance_width", "balance",
                     "wrist_height", "arm_extension", "power_index",
                     "hip_shoulder_sep", "risk_level")},
        "annotated_image": annotated,
        "confidence": round(conf * 100),
        "summary": {
            "shot_type": shot,
            "spin_type": SPIN_BY_SHOT.get(shot, "-"),
            "feedback": report["feedback"],
            "mistakes": report["mistakes"],
            "injury_flags": report["injury_flags"],
            "elbow_angle": report["elbow_angle"],
            "knee_angle": report["knee_angle"],
        },
    })


def analyze_video(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return jsonify({"success": False, "error": "Could not open the video file"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_samples = 14
    idxs = np.linspace(0, max(total - 1, 0), num=min(n_samples, max(total, 1)), dtype=int)

    frames_out, wrist_track, reports, shots = [], [], [], []
    detected_any = False

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        lm, res = detect_player(frame)
        t = round(idx / fps, 2)
        if lm is None:
            frames_out.append({"t": t, "img": frame_to_data_url(frame), "pose": False})
            continue
        detected_any = True
        wrist_track.append((t, lm[16].x, lm[16].y))
        reports.append(biomech_report(lm))
        shots.append(classify_shot(lm)[0])
        frames_out.append({"t": t, "img": frame_to_data_url(draw_skeleton(frame, res)), "pose": True})
    cap.release()

    if not detected_any:
        return jsonify({
            "success": False,
            "no_player": True,
            "error": "No tennis player detected in this video. Please upload footage where the player's full body is visible.",
        })

    # Racket-head speed proxy: peak wrist displacement between samples
    speed_kmh = 0
    for i in range(1, len(wrist_track)):
        t0, x0, y0 = wrist_track[i - 1]
        t1, x1, y1 = wrist_track[i]
        dt = max(t1 - t0, 1e-3)
        # normalized units → assume ~2m body frame width
        v = math.hypot(x1 - x0, y1 - y0) * 2.0 / dt * 3.6
        speed_kmh = max(speed_kmh, v)
    speed_kmh = int(min(max(speed_kmh * 4, 40), 190))  # scale to racket head, clamp

    main_shot = max(set(shots), key=shots.count) if shots else "Forehand"
    avg_score = int(np.mean([r["score"] for r in reports])) if reports else 75
    all_feedback, all_mistakes, all_injuries = [], [], []
    for r in reports:
        for f in r["feedback"]:
            if f not in all_feedback:
                all_feedback.append(f)
        for m_ in r["mistakes"]:
            if m_ not in all_mistakes:
                all_mistakes.append(m_)
        for inj in r["injury_flags"]:
            if inj not in all_injuries:
                all_injuries.append(inj)

    scores = [r["score"] for r in reports]
    consistency = int(max(0, 100 - (np.std(scores) * 6))) if len(scores) > 1 else 100
    last = reports[-1] if reports else {}
    return jsonify({
        "success": True,
        "type": "video",
        "form_score": avg_score,
        "ball_speed_kmh": speed_kmh,
        "metrics": {
            "elbow_angle": last.get("elbow_angle"),
            "knee_angle": last.get("knee_angle"),
            "shoulder_tilt_deg": last.get("shoulder_tilt_deg"),
            "torso_lean_deg": last.get("torso_lean_deg"),
            "stance_width": last.get("stance_width"),
            "balance": last.get("balance"),
            "wrist_height": last.get("wrist_height"),
            "arm_extension": last.get("arm_extension"),
            "power_index": last.get("power_index"),
            "hip_shoulder_sep": last.get("hip_shoulder_sep"),
            "risk_level": last.get("risk_level"),
            "consistency": consistency,
            "frames_analyzed": len(reports),
        },
        "frames": frames_out,
        "summary": {
            "shot_type": main_shot,
            "spin_type": SPIN_BY_SHOT.get(main_shot, "-"),
            "feedback": all_feedback[:6],
            "mistakes": all_mistakes[:5],
            "injury_flags": all_injuries[:4],
        },
        "shot_sequence": shots,
    })


COACH_SYSTEM_PROMPT = (
    "You are CourtVision AI Coach, a friendly professional tennis coach with 30 years "
    "of experience. Give short, specific, actionable tennis technique advice. Use the "
    "player's latest analysis data when provided. Keep answers under 120 words, use "
    "bullet points when listing drills, and stay encouraging."
)


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    messages = data.get('messages', [])[-10:]
    context = data.get('context')
    if not GROQ_API_KEY:
        return jsonify({"success": False,
                        "reply": "AI coach is offline — add GROQ_API_KEY to your .env file (get a free key at console.groq.com)."})
    sys_msg = COACH_SYSTEM_PROMPT
    if context:
        sys_msg += f"\n\nPlayer's latest analysis: {context}"
    try:
        r = requests.post(GROQ_URL, timeout=30,
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                          json={"model": GROQ_MODEL, "max_tokens": 400,
                                "messages": [{"role": "system", "content": sys_msg}] + messages})
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        return jsonify({"success": True, "reply": reply})
    except Exception as e:
        return jsonify({"success": False, "reply": f"Coach is unavailable right now ({e})."})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 CourtVision AI Server → http://127.0.0.1:{port}")
    # use_reloader=False: uploads landing in ./uploads would otherwise restart
    # the server mid-request
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
