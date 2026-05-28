"""
AI Interactive Mirror — Apple FaceTime-inspired
================================================
FEATURES:
  HAND GESTURES
  ─────────────
  ✦ Pinch (thumb + index close)
  ✦ Thumbs Up  (thumb up, all fingers curled)
  ✦ Thumbs Down (thumb down, all fingers curled)
  ✦ Peace / V-sign (index + middle up, others curled)
  ✦ Rock On  (index + pinky up, others curled)
  ✦ Open Palm (all fingers extended)
  ✦ Fist  (all fingers curled)
  ✦ Pointing (only index extended)
  ✦ OK sign  (thumb + index circle, others spread)

  FACE EXPRESSIONS
  ────────────────
  ✦ Happy  (wide mouth corners + cheek raise)
  ✦ Sad    (mouth corners drooping downward)
  ✦ Surprised (open mouth — jaw drop)
  ✦ Brow raise (eyebrows high)
  ✦ Wink   (one eye closed, other open)

  VISUAL
  ──────
  ✦ HUD overlays with rounded pill badges
  ✦ Smooth face mesh (minimal, Apple-style)
  ✦ Landmark dots for eyes / mouth
  ✦ Cooldown-gated particles (no spam)
  ✦ Floating emoji reactions that fade & drift
  ✦ FPS counter top-right

CONTROLS
  Q — Quit
"""

import cv2
import mediapipe as mp
import math
import random
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import numpy as np



mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils
mp_styles= mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")


# Landmark indices
FINGER_TIPS   = [4, 8, 12, 16, 20]
FINGER_PIPS   = [2, 6, 10, 14, 18]  # second joints (used for curl check)
FINGER_MCPS   = [1, 5, 9, 13, 17]   # knuckles

# Face
L_EYE_TOP    = 386   # left upper eyelid
L_EYE_BOT    = 374   # left lower eyelid
R_EYE_TOP    = 159   # right upper eyelid
R_EYE_BOT    = 145   # right lower eyelid
L_IRIS       = 468
R_IRIS       = 473
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13    # upper lip centre
MOUTH_BOT    = 14    # lower lip centre
L_BROW_TOP   = 105
R_BROW_TOP   = 334
L_EYE_INNER  = 362   # eye inner corner for brow distance
R_EYE_INNER  = 133


particles = []
cooldowns  = {}   # key -> last_spawn_time

font_cache = {}
def get_font(size):
    if size not in font_cache:
        try:
            font_cache[size] = ImageFont.truetype("seguiemj.ttf", size)
        except Exception:
            try:
                font_cache[size] = ImageFont.truetype("arial.ttf", size)
            except Exception:
                font_cache[size] = ImageFont.load_default()
    return font_cache[size]

def can_spawn(key: str, cooldown_sec: float = 0.8) -> bool:
    now = time.time()
    if now - cooldowns.get(key, 0) >= cooldown_sec:
        cooldowns[key] = now
        return True
    return False

def spawn_burst(cx: int, cy: int, emojis: list[str],
                count: int = 6, spread: int = 60):
    """Spawn a burst of emoji particles around (cx, cy)."""
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 4.5)
        particles.append({
            "x":    cx + random.randint(-spread // 2, spread // 2),
            "y":    cy + random.randint(-spread // 2, spread // 2),
            "dx":   math.cos(angle) * speed,
            "dy":   math.sin(angle) * speed - 2.0,   # upward bias
            "life": random.randint(40, 70),
            "max_life": 70,
            "emoji": random.choice(emojis),
            "scale": random.uniform(0.6, 1.2),
        })

def update_particles_pil(pil_img, draw):
    for p in particles[:]:
        p["x"]   += p["dx"]
        p["y"]   += p["dy"]
        p["dy"]  += 0.12          # gravity
        p["dx"]  *= 0.97          # drag
        p["life"] -= 1

        if p["life"] <= 0:
            particles.remove(p)
            continue

        alpha = p["life"] / p["max_life"]
        font_size = int(36 * (p["scale"] * alpha + 0.3))
        font_size = max(10, font_size)
        font = get_font(font_size)

        draw.text(
            (int(p["x"]), int(p["y"])),
            p["emoji"],
            font=font,
            fill=(255, 255, 255, int(255 * alpha)),
            anchor="mm"
        )


badge_positions = {}   # label -> (x, y, alpha, decay)

def draw_pill_pil(pil_img, draw, text: str, x: int, y: int,
                  fg=(255,255,255), bg=(30,30,30), alpha=1.0, font_size=18):
    """Draw a rounded-rect pill badge using Pillow (supporting emojis)."""
    font = get_font(font_size)
    left, top, right, bottom = draw.textbbox((x, y), text, font=font, anchor='ls')
    
    pad = 10
    rx1, ry1 = left - pad, top - pad
    rx2, ry2 = right + pad, bottom + pad
    
    pill_w = rx2 - rx1
    pill_h = ry2 - ry1
    if pill_w > 0 and pill_h > 0:
        pill_overlay = Image.new('RGBA', (pill_w, pill_h), (0, 0, 0, 0))
        pill_draw = ImageDraw.Draw(pill_overlay)
        
        # Swap BGR to RGB
        bg_rgb = (bg[2], bg[1], bg[0])
        pill_draw.rounded_rectangle([0, 0, pill_w, pill_h], radius=8, fill=(bg_rgb[0], bg_rgb[1], bg_rgb[2], int(alpha * 190)))
        
        pil_img.alpha_composite(pill_overlay, (rx1, ry1))
        
    fg_rgb = (fg[2], fg[1], fg[0])
    draw.text((x, y), text, font=font, fill=(fg_rgb[0], fg_rgb[1], fg_rgb[2], int(alpha * 255)), anchor='ls')

def draw_hud_pil(pil_img, draw, w: int, h: int, fps: float):
    """Draw persistent HUD: FPS, title."""
    # Title — top left
    draw_pill_pil(pil_img, draw, "AI MIRROR", 18, 32,
                  fg=(200, 230, 255), bg=(10, 10, 40))
    # FPS — top right
    fps_text = f"{fps:.0f} fps"
    font = get_font(18)
    left, top, right, bottom = draw.textbbox((0, 0), fps_text, font=font, anchor='ls')
    tw = right - left
    draw_pill_pil(pil_img, draw, fps_text, w - tw - 30, 32,
                  fg=(180, 255, 180), bg=(10, 30, 10))


def finger_extended(lm, tip_idx: int, pip_idx: int) -> bool:
    """True if tip is farther from wrist than the pip (knuckle) is."""
    wrist = lm[0]
    tip   = lm[tip_idx]
    pip   = lm[pip_idx]
    d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
    d_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
    return d_tip > d_pip * 1.1

def fingers_state(lm) -> list[bool]:
    """
    Returns [thumb, index, middle, ring, pinky] extended booleans.
    Thumb uses horizontal axis (x) relative to its base knuckle.
    """
    extended = []

    # Thumb: use x-axis distance from MCP to tip vs MCP to IP
    thumb_tip = lm[4]
    thumb_ip  = lm[3]
    thumb_mcp = lm[2]
    dx_tip = abs(thumb_tip.x - thumb_mcp.x)
    dx_ip  = abs(thumb_ip.x  - thumb_mcp.x)
    extended.append(dx_tip > dx_ip * 1.1)

    # Other fingers
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        extended.append(finger_extended(lm, tip, pip))

    return extended   # [thumb, index, middle, ring, pinky]

def thumb_direction(lm) -> str:
    """'up' / 'down' / 'neutral' based on thumb tip vs wrist y."""
    wrist = lm[0]
    tip   = lm[4]
    mcp   = lm[2]
    # All other fingers must be curled for a clean thumbs gesture
    _, idx, mid, rng, pnk = fingers_state(lm)
    if idx or mid or rng or pnk:
        return "neutral"
    dy = tip.y - mcp.y
    if dy < -0.06:
        return "up"
    if dy >  0.06:
        return "down"
    return "neutral"

def classify_gesture(lm) -> str | None:
    """
    Returns a gesture label or None.
    Priority order matters (most specific first).
    """
    ext = fingers_state(lm)   # [thumb, index, middle, ring, pinky]
    thumb, idx, mid, rng, pnk = ext

    td = thumb_direction(lm)
    if td == "up":
        return "THUMBS UP 👍"
    if td == "down":
        return "THUMBS DOWN 👎"

    if not idx and not thumb:
        thumb_tip  = lm[4]
        index_tip  = lm[8]
        dist = math.hypot(thumb_tip.x - index_tip.x,
                          thumb_tip.y - index_tip.y)
        if dist < 0.06 and mid and rng and pnk:
            return "OK 👌"

    if idx and not mid and not rng and pnk:
        return "ROCK ON 🤘"

    if idx and mid and not rng and not pnk:
        return "PEACE ✌️"

    if idx and not mid and not rng and not pnk:
        return "POINTING ☝️"

    if all(ext):
        return "OPEN PALM 🖐"

    if not any(ext):
        return "FIST ✊"

    return None

def eye_openness(lm, top_idx: int, bot_idx: int,
                 left_corner: int, right_corner: int) -> float:
    """Ratio: eye height / eye width."""
    top = lm[top_idx]
    bot = lm[bot_idx]
    lc  = lm[left_corner]
    rc  = lm[right_corner]
    height = math.hypot(top.x - bot.x, top.y - bot.y)
    width  = math.hypot(lc.x  - rc.x,  lc.y  - rc.y)
    return height / (width + 1e-6)

def classify_expression(lm, img_w: int, img_h: int) -> list[str]:
    """Returns list of active expression labels."""
    expressions = []

    ml = lm[MOUTH_LEFT]
    mr = lm[MOUTH_RIGHT]
    mouth_width = math.hypot(mr.x - ml.x, mr.y - ml.y)

    mouth_top_lm = lm[MOUTH_TOP]
    mouth_bot_lm = lm[MOUTH_BOT]
    mouth_open   = math.hypot(mouth_top_lm.x - mouth_bot_lm.x,
                              mouth_top_lm.y - mouth_bot_lm.y)
    centre_y = (ml.y + mr.y) / 2
    corner_avg_y = (ml.y + mr.y) / 2
    nose_tip = lm[1]
    left_corner_dy  = ml.y - nose_tip.y   # negative = above nose
    right_corner_dy = mr.y - nose_tip.y

    avg_corner_dy = (left_corner_dy + right_corner_dy) / 2

    chin   = lm[152]
    chin_y = chin.y
    if mouth_open > 0.04:
        expressions.append("SURPRISED 😲")

    elif avg_corner_dy < -0.07 and mouth_width > 0.04:
        expressions.append("HAPPY 😄")

    elif avg_corner_dy > -0.04 and mouth_width < 0.045:
        # corners NOT pulled up, mouth narrow → sad
        expressions.append("SAD 😢")

    l_ratio = eye_openness(lm, 386, 374, 362, 263)
    r_ratio = eye_openness(lm, 159, 145,  33, 133)

    WINK_CLOSED = 0.15
    WINK_OPEN   = 0.25
    if l_ratio < WINK_CLOSED and r_ratio > WINK_OPEN:
        expressions.append("WINK 😉")
    elif r_ratio < WINK_CLOSED and l_ratio > WINK_OPEN:
        expressions.append("WINK 😉")

    l_brow = lm[L_BROW_TOP]
    r_brow = lm[R_BROW_TOP]
    l_eye_c = lm[L_EYE_INNER]
    r_eye_c = lm[R_EYE_INNER]
    l_brow_gap = l_eye_c.y - l_brow.y
    r_brow_gap = r_eye_c.y - r_brow.y
    if l_brow_gap > 0.055 and r_brow_gap > 0.055:
        expressions.append("SURPRISED 🤨" if "SURPRISED 😲" not in expressions
                           else "VERY SURPRISED 😱")

    return expressions


EXPRESSION_EMOJIS = {
    "HAPPY 😄":        ["😄", "😊", "🎉", "✨", "💛"],
    "SAD 😢":          ["😢", "😞", "💧", "🌧"],
    "SURPRISED 😲":    ["😮", "😲", "⚡", "💥"],
    "WINK 😉":         ["😉", "✨", "💫"],
    "SURPRISED 🤨":    ["🤨", "👀"],
    "VERY SURPRISED 😱":["😱", "💥", "⚡"],
}

GESTURE_EMOJIS = {
    "THUMBS UP 👍":   ["👍", "✨", "🔥"],
    "THUMBS DOWN 👎": ["👎", "😬"],
    "PEACE ✌️":       ["✌️", "☮️", "🌈"],
    "ROCK ON 🤘":     ["🤘", "🎸", "🔥"],
    "OK 👌":          ["👌", "✅", "💯"],
    "OPEN PALM 🖐":   ["🖐", "✋"],
    "FIST ✊":        ["✊", "💪"],
    "POINTING ☝️":    ["☝️", "💡"],
    "PINCH":          ["✨", "💎", "⭐"],
}


fps_history = deque(maxlen=30)
prev_time   = time.time()


active_labels: dict[str, float] = {}   # label -> ttl seconds

LABEL_TTL = 1.8   # seconds a label stays visible

def add_label(label: str):
    active_labels[label] = time.time() + LABEL_TTL

def draw_labels_pil(pil_img, draw, w: int, h: int):
    now = time.time()
    y_offset = 70
    to_remove = []
    for label, expiry in active_labels.items():
        remaining = expiry - now
        if remaining <= 0:
            to_remove.append(label)
            continue
        alpha = min(1.0, remaining / 0.4)   # fade out last 0.4s
        draw_pill_pil(pil_img, draw, label, 18, y_offset,
                      fg=(255, 255, 255), bg=(20, 20, 60), alpha=alpha)
        y_offset += 42
    for k in to_remove:
        del active_labels[k]

MESH_SPEC = mp_draw.DrawingSpec(color=(0, 200, 180), thickness=1, circle_radius=0)
CONTOUR_CONNECTIONS = mp_face.FACEMESH_CONTOURS   # eyes, lips, face oval only


print("AI Mirror running — press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # FPS
    now       = time.time()
    fps_history.append(1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    fps       = sum(fps_history) / len(fps_history)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    hand_res = hands.process(rgb)
    face_res = face_mesh.process(rgb)

    rgb.flags.writeable = True


    if hand_res.multi_hand_landmarks:
        for hand_lm in hand_res.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(80, 200, 255), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(200, 200, 255), thickness=2),
            )

            lm = hand_lm.landmark

            tx, ty = int(lm[4].x * w), int(lm[4].y * h)
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            pinch_dist = math.hypot(tx - ix, ty - iy)

            if pinch_dist < 35:
                add_label("PINCH ✨")
                if can_spawn("pinch", 0.3):
                    spawn_burst((tx + ix) // 2, (ty + iy) // 2,
                                GESTURE_EMOJIS["PINCH"], count=5)
                cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 200), 2, cv2.LINE_AA)
                cv2.circle(frame, ((tx + ix) // 2, (ty + iy) // 2),
                           12, (0, 255, 200), -1, cv2.LINE_AA)

            else:
                gesture = classify_gesture(lm)
                if gesture:
                    add_label(gesture)
                    if can_spawn(gesture, 1.0):
                        tip = lm[FINGER_TIPS[1]]
                        gx, gy = int(tip.x * w), int(tip.y * h)
                        spawn_burst(gx, gy,
                                    GESTURE_EMOJIS.get(gesture, ["✨"]),
                                    count=6)

            for tip_idx in FINGER_TIPS:
                px = int(lm[tip_idx].x * w)
                py = int(lm[tip_idx].y * h)
                cv2.circle(frame, (px, py), 6, (255, 255, 100), -1, cv2.LINE_AA)


    if face_res.multi_face_landmarks:
        for face_lm in face_res.multi_face_landmarks:

            lm = face_lm.landmark

            mp_draw.draw_landmarks(
                frame, face_lm,
                CONTOUR_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=MESH_SPEC,
            )

            for iris_idx, colour in [(L_IRIS, (255, 120, 60)),
                                     (R_IRIS, (60, 120, 255))]:
                ix = int(lm[iris_idx].x * w)
                iy = int(lm[iris_idx].y * h)
                cv2.circle(frame, (ix, iy), 4, colour, -1, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), 8, colour, 1,  cv2.LINE_AA)

            expressions = classify_expression(lm, w, h)

            for expr in expressions:
                add_label(expr)
                if can_spawn(expr, 1.2):
                    mx = int((lm[MOUTH_LEFT].x + lm[MOUTH_RIGHT].x) / 2 * w)
                    my = int((lm[MOUTH_LEFT].y + lm[MOUTH_RIGHT].y) / 2 * h)
                    emojis = EXPRESSION_EMOJIS.get(expr, ["✨"])
                    spawn_burst(mx, my, emojis, count=7, spread=80)

    rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    pil_img = Image.fromarray(rgba_frame)
    draw = ImageDraw.Draw(pil_img)

    update_particles_pil(pil_img, draw)

    draw_hud_pil(pil_img, draw, w, h, fps)
    draw_labels_pil(pil_img, draw, w, h)

    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)


    cv2.imshow("AI Mirror", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()