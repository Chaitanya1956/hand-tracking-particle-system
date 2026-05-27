import cv2
import mediapipe as mp
import math

# =========================
# MediaPipe Setup
# =========================

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

# =========================
# Main Loop
# =========================

while True:

    success, img = cap.read()

    if not success:
        print("Failed to grab frame")
        continue

    # Mirror effect
    img = cv2.flip(img, 1)

    h, w, c = img.shape

    # Convert to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process Hands + Face
    hand_results = hands.process(rgb_img)
    face_results = face_mesh.process(rgb_img)

    # =========================
    # HAND TRACKING
    # =========================

    if hand_results.multi_hand_landmarks:

        for hand_landmarks in hand_results.multi_hand_landmarks:

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get fingertip positions
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            # Draw circles
            cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 255), -1)
            cv2.circle(img, (index_x, index_y), 10, (0, 255, 0), -1)

            # Pinch detection
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            if distance < 40:
                cv2.putText(
                    img,
                    "PINCH",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

    # =========================
    # FACE + EYE TRACKING
    # =========================

    if face_results.multi_face_landmarks:

        for face_landmarks in face_results.multi_face_landmarks:

            # Draw face mesh
            mp_draw.draw_landmarks(
                img,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=1,
                    circle_radius=1
                )
            )

            # LEFT EYE IRIS
            left_iris = face_landmarks.landmark[468]

            lx = int(left_iris.x * w)
            ly = int(left_iris.y * h)

            cv2.circle(img, (lx, ly), 5, (255, 0, 0), -1)

            # RIGHT EYE IRIS
            right_iris = face_landmarks.landmark[473]

            rx = int(right_iris.x * w)
            ry = int(right_iris.y * h)

            cv2.circle(img, (rx, ry), 5, (0, 0, 255), -1)

            # =========================
            # SIMPLE SMILE DETECTION
            # =========================

            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]

            mx1 = int(left_mouth.x * w)
            my1 = int(left_mouth.y * h)

            mx2 = int(right_mouth.x * w)
            my2 = int(right_mouth.y * h)

            mouth_distance = math.hypot(mx2 - mx1, my2 - my1)

            if mouth_distance > 70:
                cv2.putText(
                    img,
                    "SMILE DETECTED",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    3
                )

    # =========================
    # Show Window
    # =========================

    cv2.imshow("AI Tracking System", img)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()
