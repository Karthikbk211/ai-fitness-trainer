import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ── Color Palette — Dark Minimal ──────────────────────────────
C = {
    'bg':       (8,   8,   10),
    'panel':    (18,  18,  22),
    'panel2':   (28,  28,  34),
    'border':   (55,  55,  65),
    'accent':   (255, 255, 255),   # Pure white
    'dim':      (140, 140, 150),
    'good':     (180, 255, 180),   # Soft white-green
    'warn':     (140, 180, 255),   # Soft white-blue
    'alert':    (180, 180, 255),   # Soft lavender
    'curl':     (255, 255, 255),
    'squat':    (200, 200, 210),
    'pushup':   (170, 170, 185),
}

EXERCISE_COLORS = {
    'Bicep Curl': C['curl'],
    'Squat':      C['squat'],
    'Push-up':    C['pushup'],
}


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle   = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def get_lm(landmarks, idx, w, h):
    lm = landmarks[idx]
    return [lm.x * w, lm.y * h]


def get_lm_raw(landmarks, idx):
    """Return raw normalized landmark (x,y in 0-1)."""
    return landmarks[idx]


class AngleSmoother:
    def __init__(self, window=6):
        self.buf = deque(maxlen=window)

    def update(self, val):
        if val is not None:
            self.buf.append(val)
        return float(np.mean(self.buf)) if self.buf else None


class ExerciseCounter:
    def __init__(self, name):
        self.name     = name
        self.count    = 0
        self.stage    = None
        self.smoother = AngleSmoother(window=6)
        self.feedback = ''
        self.angle    = None

    def _get_raw_angle(self, landmarks, w, h):
        try:
            if self.name == 'Bicep Curl':
                r = calculate_angle(
                    get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value,    w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value,    w, h))
                l = calculate_angle(
                    get_lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value,    w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value,    w, h))
                rv = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
                lv = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
                return (r * rv + l * lv) / (rv + lv + 1e-6)

            elif self.name == 'Squat':
                r = calculate_angle(
                    get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value,   w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value,  w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h))
                l = calculate_angle(
                    get_lm(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value,   w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value,  w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h))
                return (r + l) / 2

            elif self.name == 'Push-up':
                # ── Push-up fix ───────────────────────────────
                # Only count if body is roughly horizontal
                # Check: shoulder Y ≈ hip Y (normalized coords)
                r_shoulder = get_lm_raw(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                r_hip      = get_lm_raw(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
                r_ankle    = get_lm_raw(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value)

                # Body angle from horizontal — small = lying down, large = standing
                body_vec   = np.array([r_shoulder.x - r_ankle.x, r_shoulder.y - r_ankle.y])
                horiz_vec  = np.array([1.0, 0.0])
                body_angle = np.degrees(np.arctan2(abs(body_vec[1]), abs(body_vec[0]) + 1e-6))

                # Only track elbow angle if body is somewhat horizontal (< 45°)
                if body_angle > 45:
                    self.feedback = 'Get into push-up position'
                    return None

                return calculate_angle(
                    get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value,    w, h),
                    get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value,    w, h))
        except Exception:
            return None

    def update(self, landmarks, w, h):
        raw   = self._get_raw_angle(landmarks, w, h)
        angle = self.smoother.update(raw)
        self.angle = angle

        if angle is None:
            return None

        # ── Rep logic ─────────────────────────────────────────
        if self.name == 'Bicep Curl':
            if angle > 155:
                self.stage = 'down'
            if angle < 45 and self.stage == 'down':
                self.stage = 'up'
                self.count += 1
            # Feedback
            if self.stage == 'down' and angle < 130:
                self.feedback = 'Extend fully'
            elif self.stage == 'up' and angle > 70:
                self.feedback = 'Curl higher'
            else:
                self.feedback = 'Good form'

        elif self.name == 'Squat':
            if angle > 160:
                self.stage = 'up'
            if angle < 95 and self.stage == 'up':
                self.stage = 'down'
                self.count += 1
            # Feedback
            if self.stage == 'up' and angle < 150:
                self.feedback = 'Stand fully'
            elif self.stage == 'down' and angle > 110:
                self.feedback = 'Go lower'
            else:
                self.feedback = 'Good form'

        elif self.name == 'Push-up':
            if angle > 150:
                self.stage = 'up'
            if angle < 75 and self.stage == 'up':
                self.stage = 'down'
                self.count += 1
            # Feedback
            if self.stage == 'up' and angle < 140:
                self.feedback = 'Push up fully'
            elif self.stage == 'down' and angle > 90:
                self.feedback = 'Lower your chest'
            else:
                self.feedback = 'Good form'

        return angle

    def reset(self):
        self.count    = 0
        self.stage    = None
        self.feedback = ''
        self.smoother = AngleSmoother(window=6)
        self.angle    = None


def draw_panel(frame, x1, y1, x2, y2, alpha=0.88):
    """Draw a clean dark semi-transparent panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), C['panel'], -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C['border'], 1)


def draw_thin_bar(frame, x, y, w, h, value, max_val, color):
    """Thin progress bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), C['panel2'], -1)
    fill = int(np.clip(value / max_val, 0, 1) * w)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)


def draw_hud(frame, counters, active_exercise, fps, elapsed):
    h, w = frame.shape[:2]
    panel_w = 230

    # ── Top bar ───────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), C['bg'], -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    cv2.line(frame, (0, 50), (w, 50), C['border'], 1)

    cv2.putText(frame, 'AI FITNESS TRAINER', (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, C['accent'], 1)

    mins = int(elapsed) // 60
    secs = int(elapsed) % 60
    cv2.putText(frame, f'{mins:02d}:{secs:02d}', (w // 2 - 35, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, C['dim'], 1)
    cv2.putText(frame, f'{fps:.0f} fps', (w - 80, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C['border'], 1)

    # ── Side panel background ─────────────────────────────────
    draw_panel(frame, 0, 51, panel_w, h, alpha=0.92)

    # ── Exercise cards ────────────────────────────────────────
    card_h = 110
    for i, (name, counter) in enumerate(counters.items()):
        y0        = 60 + i * (card_h + 8)
        is_active = (name == active_exercise)
        col       = C['accent'] if is_active else C['border']
        num_col   = C['accent'] if is_active else C['dim']

        # Active indicator line
        if is_active:
            cv2.rectangle(frame, (0, y0), (3, y0 + card_h), C['accent'], -1)

        # Exercise name
        cv2.putText(frame, name.upper(), (12, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

        # Key hint
        key = str(i + 1)
        cv2.putText(frame, f'[{key}]', (panel_w - 35, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C['border'], 1)

        # Rep count
        cv2.putText(frame, str(counter.count), (12, y0 + 75),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, num_col, 2)
        cv2.putText(frame, 'reps', (85, y0 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C['dim'], 1)

        # Stage
        if is_active and counter.stage:
            cv2.putText(frame, counter.stage, (140, y0 + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C['dim'], 1)

        # Angle bar
        if is_active and counter.angle is not None:
            draw_thin_bar(frame, 12, y0 + 88, panel_w - 25, 5,
                         counter.angle, 180, C['accent'])
            cv2.putText(frame, f'{counter.angle:.0f}°', (panel_w - 50, y0 + 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C['dim'], 1)

        # Divider
        if i < len(counters) - 1:
            cv2.line(frame, (12, y0 + card_h + 4), (panel_w - 12, y0 + card_h + 4),
                     C['border'], 1)

    # ── Form feedback ─────────────────────────────────────────
    fb = counters[active_exercise].feedback
    if fb:
        fb_col = C['good'] if 'Good' in fb else C['warn']
        fb_y   = h - 65
        cv2.line(frame, (12, fb_y - 10), (panel_w - 12, fb_y - 10), C['border'], 1)
        cv2.putText(frame, fb, (12, fb_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fb_col, 1)

    # ── Bottom controls ───────────────────────────────────────
    cv2.line(frame, (12, h - 35), (panel_w - 12, h - 35), C['border'], 1)
    cv2.putText(frame, '[R] Reset    [Q] Quit', (12, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C['border'], 1)

    return frame


def draw_summary(counters, elapsed):
    """Clean dark minimal summary screen."""
    h, w = 720, 1000
    img  = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = C['bg']

    # Top border line
    cv2.line(img, (60, 80), (w - 60, 80), C['border'], 1)

    cv2.putText(img, 'WORKOUT SUMMARY', (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C['accent'], 2)

    mins = int(elapsed) // 60
    secs = int(elapsed) % 60
    total = sum(c.count for c in counters.values())
    rpm   = total / (elapsed / 60 + 1e-6)

    # Stats row
    for i, (label, val) in enumerate([
        ('TIME', f'{mins:02d}:{secs:02d}'),
        ('TOTAL REPS', str(total)),
        ('REPS / MIN', f'{rpm:.1f}'),
    ]):
        x = 60 + i * 300
        cv2.putText(img, label, (x, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C['dim'], 1)
        cv2.putText(img, val, (x, 185),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, C['accent'], 2)

    cv2.line(img, (60, 210), (w - 60, 210), C['border'], 1)

    # Per exercise breakdown
    for i, (name, counter) in enumerate(counters.items()):
        y   = 270 + i * 130
        rpm = counter.count / (elapsed / 60 + 1e-6)

        cv2.putText(img, name.upper(), (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C['dim'], 1)

        cv2.putText(img, str(counter.count), (60, y + 65),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, C['accent'], 2)
        cv2.putText(img, 'reps', (175, y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C['dim'], 1)
        cv2.putText(img, f'{rpm:.1f} rpm', (280, y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C['dim'], 1)

        # Thin bar showing proportion of total
        bar_w = w - 120
        prop  = counter.count / (total + 1e-6)
        draw_thin_bar(img, 60, y + 80, bar_w, 3, prop, 1.0, C['border'])
        draw_thin_bar(img, 60, y + 80, int(bar_w * prop), 3, 1.0, 1.0, C['accent'])

        if i < len(counters) - 1:
            cv2.line(img, (60, y + 100), (w - 60, y + 100), C['panel2'], 1)

    cv2.line(img, (60, h - 50), (w - 60, h - 50), C['border'], 1)
    cv2.putText(img, 'press any key to exit', (60, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C['border'], 1)

    return img


def main():
    counters = {
        'Bicep Curl': ExerciseCounter('Bicep Curl'),
        'Squat':      ExerciseCounter('Squat'),
        'Push-up':    ExerciseCounter('Push-up'),
    }
    exercise_keys = list(counters.keys())
    active_idx    = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Could not open webcam.')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time  = time.time()
    start_time = time.time()

    print('AI Fitness Trainer v3 — Dark Minimal')
    print('[1] Bicep Curl  [2] Squat  [3] Push-up  [R] Reset  [Q] Quit')

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.75,
        model_complexity=1
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            h, w    = frame.shape[:2]
            elapsed = time.time() - start_time

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            active_exercise = exercise_keys[active_idx]

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=C['accent'], thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=C['border'], thickness=1)
                )
                counters[active_exercise].update(
                    results.pose_landmarks.landmark, w, h)

            curr_time = time.time()
            fps       = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            frame = draw_hud(frame, counters, active_exercise, fps, elapsed)
            cv2.imshow('AI Fitness Trainer', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                active_idx = 0
            elif key == ord('2'):
                active_idx = 1
            elif key == ord('3'):
                active_idx = 2
            elif key == ord('r'):
                for c in counters.values():
                    c.reset()
                start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    summary = draw_summary(counters, elapsed)
    cv2.imshow('Workout Summary', summary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\n── Summary ──────────────────────────')
    for name, c in counters.items():
        print(f'  {name}: {c.count} reps')
    print(f'  Time: {int(elapsed)//60:02d}:{int(elapsed)%60:02d}')


if __name__ == '__main__':
    main()
