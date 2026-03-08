# 🏋️ AI Fitness Trainer — Real-Time Pose Estimation & Rep Counter

A real-time fitness rep counter using **MediaPipe Pose Estimation** that tracks body keypoints via webcam and counts reps for Bicep Curls, Squats, and Push-ups by measuring joint angles.

---

## 🧠 How It Works

```
Webcam Frame → MediaPipe Pose → 33 Body Keypoints → Joint Angle → Rep Counter Logic → Live HUD
```

**For each exercise, a specific joint angle is measured:**

| Exercise | Joint | Up Angle | Down Angle |
|---|---|---|---|
| Bicep Curl | Elbow (shoulder-elbow-wrist) | < 50° | > 150° |
| Squat | Knee (hip-knee-ankle) | < 100° | > 160° |
| Push-up | Elbow (shoulder-elbow-wrist) | < 80° | > 150° |

A rep is counted when the joint moves from the **down position → up position**.

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/ai-fitness-trainer
cd ai-fitness-trainer
pip install -r requirements.txt
python fitness_trainer.py
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `1` | Switch to Bicep Curl |
| `2` | Switch to Squat |
| `3` | Switch to Push-up |
| `R` | Reset all counters |
| `Q` | Quit |

---

## 📦 Requirements

```
mediapipe>=0.10.0
opencv-python>=4.7.0
numpy>=1.24.0
```

---

## 📁 Project Structure

```
ai-fitness-trainer/
├── fitness_trainer.py    # Main application
├── requirements.txt
└── README.md
```

---

## 💡 Key Design Decisions

- **Joint angle thresholds** — each exercise uses anatomically correct angle ranges to detect movement phases
- **MediaPipe model_complexity=1** — best balance of accuracy and CPU performance
- **BGR→RGB conversion** — MediaPipe requires RGB input, OpenCV captures BGR
- **Mirror flip** — webcam is flipped horizontally for natural mirror-like interaction
- **Stage tracking** — a rep is only counted on the transition from down→up, preventing false counts

---

## 📚 References

- [MediaPipe Pose Documentation](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe Pose Landmarks](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model)

---

## 👤 Author

**Your Name** — [GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)
