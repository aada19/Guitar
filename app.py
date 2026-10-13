
import io
import time
import wave
import threading

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
)

# ======================================================
# AI HAND GESTURE GUITAR
# Live webcam + hand recognition + virtual guitar chords
# ======================================================

st.set_page_config(
    page_title="AI Hand Gesture Guitar",
    page_icon="🎸",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #142D49);
    color: white;
}
h1, h2, h3, p, label {
    color: #F4F7FB !important;
}
div.stButton > button {
    background: #20C997;
    color: #071426;
    border-radius: 12px;
    font-weight: bold;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🎸 AI HAND GESTURE GUITAR")
st.subheader("Your hand is the guitar controller!")
st.write(
    "Use your computer webcam to change virtual guitar chords "
    "with hand gestures. Keep your palm facing the camera."
)

# ------------------------------------------------------
# CHORDS
# ------------------------------------------------------

CHORDS = {
    "C Major": {
        "symbol": "C",
        "gesture": "Open palm",
        "notes": [261.63, 329.63, 392.00],
    },
    "G Major": {
        "symbol": "G",
        "gesture": "Index finger",
        "notes": [196.00, 246.94, 392.00],
    },
    "D Major": {
        "symbol": "D",
        "gesture": "Two fingers",
        "notes": [293.66, 369.99, 440.00],
    },
    "A Minor": {
        "symbol": "Am",
        "gesture": "Three fingers",
        "notes": [220.00, 261.63, 329.63],
    },
    "E Minor": {
        "symbol": "Em",
        "gesture": "Closed fist",
        "notes": [164.81, 196.00, 246.94],
    },
}

# ------------------------------------------------------
# SHARED LIVE GESTURE STATE
# ------------------------------------------------------

class GestureState:
    def __init__(self):
        self.lock = threading.Lock()
        self.chord = None
        self.gesture = "Waiting for hand..."
        self.updated = 0.0

    def update(self, chord, gesture):
        with self.lock:
            self.chord = chord
            self.gesture = gesture
            self.updated = time.time()

    def read(self):
        with self.lock:
            return self.chord, self.gesture, self.updated


STATE = GestureState()

# ------------------------------------------------------
# GESTURE RECOGNITION
# ------------------------------------------------------

class GuitarVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.drawer = mp.solutions.drawing_utils

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        chord = None
        gesture = "No hand detected"

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark

            # Finger landmarks: tips and PIP joints.
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]

            fingers = [
                lm[tip].y < lm[pip].y
                for tip, pip in zip(tips, pips)
            ]

            # Approximate thumb extension.
            # Works best with an upright, front-facing palm.
            thumb_up = (
                abs(lm[4].x - lm[0].x)
                > abs(lm[3].x - lm[0].x) + 0.025
            )

            count = sum(fingers) + int(thumb_up)

            if count >= 5:
                gesture, chord = "Open palm", "C Major"
            elif count == 0:
                gesture, chord = "Closed fist", "E Minor"
            elif count == 1:
                gesture, chord = "Index finger", "G Major"
            elif count == 2:
                gesture, chord = "Two fingers", "D Major"
            elif count == 3:
                gesture, chord = "Three fingers", "A Minor"
            else:
                gesture, chord = "Four fingers", None

            self.drawer.draw_landmarks(
                image, hand, mp.solutions.hands.HAND_CONNECTIONS
            )

        STATE.update(chord, gesture)

        # Live on-screen guitar display.
        cv2.rectangle(image, (10, 10), (440, 112), (8, 24, 42), -1)
        cv2.putText(
            image, "AI HAND GESTURE GUITAR",
            (22, 38), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (80, 230, 190), 2
        )
        cv2.putText(
            image, f"Gesture: {gesture}",
            (22, 67), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 2
        )
        cv2.putText(
            image, f"Chord: {chord or '--'}",
            (22, 96), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (90, 190, 255), 2
        )

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )

# ------------------------------------------------------
# SYNTHESIZED GUITAR-LIKE SOUND
# ------------------------------------------------------

def make_chord_audio(frequencies, duration=1.8):
    sample_rate = 22050
    t = np.arange(
        int(sample_rate * duration), dtype=np.float32
    ) / sample_rate

    audio = np.zeros_like(t)

    # Slightly stagger the notes for a strummed effect.
    for index, frequency in enumerate(frequencies):
        delay = index * 0.035
        shifted_t = np.maximum(0, t - delay)
        active = (t >= delay).astype(np.float32)

        note = (
            np.sin(2 * np.pi * frequency * shifted_t)
            + 0.35 * np.sin(
                2 * np.pi * frequency * 2 * shifted_t
            )
            + 0.12 * np.sin(
                2 * np.pi * frequency * 3 * shifted_t
            )
        )

        envelope = np.exp(-2.0 * shifted_t) * active
        audio += note * envelope

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    pcm = (audio * 30000).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    return buffer.getvalue()

# ------------------------------------------------------
# LIVE WEBCAM
# ------------------------------------------------------

st.markdown("## 📷 Live Webcam")

st.caption(
    "Click START, allow camera permission, and hold a gesture "
    "in front of the webcam. The video overlay updates live."
)

ctx = webrtc_streamer(
    key="gesture-guitar-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GuitarVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

# ------------------------------------------------------
# CHORD DISPLAY AND PLAYBACK
# ------------------------------------------------------

st.markdown("## 🎶 Your Guitar Chord")

chord, gesture, updated = STATE.read()

if chord and time.time() - updated < 4:
    info = CHORDS[chord]

    st.success(f"Detected gesture: {gesture}")
    st.markdown(f"# 🎸 {info['symbol']} — {chord}")
    st.write("Notes:", ", ".join(map(str, info["notes"])))

    if st.button("🔊 Play detected chord"):
        st.audio(
            make_chord_audio(info["notes"]),
            format="audio/wav",
        )
else:
    st.info(
        "Start the webcam and show one of the supported gestures. "
        "When a chord is recognized, it will appear on the live video."
    )

st.caption(
    "Tip: Press Play detected chord after recognition to hear "
    "the synthesized chord. The live webcam recognition itself "
    "does not automatically trigger audio."
)

# ------------------------------------------------------
# GESTURE GUIDE
# ------------------------------------------------------

st.markdown("## 🖐️ Gesture Guide")

cols = st.columns(5)

for col, (name, info) in zip(cols, CHORDS.items()):
    with col:
        st.markdown(f"### {info['symbol']}")
        st.write(name)
        st.caption(info["gesture"])

st.divider()

st.markdown("""
**How it works**

PC Webcam → MediaPipe Hand Tracking → Gesture Recognition
→ Chord Mapping → Virtual Guitar Sound

**Built with:** Python, Streamlit, OpenCV, MediaPipe,
Streamlit-WebRTC, NumPy and PyAV.
""")
