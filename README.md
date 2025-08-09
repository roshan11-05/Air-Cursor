🖱️ AirCursor

Control your computer **cursor** using just your **hand gestures** — no mouse needed!  
This project uses **computer vision** to track your hand movements in real time and move the cursor accordingly.

## 🚀 Features
- 👋 **Hand Tracking** using [MediaPipe](https://google.github.io/mediapipe/)
- 🖱️ **Real-time Cursor Control**
- ⚡ Smooth & Responsive tracking
- 🛠️ Easy to run on any computer with a webcam

## 🖥️ How It Works
1. **Capture Video Feed** from your webcam.
2. **Detect Hand Landmarks** using MediaPipe.
3. **Map Hand Movements** to screen coordinates.
4. **Move the Cursor** with PyAutoGUI.

## 📦 Installation
```bash
git clone https://github.com/YOUR-USERNAME/AirCursor.git
cd AirCursor
pip install -r requirements.txt```

▶️ Usage
python air_cursor.py
Point your camera towards your hand
Move your hand to move the cursor
Use a pinch or click gesture for mouse clicks
