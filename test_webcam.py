import cv2
import numpy as np
import time
from asl.predictor import ASLPredictor, NUM_FRAMES


print("Loading ASL predictor...")
try:
    predictor = ASLPredictor()
except FileNotFoundError:
    print("Model file not found: run `python3 train_model.py` first.")
    raise
print("Ready. Sign a word, hold for 2 seconds, release. Press Q to quit.")

def countdown(cap, seconds=3):
    """Display countdown before recording starts"""
    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(
            frame,
            str(i),
            (frame.shape[1]//2 - 40, frame.shape[0]//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            5,
        )
        cv2.imshow("ASL Test", frame)
        cv2.waitKey(1)
        time.sleep(1)
    
    # Show "GO!"
    ret, frame = cap.read()
    if ret:
        cv2.putText(
            frame,
            "GO!",
            (frame.shape[1]//2 - 60, frame.shape[0]//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            5,
        )
        cv2.imshow("ASL Test", frame)
        cv2.waitKey(500)

cap = cv2.VideoCapture(0)
recording = False
frames = []
recording_start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    # If not recording, show countdown and start
    if not recording:
        countdown(cap, seconds=3)
        recording = True
        recording_start_time = time.time()
        frames = []
        print("Recording for 4 seconds...")
    
    # If recording, capture frames and check if time is up
    if recording:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        
        elapsed = time.time() - recording_start_time
        remaining = max(0, 4 - elapsed)
        
        cv2.putText(
            frame,
            f"RECORDING {remaining:.1f}s",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
        
        # Stop after 4 seconds
        if elapsed >= 4:
            recording = False
            print(f"Captured {len(frames)} frames, running inference...")
            result = predictor._predict_from_recording(frames, top_k=3)
            top3 = result.get("top_predictions", [])
            print(
                f"Debug: windows_evaluated={result.get('windows_evaluated', 0)} "
                f"windows_selected={result.get('windows_selected', 0)}"
            )
            print("Results:")
            for p in top3:
                print(f"  {p['label']}: {p['confidence']:.4f}")
            frames = []
            time.sleep(1)  # Pause before next cycle
    else:
        cv2.putText(
            frame,
            "Getting ready... Press Q to quit",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow("ASL Test", frame)

cap.release()
cv2.destroyAllWindows()
