import cv2
import numpy as np
from asl.predictor import ASLPredictor, NUM_FRAMES


print("Loading VideoMAE (first run downloads model files)...")
predictor = ASLPredictor()
print("Ready. Sign a word, hold for 2 seconds, release. Press Q to quit.")

cap = cv2.VideoCapture(0)
recording = False
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        recording = True
        frames = []
        cv2.putText(
            frame,
            "RECORDING...",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
    elif key == ord('s') and frames:
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
    elif key == ord('q'):
        break

    if recording:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        cv2.putText(
            frame,
            f"RECORDING... {len(frames)} frames",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
    else:
        cv2.putText(
            frame,
            "R=record  S=stop+predict  Q=quit",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow("VideoMAE ASL Test", frame)

cap.release()
cv2.destroyAllWindows()
