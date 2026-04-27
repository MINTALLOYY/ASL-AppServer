import cv2
from asl.predictor import ASLPredictor, NUM_FRAMES


def main():
    print("Loading ASL predictor...")
    predictor = ASLPredictor()
    frame_count = 0

    cap = cv2.VideoCapture(0)
    print("Webcam started. Press 'q' to quit. Press 'r' to reset prediction buffer.")

    recording = False
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            recording = True
            frames = []
        elif key == ord("s") and frames:
            recording = False
            result = predictor._predict_from_recording(frames, top_k=3)
            preds = result.get("top_predictions", [])
            print(f"Captured {len(frames)} frames")
            print("Top predictions:", preds)
            frames = []

        if recording:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            cv2.putText(
                frame,
                f"RECORDING... {len(frames)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )
        else:
            cv2.putText(
                frame,
                f"SPACE=record S=stop+predict R=reset Q=quit ({NUM_FRAMES} frame window)",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        cv2.imshow("ASL Debug", frame)

        if key == ord('q'):
            break
        elif key == ord('r'):
            predictor.reset()
            print("Buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()