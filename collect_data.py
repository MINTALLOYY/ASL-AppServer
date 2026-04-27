import os

import cv2
import numpy as np

SIGNS = [
    "drink",
    "eat",
    "help",
    "yes",
    "no",
    "book",
    "walk",
    "play",
    "dance",
    "family",
    "school",
    "doctor",
    "want",
    "go",
    "finish",
    "give",
    "work",
    "meet",
    "woman",
    "how",
]

CLIPS_PER_SIGN = 15
# Record longer clips so start/stop noise is included; training trims to best segment.
FRAMES_PER_CLIP = 90
OUTPUT_DIR = "training_data/raw"


def record():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for sign in SIGNS:
        sign_dir = os.path.join(OUTPUT_DIR, sign)
        os.makedirs(sign_dir, exist_ok=True)
        existing = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])

        print(f"\n{'=' * 50}")
        print(f"SIGN: '{sign.upper()}' ({existing} clips already recorded)")
        print(f"Need {CLIPS_PER_SIGN} total. Will record {max(0, CLIPS_PER_SIGN - existing)} more.")
        print("Press SPACE to record a clip, S to skip this sign, Q to quit")
        print(f"{'=' * 50}")

        clip_num = existing
        while clip_num < CLIPS_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(
                display,
                f"Sign: {sign.upper()}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )
            cv2.putText(
                display,
                f"Clip {clip_num + 1}/{CLIPS_PER_SIGN}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "SPACE=record  S=skip sign  Q=quit",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )
            cv2.imshow("ASL Data Collection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Quit.")
                return
            if key == ord("s"):
                print(f"Skipping {sign}")
                break
            if key == ord(" "):
                for countdown in [3, 2, 1]:
                    for _ in range(20):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        display = frame.copy()
                        cv2.putText(
                            display,
                            f"GET READY: {countdown}",
                            (150, 250),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            4,
                        )
                        cv2.imshow("ASL Data Collection", display)
                        cv2.waitKey(50)

                frames = []
                for i in range(FRAMES_PER_CLIP):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    display = frame.copy()
                    cv2.putText(
                        display,
                        f"RECORDING {i + 1}/{FRAMES_PER_CLIP}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                    cv2.putText(
                        display,
                        "Do full sign naturally; model trims extra frames",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("ASL Data Collection", display)
                    cv2.waitKey(1)

                if len(frames) == FRAMES_PER_CLIP:
                    clip_path = os.path.join(sign_dir, f"clip_{clip_num:03d}.npy")
                    np.save(clip_path, np.array(frames))
                    clip_num += 1
                    print(f"  Saved clip {clip_num}/{CLIPS_PER_SIGN} for '{sign}'")
                else:
                    print("  Recording interrupted; clip not saved.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone recording!")


if __name__ == "__main__":
    record()
