import cv2
import numpy as np
from asl.predictor import ASLPredictor, NUM_FRAMES

def main():
    print("Loading ASL predictor...")
    predictor = ASLPredictor("asl/asl_model.bin", "asl/label_map.json")
    frame_count = 0
    
    cap = cv2.VideoCapture(0)
    print("Webcam started. Press 'q' to quit. Press 'r' to reset buffer.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        frame_count += 1
        
        # 1. Add RGB frame to rolling clip buffer
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.frame_buffer.append(rgb)

        # 2. If we have enough frames, run rolling clip prediction
        if len(predictor.frame_buffer) >= NUM_FRAMES and frame_count % 4 == 0:
            recent = predictor.frame_buffer[-48:]
            result = predictor._predict_from_recording(recent, top_k=3)
            preds = result.get("top_predictions", [])
            if frame_count % 20 == 0:
                print(
                    "Debug:",
                    f"buffer={len(predictor.frame_buffer)}",
                    f"windows={result.get('windows_evaluated', 0)}",
                    f"selected={result.get('windows_selected', 0)}",
                    f"best={preds[0]['label'] if preds else 'n/a'}",
                    f"conf={preds[0]['confidence'] if preds else 0:.4f}",
                )
            
            if preds:
                best = preds[0]
                # Display best prediction
                text = f"Pred: {best['label']} ({best['confidence']:.2f})"
                color = (0, 255, 0) if best['confidence'] > 0.7 else (0, 165, 255)
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                
                # Show 2nd and 3rd place on a separate line
                subtext = f"Also: {preds[1]['label']} ({preds[1]['confidence']:.2f}), {preds[2]['label']} ({preds[2]['confidence']:.2f})"
                cv2.putText(frame, subtext, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        else:
            cv2.putText(frame, f"Warming up... {len(predictor.frame_buffer)}/{NUM_FRAMES}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("ASL VideoMAE Debug", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            predictor.frame_buffer.clear()
            print("Buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()