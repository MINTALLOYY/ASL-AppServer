import cv2
import numpy as np
from asl.predictor import ASLPredictor, SEQUENCE_LENGTH

def main():
    print("Loading ASL predictor...")
    predictor = ASLPredictor("asl/asl_model.bin", "asl/label_map.json")
    
    cap = cv2.VideoCapture(0)
    print("Webcam started. Press 'q' to quit. Press 'r' to reset buffer.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        
        # 1. Extract 55 specific WLASL points in [-1, 1] range
        feats, _ = predictor.extract_features_and_debug(frame)
        
        # 2. Draw directly what the model receives
        for i, point in enumerate(feats):
            # Skip drawing the [-1, -1] phantom hands 
            # (which we mathematically banished to the top left)
            if point[0] <= -0.99 and point[1] <= -0.99:
                continue
                
            # Convert [-1, 1] back to [0, w] and [0, h] for drawing
            x_norm = (point[0] + 1.0) / 2.0
            y_norm = (point[1] + 1.0) / 2.0
            
            x = int(x_norm * w)
            y = int(y_norm * h)
            
            # Color code points for clarity
            if i < 13: # Pose Keypoints (Green)
                color = (0, 255, 0) 
            elif i < 13 + 21: # Left Hand (Blue)
                color = (255, 0, 0)
            else: # Right Hand (Red)
                color = (0, 0, 255)
                
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 4, color, -1)

        # 3. Add to the rolling buffer for the sliding window
        predictor.frame_buffer.append(feats)

        # 4. If we have 50 frames, check prediction!
        if len(predictor.frame_buffer) == SEQUENCE_LENGTH:
            seq = np.stack(predictor.frame_buffer)[np.newaxis, ...]
            probs = predictor._predict_probs(seq)
            preds = predictor._top_predictions(probs, top_k=3)
            
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
            cv2.putText(frame, f"Warming up... {len(predictor.frame_buffer)}/{SEQUENCE_LENGTH}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("ASL Zero-Depth 55-Point Skeleton", frame)
        
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