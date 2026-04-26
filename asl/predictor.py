import cv2
import json
import os
import numpy as np
import mediapipe as mp
import platform
import tempfile
import threading
import urllib.request
from collections import deque
import logging
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 50   # frames per inference window (from config: NUM_SAMPLES = 50)
STRIDE          = 15   # run inference every N frames
CONFIDENCE      = 0.85  # min confidence to emit a prediction

class GraphConvolution_att(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_A=0):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(55, 55))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)  # HW
        output = torch.matmul(self.att, support)  # g
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, is_resi=True):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_resi = is_resi

        self.gc1 = GraphConvolution_att(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)

        self.gc2 = GraphConvolution_att(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        if self.is_resi:
            return y + x
        else:
            return y

class GCN_muti_att(nn.Module):
    def __init__(self, input_feature, hidden_feature, num_class, p_dropout, num_stage=1, is_resi=True):
        super(GCN_muti_att, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution_att(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(55 * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, is_resi=is_resi))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        out = torch.mean(y, dim=1)
        out = self.fc_out(out)
        return out


class ASLPredictor:
    def __init__(self, model_path: str, labels_path: str):
        logger.info("Initializing ASL PyTorch TGCN model from sharonn18/tgcn-wlasl (ASL100)...")
        # Load from HF or given path:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        try:
            self.model_path = hf_hub_download('sharonn18/tgcn-wlasl', 'checkpoints/asl100/pytorch_model.bin')
        except Exception as e:
            logger.warning(f"Could not download checkpoint: {e}. Falling back to default.")
            self.model_path = model_path
            
        # TGCN config for asl100
        self.num_samples = 50
        self.hidden_size = 64
        self.num_stages = 20
        self.drop_p = 0.3
        self.input_feature = self.num_samples * 2 # 100
        self.num_classes = 100
        
        self.model = GCN_muti_att(
            input_feature=self.input_feature, 
            hidden_feature=self.hidden_size,
            num_class=self.num_classes, 
            p_dropout=self.drop_p, 
            num_stage=self.num_stages
        )
        
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            # if weights are wrapped
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self.runtime_inference_ok = True
            self.runtime_issue = ""
            logger.info("Successfully loaded PyTorch TGCN model.")
        except Exception as exc:
            self.runtime_inference_ok = False
            self.runtime_issue = f"Failed to load PyTorch stats: {exc}"
            logger.error(self.runtime_issue)

        # Build labels - try to load local labels, fallback to indices if fail
        with open(labels_path, "r") as f:
            label_map = json.load(f)
        self.labels = {int(k): v for k, v in label_map.items()}

        self._backend = "holistic"
        self._tasks_image_cls = None
        self._tasks_image_format = None
        self._tasks_hand = None
        self._tasks_pose = None
        self.feature_dim = 225

        try:
            holistic_cls = mp.solutions.holistic.Holistic
        except AttributeError:
            try:
                from mediapipe.python.solutions import holistic as mp_holistic
                holistic_cls = mp_holistic.Holistic
            except Exception as exc:
                self._init_tasks_backend()
                holistic_cls = None

        if holistic_cls is not None:
            self.holistic = holistic_cls(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.holistic = None
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0
        self._infer_lock = threading.Lock()

    def reset(self) -> None:
        """Clear the internal frame buffer and reset counters for a new sequence."""
        with self._infer_lock:
            self.frame_buffer.clear()
            self.frame_count = 0
            logger.info("ASLPredictor state reset.")

    def _predict_probs(self, seq: np.ndarray) -> np.ndarray:
        if not self.runtime_inference_ok:
            raise RuntimeError(self.runtime_issue or "ASL model runtime is not healthy")
        
        # seq is (1, 50, 55, 2)
        if seq.ndim != 4:
            raise ValueError(f"Expected seq rank 4, got {seq.shape}")
        
        # TGCN expects: (batch, num_nodes=55, num_features=100)
        # where 100 is 50 frames * 2 (x,y)
        batch = seq.shape[0]
        # (batch, seq_len, 55, 2) -> (batch, 55, seq_len * 2)
        seq_trans = np.transpose(seq, (0, 2, 1, 3))
        seq_flat = seq_trans.reshape(batch, 55, self.num_samples * 2)

        tensor_seq = torch.FloatTensor(seq_flat).to(self.device)
        with self._infer_lock:
            with torch.no_grad():
                out = self.model(tensor_seq)
                probs = torch.softmax(out, dim=-1).cpu().numpy()[0]

        return probs

    def _top_predictions(self, probs: np.ndarray, top_k: int = 3) -> list[dict]:
        if probs.size == 0:
            return []
        top_k = max(1, min(int(top_k), int(probs.shape[0])))
        ranked = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in ranked:
            results.append(
                {
                    "index": int(idx),
                    "label": self.labels.get(int(idx), f"word_{int(idx)}"),
                    "confidence": round(float(probs[idx]), 4),
                }
            )
        return results

    def predict_sequence(self, seq: np.ndarray, top_k: int = 3) -> dict:
        probs = self._predict_probs(seq)
        top_predictions = self._top_predictions(probs, top_k=top_k)
        best = top_predictions[0] if top_predictions else None
        return {
            "best_prediction": best,
            "top_predictions": top_predictions,
        }

    def transcribe_video_file(self, file_path: str, top_k: int = 3) -> dict:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {"text": "", "best_prediction": None, "top_predictions": []}

        frame_features = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_features.append(self._extract_keypoints(frame))
        finally:
            cap.release()

        if not frame_features:
            return {"text": "", "best_prediction": None, "top_predictions": []}

        if len(frame_features) < SEQUENCE_LENGTH:
            last = frame_features[-1]
            frame_features.extend([last] * (SEQUENCE_LENGTH - len(frame_features)))

        windows = []
        for start in range(0, len(frame_features) - SEQUENCE_LENGTH + 1, STRIDE):
            windows.append(np.stack(frame_features[start : start + SEQUENCE_LENGTH]))

        if not windows:
            windows.append(np.stack(frame_features[-SEQUENCE_LENGTH:]))

        accum_probs = None
        for w in windows:
            probs = self._predict_probs(w[np.newaxis, ...])
            if accum_probs is None:
                accum_probs = np.zeros_like(probs)
            accum_probs += probs

        avg = accum_probs / len(windows)
        top_preds = self._top_predictions(avg, top_k)
        best = top_preds[0] if top_preds else None
        
        text = ""
        if best and best["confidence"] >= CONFIDENCE:
            text = best["label"]

        return {
            "text": text,
            "best_prediction": best,
            "top_predictions": top_preds,
            "frames_processed": len(frame_features),
            "windows_evaluated": len(windows),
        }

    @property
    def backend(self) -> str:
        return self._backend

    def _init_tasks_backend(self) -> None:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision

        model_dir = os.path.join(tempfile.gettempdir(), "asl_mediapipe_models")
        os.makedirs(model_dir, exist_ok=True)
        hand_model_path = os.path.join(model_dir, "hand_landmarker.task")
        if not os.path.exists(hand_model_path):
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", hand_model_path)
            
        pose_model_path = os.path.join(model_dir, "pose_landmarker_lite.task")
        if not os.path.exists(pose_model_path):
            urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", pose_model_path)

        self._tasks_hand = mp_tasks_vision.HandLandmarker.create_from_options(
            mp_tasks_vision.HandLandmarkerOptions(
                base_options=mp_tasks_python.BaseOptions(model_asset_path=hand_model_path),
                running_mode=mp_tasks_vision.RunningMode.IMAGE,
                num_hands=2,
            )
        )
        self._tasks_pose = mp_tasks_vision.PoseLandmarker.create_from_options(
            mp_tasks_vision.PoseLandmarkerOptions(
                base_options=mp_tasks_python.BaseOptions(model_asset_path=pose_model_path),
                running_mode=mp_tasks_vision.RunningMode.IMAGE,
            )
        )
        self._tasks_image_cls = mp.Image
        self._tasks_image_format = mp.ImageFormat.SRGB
        self._backend = "tasks"

    def _extract_keypoints(self, frame_bgr: np.ndarray) -> np.ndarray:
        feats, _ = self.extract_features_and_debug(frame_bgr)
        return feats

    def extract_features_and_debug(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        with self._infer_lock:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self._backend == "tasks":
                return self._extract_keypoints_tasks_with_debug(rgb)

            results = self.holistic.process(rgb)

        def lm_array(lm_list, n):
            if lm_list:
                return np.array([[l.x, l.y] for l in lm_list.landmark]) # Only x, y
            return np.zeros((n, 2))

        pose = lm_array(results.pose_landmarks, 33)[:13] # only first 13 pose joints
        lh = lm_array(results.left_hand_landmarks, 21)
        rh = lm_array(results.right_hand_landmarks, 21)

        feats = self._compose_features(pose, lh, rh)
        return feats.astype(np.float32), {}

    def _compose_features(self, pose: np.ndarray, lh: np.ndarray, rh: np.ndarray) -> np.ndarray:
        # Expected TGCN joints: 55 = 13 pose + 21 lh + 21 rh
        # Each is (N, 2)
        raw = np.concatenate([pose, lh, rh], axis=0) # (55, 2)
        if raw.shape[0] < 55:
            raw = np.pad(raw, ((0, 55 - raw.shape[0]), (0, 0)), mode="constant")
        return raw[:55]

    def _extract_keypoints_tasks_with_debug(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        image = self._tasks_image_cls(image_format=self._tasks_image_format, data=frame_rgb)
        hand_result = self._tasks_hand.detect(image)

        lh = np.zeros((21, 2), dtype=np.float32)
        rh = np.zeros((21, 2), dtype=np.float32)

        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
            try:
                hand_name = (hand_result.handedness[idx][0].category_name or "").lower()
            except:
                hand_name = ""
            
            if hand_name == "left" or not np.any(lh):
                lh = coords
            else:
                rh = coords

        pose = np.zeros((13, 2), dtype=np.float32)
        if self._tasks_pose is not None:
            pose_result = self._tasks_pose.detect(image)
            if pose_result.pose_landmarks:
                pose_full = np.array([[lm.x, lm.y] for lm in pose_result.pose_landmarks[0]], dtype=np.float32)
                pose = pose_full[:13]
                if pose.shape[0] < 13:
                    pose = np.pad(pose, ((0, 13 - pose.shape[0]), (0,0)), mode="constant")

        feats = self._compose_features(pose, lh, rh)
        return feats.astype(np.float32), {}

    def process_frame(self, frame_bytes: bytes) -> str | None:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        keypoints = self._extract_keypoints(frame)
        self.frame_buffer.append(keypoints)
        self.frame_count += 1

        if len(self.frame_buffer) == SEQUENCE_LENGTH and self.frame_count % STRIDE == 0:
            seq = np.expand_dims(np.array(self.frame_buffer), axis=0) # (1, 50, 55, 2)
            probs = self._predict_probs(seq)
            best = int(np.argmax(probs))
            conf = float(probs[best])
            if conf >= CONFIDENCE:
                return self.labels.get(best, f"word_{best}")
        return None
