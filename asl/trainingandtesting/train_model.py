import json
import os
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

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
DATA_DIR = "training_data/raw"
SEQ_LEN = 50
CACHE_PATH = ".cache/asl_training_cache.pkl"
CACHE_VERSION = 1

import mediapipe as mp

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "Invalid mediapipe installation: 'mp.solutions' missing. "
        "Reinstall with: pip install 'mediapipe==0.10.14' 'protobuf>=4.25.3,<5'"
    )

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.3,
)


def extract_landmarks(rgb_frame):
    result = holistic.process(rgb_frame)

    def lm(landmark_list, n):
        if landmark_list:
            return np.array([[l.x, l.y] for l in landmark_list.landmark], dtype=np.float32)
        return np.zeros((n, 2), dtype=np.float32)

    pose = lm(result.pose_landmarks, 33)[:13]
    lh = lm(result.left_hand_landmarks, 21)
    rh = lm(result.right_hand_landmarks, 21)
    return np.concatenate([pose, lh, rh], axis=0).flatten()


def normalize_landmarks(flat_landmarks: np.ndarray) -> np.ndarray:
    points = flat_landmarks.reshape(-1, 2).astype(np.float32)
    pose = points[:13]

    left_shoulder = pose[11]
    right_shoulder = pose[12]
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    points = (points - center) / scale
    return points.flatten()


def sample_frames(frames, n):
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)
    return frames[idx]


def build_clip_manifest():
    manifest = []
    for sign in SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.exists(sign_dir):
            continue
        clips = sorted(f for f in os.listdir(sign_dir) if f.endswith(".npy"))
        for clip_file in clips:
            clip_path = os.path.join(sign_dir, clip_file)
            stat = os.stat(clip_path)
            manifest.append((os.path.join(sign, clip_file), stat.st_mtime_ns, stat.st_size))
    return manifest


def load_feature_cache(cache_path: str, manifest: list[tuple[str, int, int]]):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if cache.get("version") != CACHE_VERSION:
        return None
    if cache.get("manifest") != manifest:
        return None
    return cache


def save_feature_cache(cache_path: str, manifest: list[tuple[str, int, int]], x_data: np.ndarray, y_data: list[str]):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "version": CACHE_VERSION,
                "manifest": manifest,
                "x_data": x_data,
                "y_data": y_data,
            },
            f,
        )


def select_most_informative_window(landmarks_seq: np.ndarray, window_len: int) -> np.ndarray:
    """
    Pick the contiguous window with strongest signing signal.
    This makes training robust to start/stop dead time and brief out-of-frame moments.
    """
    t = landmarks_seq.shape[0]
    if t <= window_len:
        if t < window_len:
            pad = np.repeat(landmarks_seq[-1:, :], window_len - t, axis=0)
            return np.concatenate([landmarks_seq, pad], axis=0)
        return landmarks_seq

    # Hand landmark block: 21 left + 21 right points, x/y => 84 dims.
    hands = landmarks_seq[:, 26:]
    hand_presence = (np.abs(hands).sum(axis=1) > 1e-6).astype(np.float32)
    motion = np.zeros((t,), dtype=np.float32)
    motion[1:] = np.linalg.norm(landmarks_seq[1:] - landmarks_seq[:-1], axis=1)
    frame_score = hand_presence + 0.4 * motion

    best_start = 0
    best_score = -1.0
    running = frame_score[:window_len].sum()
    best_score = running
    for start in range(1, t - window_len + 1):
        running += frame_score[start + window_len - 1] - frame_score[start - 1]
        if running > best_score:
            best_score = running
            best_start = start

    return landmarks_seq[best_start : best_start + window_len]


print("Extracting landmarks from training data...")
manifest = build_clip_manifest()
cache = load_feature_cache(CACHE_PATH, manifest)

if cache is not None:
    print(f"  Loaded cached features from {CACHE_PATH}")
    x_data = cache["x_data"]
    y_data = cache["y_data"]
else:
    x_data, y_data = [], []

    for sign in SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.exists(sign_dir):
            print(f"  WARNING: no data for '{sign}', skipping")
            continue
        clips = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
        print(f"  {sign}: {len(clips)} clips")
        for clip_file in clips:
            frames = np.load(os.path.join(sign_dir, clip_file))
            if len(frames) == 0:
                continue

            # Speed up extraction on long clips while preserving temporal coverage.
            pre_sample_n = min(len(frames), 60)
            sampled = sample_frames(frames, pre_sample_n) if len(frames) > pre_sample_n else frames
            all_lm = np.array([normalize_landmarks(extract_landmarks(f)) for f in sampled], dtype=np.float32)
            seq = select_most_informative_window(all_lm, SEQ_LEN)
            x_data.append(seq.flatten())
            y_data.append(sign)

    x_data = np.array(x_data, dtype=np.float32)
    save_feature_cache(CACHE_PATH, manifest, x_data, y_data)
    print(f"  Saved feature cache to {CACHE_PATH}")

if x_data.shape[0] == 0:
    raise RuntimeError("No training clips found. Record data first in training_data/raw.")

le = LabelEncoder()
y_enc = le.fit_transform(y_data)

print(f"\nDataset: {x_data.shape[0]} samples, {len(le.classes_)} classes")
print(f"Classes: {list(le.classes_)}")

print("\nTraining models...")
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    ),
    "SVM": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    probability=True,
                    kernel="rbf",
                    C=8,
                    gamma="scale",
                    class_weight="balanced",
                ),
            ),
        ]
    ),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
}

best_model = None
best_score = 0
best_name = ""
best_cv_model = None

for name, model in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x_data, y_enc, cv=cv, scoring="accuracy")
    mean = scores.mean()
    print(f"  {name}: {mean:.3f} +- {scores.std():.3f}")
    if mean > best_score:
        best_score = mean
        best_model = model
        best_cv_model = model
        best_name = name

print(f"\nBest: {best_name} ({best_score:.3f})")

hard_threshold = 0.70
hard_repeat = 2
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_probs = cross_val_predict(best_cv_model, x_data, y_enc, cv=cv, method="predict_proba")
cv_pred = np.argmax(cv_probs, axis=1)
cv_conf = np.max(cv_probs, axis=1)
hard_mask = (cv_pred != y_enc) | (cv_conf < hard_threshold)
hard_count = int(np.sum(hard_mask))

if hard_count > 0:
    x_hard = np.repeat(x_data[hard_mask], hard_repeat, axis=0)
    y_hard = np.repeat(y_enc[hard_mask], hard_repeat, axis=0)
    x_train = np.concatenate([x_data, x_hard], axis=0)
    y_train = np.concatenate([y_enc, y_hard], axis=0)
    hard_labels = sorted(set(y_data[i] for i, flag in enumerate(hard_mask) if flag))
    print(
        f"Hard-example pass: {hard_count} clips below {hard_threshold:.2f} confidence or misclassified. "
        f"Augmented training set: {x_train.shape[0]} samples."
    )
    print(f"Hard signs: {hard_labels}")
else:
    x_train = x_data
    y_train = y_enc
    print(f"Hard-example pass: none below {hard_threshold:.2f}; training on base dataset.")

best_model.fit(x_train, y_train)

os.makedirs("asl", exist_ok=True)
with open("asl/asl_classifier.pkl", "wb") as f:
    pickle.dump({"model": best_model, "label_encoder": le}, f)

label_map = {str(i): label for i, label in enumerate(le.classes_)}
with open("asl/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

holistic.close()

print("\nSaved to asl/asl_classifier.pkl")
print(f"Label map: {label_map}")
print("Done!")
