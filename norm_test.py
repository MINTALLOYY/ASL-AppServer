import torch
import numpy as np
from asl.predictor import GCN_muti_att
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download('sharonn18/tgcn-wlasl', 'checkpoints/asl100/pytorch_model.bin')
checkpoint = torch.load(path, map_location='cpu')
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']
    
model = GCN_muti_att(input_feature=100, hidden_feature=64, num_class=100, p_dropout=0.3, num_stage=20)
model.load_state_dict(checkpoint, strict=False)
model.eval()

with open('asl/label_map.json') as f:
    labels = {int(k): v for k, v in json.load(f).items()}

# Simulate a hand in center-frame (0.5, 0.5 in MediaPipe coords)
def make_fake_seq(coords_value):
    seq = np.full((1, 55, 100), coords_value, dtype=np.float32)
    return torch.FloatTensor(seq)

for name, val in [("raw [0,1] center", 0.5), ("shifted [-1,1] center", 0.0), ("zeros", 0.0)]:
    with torch.no_grad():
        out = model(make_fake_seq(val))
        probs = torch.softmax(out, dim=-1).numpy()[0]
    top3 = np.argsort(probs)[::-1][:3]
    print(f"{name}: {[(labels[i], round(float(probs[i]),3)) for i in top3]}")