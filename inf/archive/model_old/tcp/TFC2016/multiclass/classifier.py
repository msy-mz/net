# Filename: classifier.py
# Module：inf.model.tcp.TFC2016.multiclass.classifier
# Description: 封装模型1进行推理
# Author: msy
# Date: 2025

import torch
from inf.model.tcp.TFC2016.multiclass.model import FTEncoder, MLPClassifier

# 类别映射
class_names = {
    0: "BitTorrent", 1: "FTP", 2: "Gmail", 3: "MySQL",
    4: "Outlook", 5: "SMB", 6: "Skype", 7: "Weibo",
    8: "WorldOfWarcraft", 9: "Cridex", 10: "Geodo", 11: "Htbot",
    12: "Miuref", 13: "Neris", 14: "Nsis", 15: "Shifu",
    16: "Virut", 17: "Zeus"
}

# 模型结构参数
INPUT_CHANNELS = 12
FREQ_DIM = 32
HIDDEN_DIM = 128
NUM_CLASSES = 18
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型路径
ENCODER_PATH = 'inf/model/tcp/multiclass/ft.pt'
CLASSIFIER_PATH = 'inf/model/tcp/multiclass/classifier.pt'

# 加载模型
def load_models():
    encoder = FTEncoder(INPUT_CHANNELS, FREQ_DIM).to(DEVICE)
    classifier = MLPClassifier(INPUT_CHANNELS * FREQ_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# 预测函数
def predict_class(feature_vector, encoder, classifier):
    with torch.no_grad():
        x = torch.tensor(feature_vector, dtype=torch.float32).to(DEVICE).unsqueeze(0)
        z = encoder(x)
        out = classifier(z)
        pred = torch.argmax(out, dim=1).item()
        return class_names[pred]
