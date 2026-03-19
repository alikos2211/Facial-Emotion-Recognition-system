import os, time, math, copy
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.cuda.amp import autocast, GradScaler

# ==== CONFIG ====
DATA_MODE = "folders"   
IMAGE_ROOT = "C:\\Users\\Admin\\.cache\\kagglehub\\datasets\\msambare\\fer2013\\versions\\1"
train_dir = os.path.join(IMAGE_ROOT, "train")
test_dir  = os.path.join(IMAGE_ROOT, "test")
OUTPUT_DIR = "/emotions_2/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES   = 6
BATCH_SIZE    = 128
NUM_WORKERS   = 2
EPOCHS        = 50          
BASE_LR       = 0.01
WEIGHT_DECAY  = 1e-4
MOMENTUM      = 0.9
DROPOUT_PROB  = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)