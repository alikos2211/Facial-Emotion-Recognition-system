from initing import *


class SmallVGG(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_prob=DROPOUT_PROB):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(128),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(256),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256,512,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((2,2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*2*2, 1024), nn.ReLU(True), nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),    nn.ReLU(True), nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def make_model():
    model = SmallVGG(num_classes=NUM_CLASSES)
    return model.to(DEVICE)


m = make_model()
