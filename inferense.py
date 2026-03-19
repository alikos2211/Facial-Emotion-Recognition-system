from initing import *
import torch.nn.functional as F
from model import *
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(num_classes=6, weights_path=None):
    
    model = SmallVGG(num_classes=num_classes) 
    
    if weights_path:
        
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
        
    return model.to(device).eval()

def predict_image(image_path, model, labels):
    
    img = Image.open(image_path).convert('L') 

    preprocess = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.CenterCrop(40),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        conf, index = torch.max(probs, dim=1)
        
    return labels[index.item()], conf.item() * 100

if __name__ == "__main__":
    emotion_labels = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    
    my_model = get_model(num_classes=6, weights_path="./outputs/best_model.pth")
    
    test_img = "surprise.jpg"
    label, score = predict_image(test_img, my_model, emotion_labels)
    
    print(f"Result: {label} ({score:.2f}%)")