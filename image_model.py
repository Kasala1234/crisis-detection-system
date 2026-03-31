import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# 🔹 Load pretrained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# 🔹 Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image):
    try:
        # 🔥 FIX: Ensure image is RGB (3 channels)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 🔹 Apply transforms
        image = transform(image).unsqueeze(0)

        # 🔹 Forward pass
        with torch.no_grad():
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)

        # 🔹 Get confidence
        confidence = torch.max(probs).item()

        # 🔥 Simple rule (demo purpose only)
        if confidence > 0.5:
            return 1, confidence   # possible crisis
        else:
            return 0, confidence

    except Exception as e:
        print("Image processing error:", e)
        return 0, 0.0