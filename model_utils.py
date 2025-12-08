import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        x = self.gap(x)
        x = torch.flatten(x,1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x


def load_model(model_path='car_crash_model.pth', device='cpu'):
    """
    Load the trained model
    
    Args:
        model_path: Path to the saved model weights
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in eval mode
    """
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image, device='cpu'):
    
    # Same transforms as in cnnfin.ipynb
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # If image is a file path, open it
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        # If it's already a PIL Image, use it directly
        image = Image.fromarray(image).convert('RGB')
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)


def predict_image(model, image, device='cpu', threshold=0.5):

    # Preprocess image
    image_tensor = preprocess_image(image, device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        output = output.squeeze(1)
        probability = torch.sigmoid(output).item()
    
    # Classify based on threshold
    prediction = 'Yes' if probability >= threshold else 'No'
    confidence = probability if prediction == 'Yes' else (1 - probability)
    
    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence * 100
    }

