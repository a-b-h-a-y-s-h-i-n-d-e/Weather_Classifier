from PIL import Image
import torch
import torchvision.transforms as transforms
from model import WeatherCNN
import matplotlib.pyplot as plt


data_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                          std = [0.229, 0.224, 0.225])
                                     ])


def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    # plt.imshow(image)
    image = data_transforms(image)
    image = image.unsqueeze(0)
    return image

def predict_image(img_path):
    image = load_image(img_path).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()


class_labels = ["Cloudy", "Rainy", "Shine", "Sunrise"]
device = torch.device('cpu')

model = WeatherCNN().to(device)
model.load_state_dict(torch.load("trained_model.pth", map_location = device))
model.eval()
img_path = "./test_img/test2.png"
predicted_index = predict_image(img_path)
predicted_label = class_labels[predicted_index]
print("Prediction -> ", predicted_label)