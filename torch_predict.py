from PIL import Image
import os
from os.path import join
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision
from torchvision import transforms as tt
import torch.nn as nn
torch.manual_seed(42)


# Define project/data directory/file paths
model_folder = "models/model_91.3751"
project_dir = os.getcwd()
data_dir = join(project_dir, "dog-breeds-data", "images", "Images")

print("Loading model...", end='\r')

# Load class labels
classes = None
with open(join(project_dir, model_folder, "classes.txt"), 'r') as f:
    classes = f.read().splitlines()

num_classes = len(classes)

# Load hyperparameters
with open(join(project_dir, model_folder, "hyperparameters.json"), 'r') as f:
    hp_dict = json.load(f)

size = hp_dict["size"]

# Load base model
model = torchvision.models.inception_v3(pretrained=True)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Remove fully connected layer
num_ftrs = model.fc.in_features
model.aux_logits = False
fc = nn.Sequential(
    nn.Linear(num_ftrs, num_classes)
)
model.fc = fc

state = torch.load(join(project_dir, model_folder, "model.pt"))
model.load_state_dict(state["model_state_dict"])

device = torch.device("mps")
model.to(device)

# define transforms
resize = tt.Resize((size, size))
convert_tensor = tt.ToTensor()
convert_pil = tt.ToPILImage()

softmax = nn.Softmax()

# returns top 3 predictions and confidence probabilities
def predict(img_path):
    img = Image.open(img_path)
    img = resize(img)

    if len(img.getbands()) == 4:
        img = convert_tensor(img)
        img = img[:3, :, :]
        img = convert_pil(img)

    img = convert_tensor(img)
    img = torch.unsqueeze(img, dim=0) # add dimension
    img = img.to(device) # send to gpu

    with torch.no_grad():
        model.eval()
        output_probs = softmax(model(img).data.flatten())
        probs, predicted_classes = torch.topk(output_probs, k=3, sorted=True)
        return [(classes[predicted_classes.data[i].item()], probs.data[i].item()) for i in range(len(predicted_classes))]


# borzoi
# imgPath = "/Users/PaulG/Desktop/Machine Learning/object-detection/dog-breeds/dog-breeds-data/images/Images/n02090622-borzoi/n02090622_906.jpg"
# print(predict(imgPath))

def main():
    while True:
        path = input("Enter path to image:\n").strip().replace('\ ', ' ')
        if not os.path.isfile(path):
            print("Please provide a valid file path")
            continue

        try:
            predictions = predict(path)
        except Exception as e:
            print(e)
        else:
            print()
            for i, pred in enumerate(predictions):
                print(f"{i+1}) {pred[0].replace('_', ' ')} - {round(pred[1]*100, 1)}%")

        print()

if __name__ == "__main__":
    main()