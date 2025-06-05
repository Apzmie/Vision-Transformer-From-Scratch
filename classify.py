from vision_transformer import TransformerEncoder
import os
import shutil
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

def classify(folder_name, d_model, num_heads, num_layers, image_path):
    checkpoint_path = os.path.join(folder_name, ".ipynb_checkpoints")
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)

    dataset = ImageFolder(root=folder_name)
    num_classes = len(dataset.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoder(d_model, num_heads, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load('vit.pth'))
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()

    print(dataset.classes[pred_idx])
