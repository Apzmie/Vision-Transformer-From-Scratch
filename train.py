from vision_transformer import TransformerEncoder
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def train(folder_name, d_model, num_heads, num_layers, train_num_epochs, train_save_period, batch_size=16):
    checkpoint_path = os.path.join(folder_name, ".ipynb_checkpoints")
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    dataset = ImageFolder(root=folder_name, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerEncoder(d_model, num_heads, num_layers, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())

    best_loss = float('inf')
    for epoch in range(train_num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        if (epoch + 1) % train_save_period == 0:
            print(f"[{epoch+1}/{train_num_epochs}] Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 'vit.pth')
                print(f"Model saved at epoch {epoch+1} with the best loss: {best_loss:.4f}")
