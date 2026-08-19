import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DeepGlobeDataset
from model import get_model
from loss import PartialCELoss

def train_model(data_dir, epochs:5, batch_size:4, learning_rate:1e-4):
    """The main function that initiates the model's training cycle."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim için seçilen donanım: {device}")

    image_dir = os.path.join(data_dir, "train") 
    mask_dir = os.path.join(data_dir, "train")

    #Dataloader
    train_dataset = DeepGlobeDataset(image_dir=image_dir, mask_dir=mask_dir, num_points=20)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = get_model(classes=6).to(device)
    criterion = PartialCELoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Educational Loss: {epoch_loss:.4f}\n")

if __name__ == "__main__":
    data_directory = "../data"  # Adjust this path to your dataset directory
    train_model(data_dir=data_directory)