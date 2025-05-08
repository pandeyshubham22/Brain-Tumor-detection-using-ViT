# ===== Setup: Install packages and import dependencies =====
!pip install timm kagglehub ipywidgets matplotlib gradio  # Added ipywidgets, matplotlib, and gradio

import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import kagglehub
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import gradio as gr
from IPython.display import display, HTML

# ===== Download & Prepare Dataset via KaggleHub =====
# Downloads the latest version and returns either a directory or a ZIP path
raw_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset", force_download=True)
print("Raw dataset path:", raw_path)

# If it's a ZIP file, extract it; if it's already a directory, use it directly
if os.path.isdir(raw_path):
    data_dir = raw_path
else:
    extract_to = "/content/brain_tumor_data"
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(raw_path, "r") as zf:
        zf.extractall(extract_to)
    # If ZIP contained a single root folder, use it
    items = os.listdir(extract_to)
    if len(items) == 1 and os.path.isdir(os.path.join(extract_to, items[0])):
        data_dir = os.path.join(extract_to, items[0])
    else:
        data_dir = extract_to

# Define train/test directories
train_dir = os.path.join(data_dir, "Training")
test_dir  = os.path.join(data_dir, "Testing")
print("Training data folder:", train_dir)
print("Testing data folder: ", test_dir)

# ===== Device & Reproducibility =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.manual_seed(42)
if device.type == "cuda":
    torch.cuda.manual_seed(42)

# ===== Data Transforms & Loaders =====
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
test_ds  = datasets.ImageFolder(test_dir,  transform=test_transform)

batch_size = 32 if torch.cuda.is_available() else 16
num_workers = 4 if torch.cuda.is_available() else 0

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

class_names = train_ds.classes
print(f"Found {len(class_names)} classes:", class_names)
print("Train samples:", len(train_ds), "| Test samples:", len(test_ds))

# ===== Model: Vision Transformer Setup =====
num_classes = len(class_names)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
model.to(device)

# Optional PyTorch 2.0 compile
if hasattr(torch, 'compile') and callable(torch.compile):
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile")
    except Exception as e:
        print(f"Model compilation failed: {e}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

# ===== Training Loop with Mixed Precision & tqdm =====
num_epochs = 3
best_acc = 0.0

# Function to save model
def save_model(model, filename="best_vit_brain_tumor.pth"):
    torch.save(model.state_dict(), filename)
    return filename

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += bs

        loop.set_postfix(loss=running_loss/total, acc=100*running_correct/total)

    train_loss = running_loss / total
    train_acc  = 100 * running_correct / total

    # Evaluation
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            bs = imgs.size(0)
            test_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += bs

    test_loss /= test_total
    test_acc  = 100 * test_correct / test_total

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        model_path = save_model(model)
        print(f"❗ New best model saved (Acc: {best_acc:.2f}%)")

# ===== Inference Function =====
def preprocess_image(img):
    """Preprocess an image for inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

def predict_image(image):
    """Run inference on a single image"""
    if image is None:
        return "No image uploaded"

    # Convert to PIL image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Ensure model is in eval mode
    model.eval()

    # Preprocess image
    img_tensor = preprocess_image(image).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    # Create result string with probabilities
    result = f"Prediction: {class_names[pred_idx]}\n\nProbabilities:\n"
    for i, (class_name, prob) in enumerate(zip(class_names, probs.cpu().numpy())):
        result += f"{class_name}: {prob*100:.2f}%\n"

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display image
    ax1.imshow(image, cmap='gray' if image.mode == 'L' else None)
    ax1.set_title(f"Prediction: {class_names[pred_idx]}")
    ax1.axis('off')

    # Display probability bar chart
    bars = ax2.bar(class_names, probs.cpu().numpy() * 100)
    ax2.set_title("Prediction Probabilities (%)")
    ax2.set_ylim(0, 100)
    for bar, p in zip(bars, probs):
        height = p.item() * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}%', ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, result

# ===== Gradio Interface =====
def setup_gradio_ui():
    """Setup and launch Gradio interface"""
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Plot(label="Visualization"),
            gr.Textbox(label="Prediction Details")
        ],
        title="Brain Tumor MRI Classifier",
        description=f"Upload a brain MRI image to classify it into one of these categories: {', '.join(class_names)}",
        examples=[
            os.path.join(test_dir, class_name, os.listdir(os.path.join(test_dir, class_name))[0])
            for class_name in class_names if os.path.exists(os.path.join(test_dir, class_name))
        ],
        allow_flagging="never"
    )
    return interface.launch(share=True, inline=True)

# ===== Load Best Model and Start UI =====
print("\n===== Training complete! Setting up inference UI =====")
try:
    model.load_state_dict(torch.load("best_vit_brain_tumor.pth"))
    print("Loaded best model from saved checkpoint")
except:
    print("Using most recent model state (could not load best model)")

# Start Gradio UI
ui = setup_gradio_ui()
