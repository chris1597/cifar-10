import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch
import torch.optim as optim
from pytorch_helper import PyTorchHelper
import matplotlib.pyplot as plt
from dataset_helper import DatasetHelper
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from plot_helper import PlotHelper

print("start")

transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset_helper = DatasetHelper(
    csv_file='label.csv',
    img_dir='data/',
    transform=transform_pipeline
)

dataset = dataset_helper.create_dataset()

print("dataset created")

train_size = 50000
test_size = len(dataset) - train_size
print(test_size)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("split dataset")


device = torch.device("mps")
print(f"Using device: {device}")
num_classes = len(dataset.label_to_int)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True


optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device) 

print("model created")

# Create an instance of PyTorchHelper
pytorch_helper = PyTorchHelper(model=model, device=device)
# Train and evaluate the model
losses = pytorch_helper.train_model(train_loader, optimizer, criterion, epochs=10)  # Train the model

plot_helper = PlotHelper()
plot_helper.plot_losses(losses)

accuracy, precision, recall, predictions, labels = pytorch_helper.evaluate_model(test_loader)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

plot_helper.plot_confusion_matrix(predictions, labels)
