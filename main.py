from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import torch
import torch.optim as optim
from pytorch_helper import PyTorchHelper
from dataset_helper import DatasetHelper
from plot_helper import PlotHelper

# define training config
config = {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001
}

# define transform pipeline for custom dataset
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# create custom dataset from image folder and label file
dataset_helper = DatasetHelper(
    csv_file='label.csv',
    img_dir='data/',
    transform=transform_pipeline
)
dataset = dataset_helper.create_dataset()


# Split the dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset)) 
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)


# define model with resnet weights frozen -> just fully connected layer is trained
device = torch.device("mps")
num_classes = len(dataset.label_to_int)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True


# train the model
optimizer = optim.Adam(model.fc.parameters(), lr=config["learning_rate"])
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device) 
pytorch_helper = PyTorchHelper(model=model, device=device)
losses = pytorch_helper.train_model(train_loader, val_loader, optimizer, criterion, epochs=config["epochs"])  # Train the model
# plot training results
plot_helper = PlotHelper()
plot_helper.plot_losses(losses)

# plot evaluation results
accuracy, precision, recall, predictions, labels = pytorch_helper.evaluate_model(test_loader)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
plot_helper.plot_confusion_matrix(predictions, labels)
