from pydantic import BaseModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

class PyTorchHelper(BaseModel):
    model: torch.nn.Module
    device: torch.device
    
    class Config:
        arbitrary_types_allowed = True



    def train_model(self, train_loader, val_loader, optimizer, criterion, epochs=5):
        """Train the model and compute validation loss for each epoch.
        
        Args:
            model (torch.nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): Optimizer for model training.
            criterion (LossFunction): Loss function used during training.
            device (torch.device): Device on which the model is trained (e.g., 'cuda', 'cpu').
            epochs (int): Number of training epochs.

        Returns:
            list: List of validation losses for each epoch.
        """
        val_losses = []

        for epoch in range(epochs):
            self.model.train()  # Set the model to training mode
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            print(f'Epoch {epoch + 1}/{epochs} - Validation Loss: {val_loss:.4f}')

        return val_losses


    def evaluate_model(self, data_loader) -> tuple:
        """Evaluate the model's performance on the provided data loader.

         Args:
            data_loader (DataLoader): DataLoader for evaluation.

        Returns:
            tuple: Tuple of accuracy, precision, recall, predictions list, and labels list"""
        self.model.eval()
        predictions = []
        labels_list = []
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        accuracy = accuracy_score(labels_list, predictions)
        precision = precision_score(labels_list, predictions, average='macro', zero_division=0)
        recall = recall_score(labels_list, predictions, average='macro', zero_division=0)

        return accuracy, precision, recall, predictions, labels_list
