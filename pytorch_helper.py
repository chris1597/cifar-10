from pydantic import BaseModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

class PyTorchHelper(BaseModel):
    model: torch.nn.Module
    device: torch.device
    
    class Config:
        arbitrary_types_allowed = True

    def train_model(self, train_loader, optimizer, criterion, epochs=5):
        """Train the model with the provided data loader, optimizer, loss function, and number of epochs."""
        self.model.train()  # Set the model to training mode
        epoch_losses = []  # List to store loss per epoch

        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

        return epoch_losses

    from sklearn.metrics import accuracy_score, precision_score, recall_score

    def evaluate_model(self, data_loader):
        """Evaluate the model's performance on the provided data loader."""
        self.model.eval()  # Set the model to evaluation mode
        predictions = []
        labels_list = []
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                predictions.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(labels_list, predictions)
        # Calculate precision and recall using 'macro' averaging to treat all classes equally
        precision = precision_score(labels_list, predictions, average='macro', zero_division=0)
        recall = recall_score(labels_list, predictions, average='macro', zero_division=0)

        return accuracy, precision, recall, predictions, labels_list

