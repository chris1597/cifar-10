from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class PlotHelper(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    def plot_losses(self, losses):
        """Plot the training losses over epochs."""
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, predictions, labels):
        """Plot the confusion matrix based on predictions and actual labels."""
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, xticklabels=True, yticklabels=True)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()