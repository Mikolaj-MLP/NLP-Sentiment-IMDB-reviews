import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class Visualizer:
    def __init__(self, train_losses, valid_losses, all_labels, all_preds, all_probs, model_name):
        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.all_labels = all_labels
        self.all_preds = all_preds
        self.all_probs = all_probs
        self.model_name = model_name
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.all_labels, self.all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_losses(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
        plt.plot(range(1, len(self.valid_losses) + 1), self.valid_losses, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train and Validation Loss Over Epochs - {self.model_name}')
        plt.legend()
        plt.show()
    
    def plot_all(self):
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_losses()