import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class ModelComparator:
    def __init__(self):
        self.results = {}
    
    def add_model(self, model_name, train_losses, valid_losses, valid_accuracies, all_labels, all_preds, all_probs, train_time):
        """Add metrics for a model."""
        self.results[model_name] = {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'valid_accuracies': valid_accuracies,
            'all_labels': all_labels,
            'all_preds': all_preds,
            'all_probs': all_probs,
            'train_time': train_time  # In seconds
        }
    
    def summary_table(self):
        """Generate a summary table of key metrics."""
        summary = []
        for model_name, metrics in self.results.items():
            accuracy = sum(a == p for a, p in zip(metrics['all_labels'], metrics['all_preds'])) / len(metrics['all_labels'])
            fpr, tpr, _ = roc_curve(metrics['all_labels'], metrics['all_probs'])
            roc_auc = auc(fpr, tpr)
            summary.append({
                'Model': model_name,
                'Test Accuracy': accuracy,
                'AUC': roc_auc,
                'Final Train Loss': metrics['train_losses'][-1],
                'Final Valid Loss': metrics['valid_losses'][-1],
                'Final Valid Accuracy': metrics['valid_accuracies'][-1],
                'Training Time (s)': metrics['train_time']
            })
        df = pd.DataFrame(summary)
        return df
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models on one graph."""
        plt.figure(figsize=(10, 8))
        for model_name, metrics in self.results.items():
            fpr, tpr, _ = roc_curve(metrics['all_labels'], metrics['all_probs'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Across Models')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot 4 confusion matrices in a 2x2 grid."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = confusion_matrix(metrics['all_labels'], metrics['all_preds'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'], ax=axes[idx])
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        plt.tight_layout()
        plt.show()
    
    def plot_losses(self):
        """Plot train and validation losses for all models in a 2x2 grid."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            epochs = range(1, len(metrics['train_losses']) + 1)
            axes[idx].plot(epochs, metrics['train_losses'], label='Train Loss')
            axes[idx].plot(epochs, metrics['valid_losses'], label='Valid Loss', linestyle='--')
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].legend()
        plt.tight_layout()
        plt.show()
    
    def plot_accuracies(self):
        """Plot validation accuracies for all models in a 2x2 grid."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            epochs = range(1, len(metrics['valid_accuracies']) + 1)
            axes[idx].plot(epochs, metrics['valid_accuracies'], label='Valid Accuracy')
            axes[idx].set_title(f'{model_name}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Accuracy')
            axes[idx].legend()
        plt.tight_layout()
        plt.show()
    
    def plot_all(self):
        """Display all comparative plots and table."""
        print("\nModel Comparison Summary:")
        print(self.summary_table())
        self.plot_roc_curves()
        self.plot_confusion_matrices()
        self.plot_losses()
        self.plot_accuracies()