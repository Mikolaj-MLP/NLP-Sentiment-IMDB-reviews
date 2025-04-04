import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, epochs=10, lr=0.001, patience=3, save_path=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.save_path = save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.train_losses = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.test_accuracy = None
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
        
        self.best_valid_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None
        
        self.temperature = 1.0  # Initialize temperature for calibration

    def train_epoch(self):
        self.model.train()
        total_train_loss = 0
        train_bar = tqdm(self.train_loader, desc=f"Epoch {len(self.train_losses)+1}/{self.epochs} [Train]", leave=False)
        for batch_x, batch_y in train_bar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            if "Attention" in self.model.__class__.__name__:
                outputs = self.model(batch_x, device=self.device)
            else:
                outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix({'batch_loss': loss.item()})
        avg_train_loss = total_train_loss / len(self.train_loader)
        self.train_losses.append(avg_train_loss)
        
    def evaluate_epoch(self):
        self.model.eval()
        total_valid_loss = 0
        correct = 0
        valid_bar = tqdm(self.valid_loader, desc=f"Epoch {len(self.valid_losses)+1}/{self.epochs} [Valid]", leave=False)
        with torch.no_grad():
            for batch_x, batch_y in valid_bar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if "Attention" in self.model.__class__.__name__:
                    outputs = self.model(batch_x, device=self.device)
                else:
                    outputs = self.model(batch_x)
                total_valid_loss += self.criterion(outputs, batch_y).item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
        avg_valid_loss = total_valid_loss / len(self.valid_loader)
        valid_accuracy = correct / len(self.valid_loader.dataset)
        self.valid_losses.append(avg_valid_loss)
        self.valid_accuracies.append(valid_accuracy)
        return avg_valid_loss, valid_accuracy
        
    def calibrate(self):
        """
        A naive calibration procedure that computes the average positive-class probability
        on the validation set and adjusts the temperature scaling factor within ±5% of 1.0.
        """
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch_x, batch_y in self.valid_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if "Attention" in self.model.__class__.__name__:
                    outputs = self.model(batch_x, device=self.device)
                else:
                    outputs = self.model(batch_x)
                # Get probability of positive class for the batch
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.append(probs)
        avg_prob = torch.cat(all_probs).mean().item()
        
        # Adjust temperature so that the scaling does not change more than 5%
        if avg_prob > 0.55:
            # Overconfident toward positive; increase temperature (max +5%)
            delta = min(avg_prob - 0.5, 0.05)
            self.temperature = 1.0 + delta
        elif avg_prob < 0.45:
            # Underconfident; decrease temperature (max -5%)
            delta = min(0.5 - avg_prob, 0.05)
            self.temperature = 1.0 - delta
        else:
            self.temperature = 1.0

        print(f"Calibration complete. Average validation positive prob: {avg_prob:.4f}, temperature set to {self.temperature:.4f}")
        
    def evaluate_test(self, apply_calibration=True):
        """
        Evaluate the test set. If apply_calibration is True, adjust the model's logits using the
        calibrated temperature before applying softmax.
        """
        self.model.eval()
        test_correct = 0
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        test_bar = tqdm(self.test_loader, desc="Test", leave=False)
        with torch.no_grad():
            for batch_x, batch_y in test_bar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                if "Attention" in self.model.__class__.__name__:
                    outputs = self.model(batch_x, device=self.device)
                else:
                    outputs = self.model(batch_x)
                # Optionally apply calibration by dividing the logits by the temperature
                if apply_calibration:
                    outputs = outputs / self.temperature
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                test_correct += (preds == batch_y).sum().item()
                self.all_preds.extend(preds.cpu().numpy())
                self.all_labels.extend(batch_y.cpu().numpy())
                self.all_probs.extend(probs.cpu().numpy())
        self.test_accuracy = test_correct / len(self.test_loader.dataset)
        
    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            avg_valid_loss, valid_accuracy = self.evaluate_epoch()
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train Loss: {self.train_losses[-1]:.4f}")
            print(f"Valid Loss: {avg_valid_loss:.4f}")
            print(f"Valid Accuracy: {valid_accuracy:.4f}")
            
            if avg_valid_loss < self.best_valid_loss:
                self.best_valid_loss = avg_valid_loss
                self.epochs_no_improve = 0
                self.best_model_state = self.model.state_dict()
                if self.save_path:
                    torch.save(self.best_model_state, self.save_path)
                    print(f"Saved best model to {self.save_path}")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve}/{self.patience} epochs")
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        # Calibrate the model on the validation set (without overreaching ±5%)
        self.calibrate()
        
        # Evaluate on the test set with calibration applied
        self.evaluate_test(apply_calibration=True)
        print(f"Test Accuracy (with calibration): {self.test_accuracy:.4f}")
        
    def get_metrics(self):
        return (self.train_losses, self.valid_losses, self.valid_accuracies, 
                self.all_labels, self.all_preds, self.all_probs)
