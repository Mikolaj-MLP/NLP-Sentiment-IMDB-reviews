import torch
import os
from utils.data_preprocessor import DataPreprocessor
from Models.LSTM import UnidirectionalLSTM
from Models.BiLSTM import BidirectionalLSTM
from Models.AttentionBasedModel import AttentionBasedModel
from Models.Text_cnn import TextCNN

class PredictSentiment:
    def __init__(self, save_dir="Trained_models", vocab_path="Trained_models/vocab.pt"):
        """Initialize with model save directory and vocab path."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(remove_stopwords=True, use_stemming=True)
        self.preprocessor.word_to_idx = torch.load(vocab_path)  
        
        self.models = {
            "Unidirectional LSTM": UnidirectionalLSTM(vocab_size=10000, embedding_dim=100, hidden_dim=128, fc1_neurons=64, output_dim=2),
            "Bidirectional LSTM": BidirectionalLSTM(vocab_size=10000, embedding_dim=100, hidden_dim=128, fc1_neurons=128, fc2_neurons=64, output_dim=2),
            "Attention-Based Model": AttentionBasedModel(vocab_size=10000, embedding_dim=100, num_heads=4, num_layers=2, hidden_dim1=256, hidden_dim2=128, output_dim=2),
            "TextCNN": TextCNN(vocab_size=10000, embedding_dim=100, filter_sizes=[2, 3, 4, 5], num_filters=128, fc1_neurons=256, fc2_neurons=128, output_dim=2)
        }
        
        self.save_paths = {
            "Unidirectional LSTM": os.path.join(save_dir, "uni_lstm.pt"),
            "Bidirectional LSTM": os.path.join(save_dir, "bi_lstm.pt"),
            "Attention-Based Model": os.path.join(save_dir, "attn_model.pt"),
            "TextCNN": os.path.join(save_dir, "text_cnn.pt")
        }
        
        self.loaded_models = {}
        for model_name, model in self.models.items():
            # Load the state dict with map_location to CPU if no CUDA
            state_dict = torch.load(self.save_paths[model_name], map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)  # Move model to appropriate device (CPU or GPU)
            model.eval()
            self.loaded_models[model_name] = model

    def predict(self, review_text):
        """Process review and predict sentiment with all models."""
        tokens = self.preprocessor.process_text(review_text)
        
        vector = self.preprocessor.vectorize([tokens], max_length=300)
        input_tensor = torch.tensor(vector, dtype=torch.long).to(self.device)
        probs = []
        predictions = {}
        with torch.no_grad():
            for model_name, model in self.loaded_models.items():
                if model_name == "Attention-Based Model":
                    output = model(input_tensor, device=self.device)
                else:
                    output = model(input_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()
                pred = torch.argmax(output, dim=1).item()
                sentiment = "Positive" if pred == 1 else "Negative"
                predictions[model_name] = (sentiment, prob)
                print(f"{model_name}: {sentiment} (P(1): {prob:.4f})")
                probs.append(prob)
        
        overall_avg = sum(probs) / len(probs)
        overall_sentiment = "Positive" if overall_avg > 0.5 else "Negative"
        print(f"\nThe review has a {overall_sentiment} sentiment\n")
        
        return predictions
