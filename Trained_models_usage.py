from utils.predict_sentiment import PredictSentiment

predictor = PredictSentiment(save_dir="Trained_models", vocab_path="Trained_models/vocab.pt")

review = "I really like this film, it showed me many interesting things and therefore i will recommend it to others i case they have not yet seen it"

prediction = predictor.predict(review)

