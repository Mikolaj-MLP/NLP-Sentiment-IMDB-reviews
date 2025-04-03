from utils.predict_sentiment import PredictSentiment

predictor = PredictSentiment(save_dir="Trained_models", vocab_path="Trained_models/vocab.pt")

review = '''
I really liked this movie. Just kidding I lied.
'''

prediction = predictor.predict(review)

