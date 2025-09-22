from transformers import pipeline

def modernbert_model():
    pipe = pipeline("sentiment-analysis", model="Modernbert_model")
    return pipe

