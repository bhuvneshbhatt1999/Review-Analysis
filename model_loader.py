from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

def load_sentiment_model(device=None):
    """
    Loads the RoBERTa sentiment model and tokenizer, moves model to the specified device.
    Returns: tokenizer, model, device (for later use)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Could not load model or tokenizer: {e}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict_sentiment(texts, tokenizer, model, device):
    """
    Predict sentiment for a list of texts or a single string.
    Returns: list of labels ("Positive", "Neutral", "Negative") or a single label.
    """
    if isinstance(texts, str):
        texts = [texts]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().tolist()
    id2label = model.config.id2label
    labels = [id2label.get(pred, str(pred)).capitalize() for pred in preds]
    return labels if len(labels) > 1 else labels[0]
