import re

EMOTIONS = ["Anger","Fear","Joy","Sadness","Surprise"]

def preprocess_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text
