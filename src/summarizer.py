from transformers import pipeline
from textblob import TextBlob
import numpy as np

# Load summarization model
summarizer = pipeline("summarization", model="google/flan-t5-base")

def generate_summary(report):
    """
    Takes the 'report' dictionary and returns a concise summary +
    sentiment analysis results.
    """
    reviews = report.get("analysis", {}).get("review_text", {}).get("sample_reviews", [])
    if not reviews:
        return {"summary": "No reviews found.", "sentiment_summary": {}}

    # Combine reviews into one text block
    text = " ".join(reviews)
    text = text[:3000]  # Keep it within token limits

    # 🔥 Explicit summarization instruction (important!)
    input_text = "\n" + text

    # Generate concise summary
    raw_summary = summarizer(
        input_text,
        max_length=120,
        min_length=30,
        do_sample=False
    )
    summary_text = raw_summary[0]["summary_text"].strip()

    # --- Sentiment Analysis ---
    sentiments = []
    for r in reviews:
        blob = TextBlob(r)
        sentiments.append(blob.sentiment.polarity)

    sentiments = np.array(sentiments)
    pos = int((sentiments > 0.1).sum())
    neg = int((sentiments < -0.1).sum())
    neu = len(sentiments) - pos - neg

    overall = (
        "mostly positive" if pos > neg else
        "mostly negative" if neg > pos else
        "mixed"
    )

    sentiment_summary = {
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "overall_sentiment": overall
    }

    return {
        "summary": summary_text,
        "sentiment_summary": sentiment_summary
    }
