import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib


def predict_top5_categories(text, model, tokenizer, label_encoder, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(
            probabilities,
            k=5,
            dim=-1
        )

    predictions = []

    for prob, idx in zip(top_probs[0], top_indices[0]):

        category = label_encoder.inverse_transform([idx.item()])[0]

        predictions.append({
            "Category": category,
            "Confidence": prob.item()
        })

    return predictions


# =====================================================
# LOAD MODEL
# =====================================================

model_path = "./news_classifier_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load Label Encoder
le = joblib.load("label_encoder.pkl")

# =====================================================
# SAMPLE TEXT
# =====================================================

sample_text = """
Afghan Taliban launch strikes on border with Pakistan as tensions escalate
"""
# =====================================================
# PREDICT
# =====================================================

predictions = predict_top5_categories(
    sample_text,
    model,
    tokenizer,
    le
)

print(f"\nNews:\n{sample_text}\n")

print("Top 5 Predictions")
print("-" * 50)

for i, pred in enumerate(predictions, start=1):

    print(
        f"{i}. {pred['Category']:<20} "
        f"{pred['Confidence']*100:.2f}%"
    )