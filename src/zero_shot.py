classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = ["Order Issue","Delivery Delay","Return Issue","Technical Issue","Fraud/Security"]
def predict_category(text):
  if not text:
    return 'No Text'
  out = classifier(text, candidate_labels = labels)
  return out['labels'][0]

df["predicted_category"] = df["message_text"].fillna('').apply(predict_category)