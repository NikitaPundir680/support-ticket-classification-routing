MODEL_NAME = "distilbert-base-uncased"   # or "bert-base-uncased" / "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(examples["message_text"], truncation=True, padding=False, max_length=256)

train_ds = Dataset.from_pandas(train_df[["message_text","label"]])
test_ds  = Dataset.from_pandas(test_df[["message_text","label"]])

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds  = test_ds.map(tokenize_fn, batched=True)

columns = ['input_ids', 'attention_mask', 'label']
train_ds = train_ds.remove_columns([col for col in train_ds.column_names if col not in columns])
test_ds  = test_ds.remove_columns([col for col in test_ds.column_names if col not in columns])
train_ds.set_format(type='torch', columns=columns)
test_ds.set_format(type='torch', columns=columns)

num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

training_args = TrainingArguments(output_dir="./bert_ticket_classifier",
                                  eval_strategy="epoch",
                                  save_strategy="epoch",
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=32,
                                  num_train_epochs=3,
                                  weight_decay=0.01,
                                  logging_steps=50,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1",
                                  greater_is_better=True,
                                  fp16=torch.cuda.is_available()
                                  )

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy_mterics = evaluate.load("accuracy")
f1_metrics = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_mterics.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metrics.compute(predictions=preds, references=labels, average="weighted")["f1"]
    precision = precision_metric.compute(predictions=preds, references=labels, average="weighted")["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average="weighted")["recall"]
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

pred_output = trainer.predict(test_ds)
logits = pred_output.predictions
preds = np.argmax(logits, axis=-1)
labels = pred_output.label_ids

# Human-readable classification report
print(classification_report(labels, preds, zero_division=0))
# Confusion matrix (integers)
cm = confusion_matrix(labels, preds)
print("Confusion matrix:\n", cm)

model_dir = "./bert_ticket_classifier/best_model"
os.makedirs(model_dir, exist_ok=True)
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)
# Save label encoder classes for inference
import json
with open(os.path.join(model_dir, "label_mapping.json"), "w") as f:
    json.dump(label_mapping, f)
print("Saved model to", model_dir)