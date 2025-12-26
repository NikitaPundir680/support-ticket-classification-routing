# Support Ticket Classification & Routing

## Problem
Customer support teams receive a large volume of unstructured support tickets that must be categorized and routed to the correct team efficiently. Manual triaging is slow, inconsistent, and does not scale.

## Solution
This project compares two approaches for classifying customer support tickets:

- **Fine-tuned DistilBERT**
- **Zero-shot classification using BART-MNLI**

The predicted category is then mapped to a support team using a simple rule-based routing mechanism.

## System Overview
Ticket Text
→ Classifier (Fine-tuned BERT / Zero-shot)
→ Category
→ Routing Rule
→ Support Team


## Evaluation Summary

| Model              | Accuracy | F1 Score |
|--------------------|----------|----------|
| Fine-tuned BERT    | ~0.18    | ~0.18    |
| Zero-shot Model    | ~0.64    | ~0.64    |

Due to the limited size of labeled training data, the zero-shot model significantly outperformed the fine-tuned BERT model.

## Key Learnings
- Fine-tuning transformer models is not always effective with small datasets
- Zero-shot classification can outperform trained models in low-data scenarios
- Evaluation metrics should guide model and architecture decisions

## Limitations
- Small labeled dataset
- No production deployment
- Routing logic is rule-based and not learned
