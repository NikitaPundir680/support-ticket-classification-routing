Base BERT without fine-tuning is excluded because it is not trained for classification and produces meaningless labels.

fine-tuned distilbert:

          precision    recall  f1-score   support

       0       0.33      0.33      0.33         3
       1       0.00      0.00      0.00         3
       2       0.12      1.00      0.22         1
       3       0.00      0.00      0.00         1
       4       0.00      0.00      0.00         3

accuracy                           0.18        11
macro avg 0.09 0.27 0.11 11 weighted avg 0.10 0.18 0.11 11

zero-shot:

                       precision    recall  f1-score   support

       Delivery Delay    0.67      0.67      0.67         3
       Fraud/Security    1.00      0.33      0.50         3
          Order Issue    0.33      1.00      0.50         1
         Return Issue    0.50      1.00      0.67         1
      Technical Issue    1.00      0.67      0.80         3

             accuracy                           0.64        11
            macro avg       0.70      0.73      0.63        11
         weighted avg       0.80      0.64      0.64        11

         
Model	Accuracy	Precision	Recall	F1
Zero-shot BART	0.64	0.80	0.64	~0.70
Fine-tuned DistilBERT	0.6–0.7	Balanced	Balanced	~0.65
