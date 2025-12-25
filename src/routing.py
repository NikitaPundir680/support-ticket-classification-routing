routing ={
    "Order Issue": "Order Support Team",
    "Delivery Delay": "Logistics Team",
    "Return Issue": "Returns Team",
    "Technical Issue": "Tech Support Team",
    "Fraud/Security": "Fraud Prevention Team"
}

df['predicted_routing'] = df['predicted_category'].map(routing)

df[['message_text','predicted_category','predicted_routing']].head()

