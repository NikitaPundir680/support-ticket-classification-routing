from google.colab import files
uploaded = files.upload()

df=pd.read_excel('gpt.xlsx')

df.head()

df['message_text'] = df['message_text'].fillna("").astype(str)
df['category'] = df['category'].fillna("Unknown").astype(str)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

le = LabelEncoder()
le.fit(train_df["category"])
train_df["label"] = le.transform(train_df["category"])
test_df["label"]  = le.transform(test_df["category"])

label_mapping = dict(zip(range(len(le.classes_)), le.classes_.tolist()))
print("Label mapping (int -> label):", label_mapping)