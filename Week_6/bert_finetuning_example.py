import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# 1. Setup Mock Dataset (Replace this with your actual data loading logic)
texts = [
    "I absolutely love this product, it works perfectly!",
    "This is the worst purchase I have ever made.",
    "Great customer service and fast shipping.",
    "It broke on the first day. Completely useless.",
    "Extremely satisfied with the overall quality.",
    "Do not waste your money on this garbage."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Split into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 2. Define Custom PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize inputs and generate attention masks
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. Initialize Tokenizer, Model, and DataLoaders
MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# 4. Optimizer and Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 5. Training Loop
EPOCHS = 3
print(f"Starting training on device: {device}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

# 6. Evaluation Loop
model.eval()
correct_predictions = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        correct_predictions += torch.sum(preds == labels)

accuracy = correct_predictions.double() / len(val_dataset)
print(f"Validation Accuracy: {accuracy.item() * 100:.2f}%")

# 7. Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_bert_sentiment")
tokenizer.save_pretrained("./fine_tuned_bert_sentiment")
print("Model saved successfully!")
