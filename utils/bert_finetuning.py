import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoConfig

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.encodings = tokenizer(data['text'], truncation=True, padding=True,
                                 max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(data['label'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Load your dataset
dataset = load_dataset("shubham212/new_med_27dec") # same as Classification_bert.json
id2label={0: 'urgent care',1: 'scheduled operations',2: 'emergency',3: 'routine care'}
label2id = {v: k for k, v in id2label.items()}
split = dataset['train'].train_test_split(test_size=0.4, seed=42)

# Initialize tokenizer and model
model_name = "answerdotai/ModernBERT-base"
config = AutoConfig.from_pretrained(model_name, id2label=id2label, label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name,config=config)

# Create datasets
train_dataset = TextDataset(split['train'], tokenizer)
test_dataset = TextDataset(split['test'], tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


# Training loop
num_epochs = 7

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    avg_train_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Average training loss: {avg_train_loss:.4f}")

# Final test set evaluation
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        test_loss += outputs.loss.item()

        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)


test_accuracy = correct / total
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")


# Save the model
model.save_pretrained('modernbert_med')
tokenizer.save_pretrained('modernbert_med')