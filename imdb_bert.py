import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
# from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes=2, dropout_rate=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # 4 Fully Connected Layers
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

        # Activation functions
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.pooler_output

        # Apply dropout
        x = self.dropout(pooled_output)

        # Pass through 4 fully connected layers
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)

        return x


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, dim=1)
        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)

    return avg_loss, accuracy, predictions, true_labels


def predict_sentiment(model, tokenizer, text, device, max_length=512):
    """Function to predict sentiment of a single text"""
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs, dim=1)
        confidence = torch.softmax(outputs, dim=1)

    sentiment = "Positive" if prediction.item() == 1 else "Negative"
    confidence_score = confidence[0][prediction.item()].item()

    return sentiment, confidence_score


# Example usage for prediction
def test_predictions():
    # Load the trained model
    model = BERTClassifier('bert-base-uncased', num_classes=2)
    model.load_state_dict(torch.load('best_bert_classifier.pth'))
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Test examples
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible movie, waste of time. Very disappointed.",
        "The movie was okay, nothing special but not bad either."
    ]

    print("\nTesting predictions:")
    print("-" * 40)

    for text in test_texts:
        sentiment, confidence = predict_sentiment(model, tokenizer, text, device)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
        print("-" * 40)

    # Uncomment the line below to test predictions after training
    # test_predictions()


def predict_batches(clf, samples):
    all_preds = []
    len_test = len(samples)
    # print(len_test)
    for batch in samples:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        # print(len(b_input_ids))
        # b_labels = batch[2].to(device)
        # print(b_labels)
        with torch.no_grad():
            out = clf(b_input_ids.cuda(), b_input_mask.cuda())
        all_preds.append(out.cpu().numpy())
        # if idx == 2:
        #     break
    return  all_preds