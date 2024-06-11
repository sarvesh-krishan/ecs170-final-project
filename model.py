import torch
import time
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from processing_data import glove


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_hidden_state = output[:, -1, :]
        logits = self.fc(last_hidden_state)
        return logits
    

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply hidden_size by 2 for bidirectional LSTM

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        last_hidden_state = torch.cat((output[:, -1, :self.hidden_size], output[:, 0, self.hidden_size:]), dim=1)
        logits = self.fc(last_hidden_state)
        return logits


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        last_hidden_state = output[:, -1, :]
        logits = self.fc(last_hidden_state)
        return logits


class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience # num epochs to wait after last time val_loss improved
        self.delta = delta # min change to qualify as an improvement
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False # early stop flag

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta: # improvement
            self.best_loss = val_loss
            self.counter = 0
        else: # no improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train(model, num_epochs, train_loader, val_loader, optimizer, criterion):
    #start runtime for training model
    start_time_train = time.time()

    train_losses = []
    val_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    early_stopping = EarlyStopping(patience = 5, delta = 0.01)

    for epoch in range(num_epochs):
        # Early stopping
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Training
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(sequences)

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            _, predictions = torch.max(logits, 1)
            total_correct += torch.sum(predictions == labels).item()
            total_samples += labels.size(0)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Compute average loss and accuracy
        average_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples

        # compute validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for val_sequences, val_labels in val_loader:
                val_sequences, val_labels = val_sequences.to(device), val_labels.to(device)
                val_logits = model(val_sequences)
                val_loss += criterion(val_logits, val_labels).item()
                _, val_predictions = torch.max(val_logits, 1)
                val_correct += torch.sum(val_predictions == val_labels).item()
                val_samples += val_labels.size(0)

        average_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_samples

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {average_loss:.4f} | Train Accuracy: {accuracy * 100:.2f}%")
        train_losses.append(average_loss)
        print(f"  Val Loss: {average_val_loss:.4f} | Val Accuracy: {val_accuracy * 100:.2f}%")
        val_losses.append(average_val_loss)

        early_stopping(average_val_loss)

    #end runtime for training model
    end_time_train = time.time()

    #calculated runtime for training model
    runtime_train = end_time_train - start_time_train
    print(" Model Training Time:", runtime_train, "seconds")

    return train_losses, val_losses


def evaluate(model, test_loader, test_dataset, criterion):
    #start runtime for evaluating model
    start_time_eval = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluation
    model.eval()
    eval_loss = 0.0
    eval_predictions = []
    eval_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            eval_loss += loss.item() * sequences.size(0)

            _, predicted = torch.max(outputs, dim=1)
            eval_predictions.extend(predicted.tolist())
            eval_labels.extend(labels.tolist())

    eval_loss /= len(test_dataset)
    eval_accuracy = accuracy_score(eval_labels, eval_predictions)
    eval_precision = precision_score(eval_labels, eval_predictions)
    eval_recall = recall_score(eval_labels, eval_predictions)
    eval_f1 = f1_score(eval_labels, eval_predictions)

    print(f"Eval Loss: {eval_loss:.4f} | Accuracy: {eval_accuracy:.4f} | "
          f"Precision: {eval_precision:.4f} | Recall: {eval_recall:.4f} | F1 Score: {eval_f1:.4f}")
    
    #end runtime for evaluating model
    end_time_eval = time.time()

    #calculated runtime for training model
    runtime_eval = end_time_eval - start_time_eval
    print("Model Evaluation Time:", runtime_eval, "seconds")
    # Confusion Matrix
    cm = confusion_matrix(eval_labels, eval_predictions)
    print("Confusion Matrix:")
    print(cm)
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(eval_labels, eval_predictions) 
    roc_auc = auc(fpr, tpr)
    print("Area Under the ROC Curve:")
    print(roc_auc)
    return [fpr, tpr, roc_auc]
    