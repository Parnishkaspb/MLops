import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from transformers import BertModel, BertTokenizer

EMBED_MODEL = "DeepPavlov/rubert-base-cased"

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'

class MyModel(nn.Module):
    def __init__(
        self,
        dataloader: DataLoader,
        input_dim: int,
        output_dim: int = 1,
        dropout_rate: float = 0.5,
    ):
        super(MyModel, self).__init__()
        self.dataloader = dataloader
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=384, kernel_size=3, stride=1, padding=2)
        self.maxpooling1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.last_linear = nn.Linear(384, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.to(device)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 2:  # Если x имеет размерность [batch_size, sequence_length]
            x = x.unsqueeze(1)  # Добавляем измерение каналов: [batch_size, 1, sequence_length]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, sequence_length]
        x = self.conv1d(x)
        x = self.maxpooling1d(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Возвращаем размерность [batch_size, sequence_length, features]
        x = self.last_linear(x)
        x = self.sigmoid(x)
        return x

    def fit(
        self,
        num_epoch: int = 10,
        lr: float = 1e-10,
    ):
        self.epoch_loss = []
        self.batch_loss = []
        self.f1_score = []
        self.all_labels = []
        self.all_probs = []
        criterion = nn.BCELoss()  # Бинарная кросс-энтропия
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epoch):
            self.train()
            epoch_losses = []
            for batch_X, batch_y in tqdm(self.dataloader, desc=f"Epoch [{epoch+1}/{num_epoch}]; learning state\t"):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss: torch.Tensor = criterion(outputs.squeeze(1), batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                self.batch_loss.append(loss.item())
            
            self.epoch_loss.append(sum(epoch_losses) / len(epoch_losses))
            
            # Вычисление метрик
            self.eval()
            with torch.no_grad():
                self.all_preds = []
                self.all_labels = []
                for batch_X, batch_y in tqdm(self.dataloader, desc=f"Epoch [{epoch+1}/{num_epoch}]; metric state\t"):
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self(batch_X).squeeze(1)
                    preds = (outputs > 0.5).float()
                    self.all_preds.extend(preds.squeeze(1).cpu().numpy())
                    self.all_labels.extend(batch_y.cpu().numpy())

                self.all_labels = np.array(self.all_labels, dtype=int)
                self.all_preds = np.array(self.all_preds, dtype=int)
                f1 = f1_score(self.all_labels, self.all_preds)
                self.f1_score.append(f1)
                print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}, F1 score: {f1:.5f}")
    
    def plot_metrics(self, X_test: torch.Tensor, y_test:torch.Tensor):
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.epoch_loss, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss per Epoch")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(self.batch_loss, label="Loss")
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.title("Loss per batch")
        plt.legend()

        # График F1-score
        plt.subplot(1, 3, 3)
        plt.plot(self.f1_score, label="F1-score", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("F1-score")
        plt.title("F1-score per Epoch")
        plt.legend()

        X_test = X_test.float()
        X_test = X_test.unsqueeze(1)
        X_test = X_test.to(device)
        self.eval()
        with torch.no_grad():
            preds = self(X_test)
        preds_squeezed = preds.squeeze(1).squeeze(1)
        result_pres = (preds_squeezed > 0.5).float()
        f1 = f1_score(y_test.numpy(), result_pres.cpu().numpy())
        print(f"F1 score on test tensors is: {f1*100:.2f}")

print('start init tokenizer')
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
print('start init embedding model')
embedding_model = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def check_ur_comment(comment: str, model: MyModel):
    my_comment_embedding = text_to_embedding(comment)
    my_comment_tensor = torch.tensor(my_comment_embedding, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds = model(my_comment_tensor.unsqueeze(0).to(device))
    preds_squeezed = preds.squeeze(1).squeeze(1)
    return round(preds_squeezed.item(), 2)

