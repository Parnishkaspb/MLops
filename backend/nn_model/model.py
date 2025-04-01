import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

EMBED_MODEL = "DeepPavlov/rubert-base-cased"


class MyModel(nn.Module):
    def __init__(
        self,
        dataloader: DataLoader,
        input_dim: int,
        output_dim: int = 1,
        dropout_rate: float = 0.5,
    ):
        super(MyModel, self).__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.mps.is_available():
            self.device = 'mps'

        self.dataloader = dataloader
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=2,
        )
        self.maxpooling1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.last_linear = nn.Linear(384, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained(EMBED_MODEL)
        self.embedding_model = BertModel.from_pretrained(EMBED_MODEL)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 2:  # [batch_size, sequence_length]
            x = x.unsqueeze(1)  # [batch_size, 1, sequence_length]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, sequence_length]
        x = self.conv1d(x)
        x = self.maxpooling1d(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, features]
        x = self.last_linear(x)
        x = self.sigmoid(x)
        return x

    def check_ur_comment(self, comment: str) -> bool:
        my_comment_embedding = self.embedding(comment)
        my_comment_tensor = torch.tensor(
            my_comment_embedding,
            dtype=torch.float32,
        )

        self.eval()
        with torch.no_grad():
            preds = self(my_comment_tensor.unsqueeze(0).to(self.device))
        preds_squeezed: torch.Tensor = preds.squeeze(1).squeeze(1)
        return preds_squeezed.item() > 0.5
