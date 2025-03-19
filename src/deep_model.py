import torch
import torch.nn as nn
from sklearn.metrics import f1_score

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation_func):
        super(FCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))

        # Select the activation function based on the activation function name
        if activation_func == "relu":
            self.activation = nn.ReLU()
        elif activation_func == "tanh":
            self.activation = nn.Tanh()
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x


class CNN(nn.Module):
    def __init__(
        self, n_features, output_dim, n_layers, n_filters, kernel_size, activation_func
    ):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList()

        if activation_func == "relu":
            self.activation = nn.ReLU()
        elif activation_func == "tanh":
            self.activation = nn.Tanh()
        elif activation_func == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

        self.convs.append(
            nn.Conv1d(
                1, n_filters[0], kernel_size=kernel_size, padding=kernel_size // 2
            )
        )
        for i in range(1, n_layers):
            self.convs.append(
                nn.Conv1d(
                    n_filters[i - 1],
                    n_filters[i],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )

        self.fc = nn.Linear(n_filters[-1] * n_features, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = self.activation(conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_evaluate(
    model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs=300
):
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

    best_f1 = 0
    no_improve_epochs = 0
    early_stopping_rounds = 50

    # Train
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            predictions_prob = torch.sigmoid(val_outputs)
            y_pred = (predictions_prob.cpu().numpy() >= 0.5).astype(int)
            f1 = f1_score(y_val.cpu().numpy(), y_pred)

        if f1 > best_f1:
            best_f1 = f1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_rounds:
            # print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return best_f1
