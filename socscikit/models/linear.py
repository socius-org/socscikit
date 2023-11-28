from .core import *

class Classification(nn.Module):
    """
    Neural network model designed for classification tasks.

    Parameters:
    - input_size (int): Number of features in the input data.
    - hidden_layers (list): List of tuples specifying hidden layers. Each tuple contains:
        - size (int): Number of neurons in the layer.
        - normalize (bool): Whether to apply batch normalization.
        - activation (str): Activation function (options: 'ReLU', 'Sigmoid', 'Tanh').
        - dropout (float): Dropout probability (None if not applied).
    - output_size (int): Number of output classes.

    Example:
    ```
    input_size = X_train.shape[1]
    output_size = 10
    hidden_layers = [
        (128, False, "ReLU", None),
        (1024, True, "ReLU", 0.25),
        (128, False, "ReLU", None)
    ]
    model = Classification(input_size, hidden_layers, output_size)
    model.fit(train_data, optimizer='adam', lr=0.001, batch_size=32, epochs=50)
    ```

    Note:
    The `fit` method is used to train the model on the provided TensorDataset.

    """

    def __init__(self, input_size: int, hidden_layers: list, output_size: int):
        super(Classification, self).__init__()

        self.input_layer = nn.Flatten()

        self.hidden_layers = nn.ModuleList()

        for i, (size, normalize, activation, dropout) in enumerate(hidden_layers):
            layers = []
            layers.append(
                nn.Linear(input_size if i == 0 else hidden_layers[i - 1][0], size)
            )

            if normalize is True:
                layers.append(nn.BatchNorm1d(size))
            layers.append(self.build_activation(activation))

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

            layer = nn.Sequential(*layers)
            self.hidden_layers.add_module(f"layer_{i}", layer)

        self.output_size = output_size
        if self.output_size == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.output_layer = nn.Linear(hidden_layers[-1][0], self.output_size)

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass of the neural network.

        Parameters:
        - inputs (torch.Tensor): Input data.

        Returns:
        - torch.Tensor: Output predictions.

        """
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def build_activation(self, activation: str):
        """
        Build activation function module based on the given activation name.

        Parameters:
        - activation (str): Name of the activation function.

        Returns:
        - nn.Module: Activation function module.

        """
        if activation == "ReLU":
            return nn.ReLU()
        elif activation == "Sigmoid":
            return nn.Sigmoid()
        elif activation == "Tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def build_optimizer(self, optimizer: str, lr: float):
        """
        Build optimizer module based on the given optimizer name.

        Parameters:
        - optimizer (str): Name of the optimizer.
        - lr (float): Learning rate.

        Returns:
        - torch.optim.Optimizer: Optimizer module.

        """
        if optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "adam":
            return optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    def build_train_valid_dataloader(self, train_data: TensorDataset, batch_size: int):
        """
        Split the training data into training and validation sets and create DataLoader instances.

        Parameters:
        - train_data (torch.utils.data.TensorDataset): Training dataset.
        - batch_size (int): Batch size for DataLoader.

        Returns:
        - tuple: (train_loader, val_loader) DataLoader instances for training and validation sets.

        """
        train_ratio = 0.9
        train_size = int(train_ratio * len(train_data))
        val_size = len(train_data) - train_size

        train_data, val_data = random_split(train_data, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Train the model for one epoch.

        Parameters:
        - train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        - criterion (torch.nn.modules.loss._Loss): Loss function.
        - optimizer (torch.optim.Optimizer): Optimizer.

        Returns:
        - torch.Tensor: Loss value.

        """
        self.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            y_pred = self(inputs)
            if self.output_size == 1:
                y_pred = y_pred.squeeze()
                labels = labels.float()
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
        return loss

    def validate(self, val_loader: DataLoader, criterion: nn.modules.loss._Loss):
        """
        Validate the model on the validation set.

        Parameters:
        - val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        - criterion (torch.nn.modules.loss._Loss): Loss function.

        Returns:
        - tuple: (average validation loss, validation accuracy).

        """
        val_loss, correct_val, total_val = float(0), int(0), int(0)

        self.eval() #Set the model to validation (evaluation mode)

        with torch.no_grad():
            for inputs, labels in val_loader:
                val_pred = self(inputs)
                val_labels = labels
                if self.output_size == 1:
                    val_pred = val_pred.squeeze()
                    val_labels = val_labels.float()
                    predictions = torch.round(torch.sigmoid(val_pred))
                else:
                    predictions = torch.argmax(val_pred, dim=1)

                val_loss += criterion(val_pred, val_labels).item()
                total_val += labels.size(0)
                correct_val += torch.sum(torch.eq(predictions, val_labels)).item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        return avg_val_loss, val_acc

    def fit(
        self,
        train_data: TensorDataset,
        optimizer: str,
        batch_size: int,
        lr: float,
        epochs: int,
    ):
        """
        Train the model on the given training data.

        Parameters:
        - train_data (torch.utils.data.TensorDataset): Training dataset.
        - optimizer (str): Name of the optimizer.
        - batch_size (int): Batch size for DataLoader.
        - lr (float): Learning rate.
        - epochs (int): Number of training epochs.

        """
        train_loader, val_loader = self.build_train_valid_dataloader(
            train_data, batch_size
        )

        criterion = self.criterion
        optimizer = self.build_optimizer(optimizer=optimizer, lr=lr)

        for epoch in track(range(epochs), description="Training Model..."):
            loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(val_loader, criterion)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Training Loss: {loss.item():.4f}, "
                    f"Validation Loss: {val_loss:.4f}, "
                    f"Validation Accuracy: {val_acc:.4f}"
                )