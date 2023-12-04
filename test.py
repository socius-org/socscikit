from socscikit.models.linear import Classification 
from torch.utils.data import TensorDataset
import torch
from sklearn.datasets import load_digits
digits = load_digits()
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    
    model = Classification(
    input_size = 64, 
    hidden_layers = [
        (128, False, "ReLU", None),
        (512, True,  "Tanh", 0.2),
        (128, True, "ReLU", None)
    ],
    output_size = 10
    )
    
    print(model)
    
    model.fit(
        train_data=train_dataset,
        epochs=100, 
        lr=0.01,
        optimizer="adam",
        batch_size=32)
