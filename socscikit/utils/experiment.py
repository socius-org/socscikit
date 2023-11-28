import wandb
from rich.progress import track
from torch.utils.data import TensorDataset


class Experiment:
    """
    Class to manage and run experiments.

    Parameters:
    - project (str): Name of the project for logging experiments.
    - config (dict): Configuration settings for the experiment. Must follow sweep configuration format provided by wandb.
      For further information on the format, visit https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
    - model: Model class to be used in the experiment (from socscikit.models).
    - train_data (torch.utils.data.TensorDataset): Training dataset.

    Example:
    ```python
    project_name = "my_project"
    config = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize', 
            'name': 'validation_loss'
        },
        'parameters': {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'values': [5, 10, 15]},
            'lr': {'max': 0.1, 'min': 0.0001}
        }
    }
    model = socscikit.models.Classification
    train_data = MyDataset(...)
    
    exp = Experiment(project_name, config, model, train_data)
    exp.run(count=5)
    ```

    """

    def __init__(self, project: str, config: dict, model, train_data: TensorDataset):
        self.project = project
        self.config = config
        self.model = model
        self.train_data = train_data

        self.sweep_id = wandb.sweep(sweep=self.config, project=self.project)

    def train(self, config=None):
        """
        Train the model with the specified configuration.

        Parameters:
        - config (dict, optional): Configuration settings for the experiment. If None, uses the default configuration.

        """
        with wandb.init(config=config):
            config = wandb.config

            # Define hidden layers based on config
            hidden_layers = [
                (
                    config.hidden_layer_size,
                    config.normalise,
                    config.activation,
                    config.dropout
                )
                for _ in range(config.num_hidden_layers)
            ]

            model = self.model(config.input_size, hidden_layers, config.output_size)

            train_loader, val_loader = model.build_train_valid_dataloader(
                self.train_data, config.batch_size
            )

            criterion = model.criterion
            optimizer = model.build_optimizer(
                optimizer=config.optimizer, lr=config.learning_rate
            )

            for epoch in track(range(config.epochs), description="Training Model..."):
                loss = model.train_epoch(train_loader, criterion, optimizer)
                val_loss, val_acc = model.validate(val_loader, criterion)

                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{config.epochs}], "
                        f"Training Loss: {loss.item():.4f}, "
                        f"Validation Loss: {val_loss:.4f}, "
                        f"Validation Accuracy: {val_acc:.4f}"
                    )
                    wandb.log({"loss": loss, "val_loss": val_loss, "val_acc": val_acc})

    def run(self, count: int):
        """
        Run the experiment for the specified number of times.

        Parameters:
        - count (int): Number of times to run the experiment.

        """
        wandb.agent(self.sweep_id, lambda: self.train(), count=count)
