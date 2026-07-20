import copy
import numpy as np

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Maintains a deep copy of the model's best state dictionary.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = False):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Default: 10
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 1e-4
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_state = None
        self.best_epoch = 0

    def __call__(self, val_loss: float, model, epoch: int):
        """
        Call this method at the end of each epoch to evaluate early stopping criteria.
        
        Args:
            val_loss (float): The current validation loss.
            model: The PyTorch model being trained.
            epoch (int): The current epoch number.
        """
        if self.best_loss - val_loss > self.min_delta:
            if self.verbose:
                print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
