"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import torch
import matplotlib.pyplot as plt


# Plot loss curves of a model
def plot_loss_curves(results, model_name, train_time):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss", color='#F0E442')
    plt.plot(epochs, test_loss, label="test_loss", color='#0072B2')
    plt.title(f"{model_name} Loss, trained in {train_time:.3f} seconds", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.tick_params('both', which='major', labelsize=14)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy", color='#F0E442')
    plt.plot(epochs, test_accuracy, label="test_accuracy", color='#0072B2')
    plt.title(f"{model_name} Accuracy, trained in {train_time:.3f} seconds", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.tick_params('both', which='major', labelsize=14)
    plt.legend()

    plt.savefig(f'images/{model_name}_loss_curve')


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)