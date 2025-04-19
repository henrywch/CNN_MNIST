# plot the score and loss
import matplotlib.pyplot as plt

colors_set = {'Kraftime': ('#E3E37D', '#968A62')}


def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    # Create epoch numbers based on stored metrics
    epochs = list(range(1, len(runner.train_loss) + 1))

    # Loss plot
    axes[0].plot(epochs, runner.train_loss, color=train_color, label="Train loss")
    axes[0].plot(epochs, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(loc='upper right')

    # Accuracy plot
    axes[1].plot(epochs, runner.train_scores, color=train_color, label="Train accuracy")
    axes[1].plot(epochs, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc='lower right')

    plt.tight_layout()