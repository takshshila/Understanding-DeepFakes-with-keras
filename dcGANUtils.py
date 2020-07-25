import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_data(one_hot=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], 784))/255.
    x_test = np.reshape(x_test, (x_test.shape[0], 784))/255.
    if one_hot:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)

def plot_ten_random_examples(plt, x, y, p=None):
    indices = np.random.choice(range(0, x.shape[0]), 10, replace=False)
    y = np.argmax(y, axis=1)
    if p is None:
        p = y
    plt.figure(figsize=(10, 5))
    for i, index in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(x[index].reshape((28, 28)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
        if y[index] == p[index]:
            col = 'g'
        else:
            col = 'r'
        plt.xlabel(str(p[index]), color=col)
    return plt

def load_subset(classes, x, y):
    """
    y should not be one hot encoded
    """
    x_subset = None
    for i, c in enumerate(classes):
        indices = np.squeeze(np.where(y == c))
        x_c = x[indices]
        if i == 0:
            x_subset = np.array(x_c)
        else:
            x_subset = np.concatenate([x_subset, x_c], axis=0)
    return x_subset


class SimpleTrainingPlot(tf.keras.callbacks.Callback):
    """
    Requires matplotlib.pyplot passed as argument plt
    when this callback is instantiated.
    Training metric for accuracy needs to be set as 'accuracy'
    and not 'acc'
    """

    def __init__(self, plt):
        super(SimpleTrainingPlot, self).__init__()

        self.fig = plt.figure(figsize=(9, 4))
        self.ax1 = plt.subplot(1, 2, 1)  # Losses
        self.ax2 = plt.subplot(1, 2, 2)  # Accuracies
        plt.ion()

    def plot(self, epoch=None):
        if epoch is not None:
            self.ax1.clear()
            self.ax2.clear()

            self.ax1.plot(range(epoch), self.losses, label='Train')
            self.ax1.plot(range(epoch), self.val_losses, label='Val')
            self.ax1.set_xlabel('Epochs')
            self.ax1.set_ylabel('Loss')
            self.ax1.legend()

            self.ax2.plot(range(epoch), self.accs, label='Train')
            self.ax2.plot(range(epoch), self.val_accs, label='Val')
            self.ax2.set_xlabel('Epochs')
            self.ax2.set_ylabel('Accuracy')
            self.ax2.legend()

            self.fig.canvas.draw()

    def on_train_begin(self, logs=None):
        self.val_accs = []
        self.accs = []
        self.val_losses = []
        self.losses = []

        self.fig.show()
        self.fig.canvas.draw()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.accs.append(logs.get('accuracy'))
            self.val_accs.append(logs.get('val_accuracy'))
            self.plot(epoch + 1)


class PlotEmbedding(tf.keras.callbacks.Callback):
    """
    Plot Embedding using the feature embedding generated from an
    embedding model passed in this callback's instance
    """

    def __init__(self, plt, embedding_model, x_test, y_test, use_tsne=True):
        super(PlotEmbedding, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = x_test
        self.y_test = y_test
        self.use_tsne = use_tsne
        self.fig = plt.figure()
        self.ax = plt.subplot(1, 1, 1)
        plt.ion()

    def plot(self, epoch=None):
        x_test_embeddings = self.embedding_model.predict(self.x_test)
        if self.use_tsne:
            out = TSNE(n_components=2).fit_transform(x_test_embeddings)
        else:
            out = PCA(n_components=2).fit_transform(x_test_embeddings)
        self.ax.clear()
        self.ax.scatter(out[:, 0], out[:, 1], c=self.y_test, cmap='seismic')
        self.fig.canvas.draw()

    def on_train_begin(self, logs=None):
        self.fig.show()
        self.fig.canvas.draw()
        self.plot()

    def on_epoch_end(self, epoch, logs=None):
        self.plot(epoch + 1)

def triplet_loss(dim, alpha=0.2):
    def loss(y_true, y_pred):
        # Assumes a shape of (batch_size, embedding_size*3)
        anchor, positive, negative = y_pred[:,:dim], y_pred[:,dim:2*dim], y_pred[:,2*dim:]
        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        loss = positive_dist - negative_dist + alpha
        return tf.maximum(0., loss)
    return loss


def plot_training_history(plt, history):
    history = history.history
    plt.figure(figsize=(12, 4))
    epochs = len(history['val_loss'])
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history['val_loss'], label='Val Loss')
    plt.plot(range(epochs), history['loss'], label='Train Loss')
    plt.xticks(list(range(epochs)))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history['val_accuracy'], label='Val Acc')
    plt.plot(range(epochs), history['accuracy'], label='Acc')
    plt.xticks(list(range(epochs)))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    return plt


class DynamicPlot:
    def __init__(self, plt, rows, cols, figsize):
        self.rows = rows
        self.cols = cols

        self.plt = plt

        self.fig = self.plt.figure(figsize=figsize)
        self.plt.ion()

    def draw_fig(self):
        self.fig.show()
        self.fig.canvas.draw()

    def start_of_epoch(self, epoch):
        self.ax = self.plt.subplot(self.rows, self.cols, 1 + epoch)

    def end_of_epoch(self, image, cmap, xlabel, ylabel):
        self.ax.imshow(image, cmap=cmap)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw_fig()