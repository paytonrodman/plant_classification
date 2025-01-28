import numpy as np
import matplotlib.pyplot as plt

def plot_time(history, output_path):
    """
    A function to plot the accuracy and loss of a trained model over time.

    Args:
        pd.DataFrame history: DataFrame containing history of the model per epoch
        Path output_path: Location to save image

    Returns:
        None.
    """

    epochs = history['epoch'].tolist()

    plt.style.use("ggplot")
    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(epochs, history["accuracy"], label="train_acc")
    plt.plot(epochs, history["val_accuracy"], label="val_acc")
    plt.title("Accuracy", size=15)
    plt.xlabel("Epoch no.")
    plt.legend()

    plt.subplot(122)
    plt.plot(epochs, history["loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.title("Loss", size=15)
    plt.xlabel("Epoch no.")
    plt.legend()

    plt.savefig(output_path / 'performance_over_time.png', dpi=400, bbox_inches='tight')

def plot_confusion(model, test_gen, output_path):
    """
    A function to plot the confusion matrix for a given model based on the
    true and predicted classes

    Args:
        keras.model model: CNN model to assess
        generator test_gen: Test data generator
        Path output_path: Location to save image

    Returns:
        None.
    """
    import sklearn.metrics as metrics
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix
    import math

    test_steps_per_epoch = math.ceil(test_gen.samples / test_gen.batch_size)
    predictions = model.predict(test_gen, steps=test_steps_per_epoch)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    plt.style.use("ggplot")
    plt.figure(figsize=(5, 5))

    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.grid(False)
    plt.savefig(output_path / 'confusion_matrix.png', dpi=400, bbox_inches='tight')
