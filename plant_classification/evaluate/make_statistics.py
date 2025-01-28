import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import math
import numpy as np

def get_basic_stats(model, valid_gen, batch_size):
    """
    A function to generate the basic test accuracy and loss statistics for a given model.

    Args:
        keras.model model: CNN model to assess
        valid_gen: Validation data generator
        int batch_size:

    Returns:
        array score: Array of test assessment values
    """
    score = model.evaluate(valid_gen, batch_size=batch_size)

    return score

def get_report(model, test_gen):
    """
    A function to generate a prediction report for a given model.

    Args:
        keras.model model: CNN model to assess
        test_gen: Test data generator

    Returns:
        str report: String report, formatted as table
    """
    test_steps_per_epoch = math.ceil(test_gen.samples / test_gen.batch_size)
    predictions = model.predict(test_gen, steps=test_steps_per_epoch)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    report = metrics.classification_report(true_classes,
                                           predicted_classes,
                                           target_names=class_labels)
    return report
