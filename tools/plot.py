import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class PLOT:
    def __init__(self):
        return

    def plot_confusion_matrix(self, cm, title='Confusion matrix',
                              xticks=None, yticks=None,
                              cmap=plt.cm.Blues, norm=True):
        if norm:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(cm[0]))
        if xticks is None:
            xticks = tick_marks
        if yticks is None:
            yticks = tick_marks
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

p = PLOT()
cm = np.array([[0.8, 0.0, 0.2],
              [0.1, 0.8, 0.1],
              [0.0, 0.9, 0.1]])
p.plot_confusion_matrix(cm)
