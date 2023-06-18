import logging

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

esp = 1e-5

class Model:
    """
    Thin wrapper on the SVM scikit model
    https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('max_iter', 1e4)

        [self.__setattr__(key, kwargs.get(key)) for key in kwargs]

        self.logger = logging.getLogger(__name__)
        self.model = make_pipeline(
            StandardScaler(),
            LinearSVC(tol=esp, max_iter=self.max_iter)
        )

    def train(self, X, y):
        """
        Train a Linear SVC model

        :param X: the training feature matrix
        :param y: the relavent target class vector

        y should be reshaped into a vector ie. `y.reshape(1, -1)`
        """

        self.logger.info("training up a new SVM")
        self.model.fit(X, y)
        self.logger.info("completed the SVM training")

    def predict(self, X): return self.model.predict(X)

