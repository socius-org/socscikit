from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from numpy import ndarray


class SVM():
    """
    Support Vector Machine (SVM) classifier.
    
    Parameters:
    - X: The input data for training/testing.
    - y: The labels for the input data.
    - kernel: The kernel function to be used for classification. Defaults to 'linear'.
    - test_size: The proportion of the dataset to include in the test split. Defaults to 0.2.
    - random_state: Controls the shuffling applied to the data before applying the split. Defaults to 42.
    
    Example:
    ```
    X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
    y = np.array([0, 1, 0, 1, 0, 1])
    model = SVM(X, y, kernel='linear')
    model.fit()
    ```
    """
    
    def __init__(self, X, y, kernel='linear', test_size=0.2, random_state=42):
        """
        Initializes the SVM classifier with the given parameters and splits the data into training and testing sets.

        Args:
            X (array-like): The input data for training/testing.
            y (array-like): The labels for the input data.
            kernel (str, optional): The kernel function to be used for classification. Defaults to 'linear'.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.kernel = kernel
        self.clf = SVC(kernel=self.kernel)
    
    def multi_class(self):
        """
        Changes the decision function shape of the SVM classifier to 'ovo' for multi-class classification.
        """
        self.clf = SVC(decision_function_shape='ovo')
        
    def fit(self):
        """
        Fits the SVM classifier to the training data.
        """
        self.clf.fit(self.X_train, self.y_train)
        
    def predict(self) -> ndarray:
        """
        Makes predictions on the test data using the fitted SVM classifier.

        Returns:
            array-like: The predicted labels for the test data.
        """
        return self.clf.predict(self.X_test)
    
    def evaluate(self)-> str:
        """
        Evaluates the performance of the SVM classifier by comparing the predicted labels to the true labels.

        Prints the accuracy of the classifier.
        """
        predictions = self.predict()
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy:", accuracy)