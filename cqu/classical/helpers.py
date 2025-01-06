import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class FraudDetectionNN(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),  # Input Layer
            nn.ReLU(),  # Activation
            nn.BatchNorm1d(64),  # Batch Normalization
            nn.Dropout(0.3),  # Dropout Layer to prevent overfitting
            nn.Linear(64, 32),  # Hidden Layer 1
            nn.ReLU(),  # Activation
            nn.BatchNorm1d(32),  # Batch Normalization
            nn.Dropout(0.2),  # Dropout Layer
            nn.Linear(32, 16),  # Hidden Layer 2
            nn.ReLU(),  # Activation
            nn.BatchNorm1d(16),  # Batch Normalization
            nn.Linear(16, 1),  # Output Layer
            nn.Sigmoid(),  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.layers(x)


class CustomVotingClassifier:
    def __init__(self, models, weights=None, threshold=0.5, epochs=50):
        """
        Custom voting classifier that handles both sklearn and custom models.

        Args:
            models (dict): Dictionary of model names and their instances
            weights (dict, optional): Dictionary of model names and their voting weights
        """
        self.models = models
        self.weights = (
            weights if weights is not None else {name: 1 for name in models.keys()}
        )
        self.threshold = threshold
        self.epochs = epochs
        self.fitted_models = {}

    def fit(self, X, y, class_weights=None):
        """
        Fit all models in the ensemble.

        Args:
            X: Training features
            y: Target values
        """
        for name, model in self.models.items():
            if hasattr(model, "fit"):
                if class_weights is not None and hasattr(model, "class_weight"):
                    model.set_params(class_weight=class_weights)

                self.fitted_models[name] = model.fit(X, y)
            else:
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
                y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

                lossfn = nn.BCELoss()
                if class_weights is not None:
                    class_weights_tensor = torch.tensor(
                        [class_weights[int(label)] for label in y.values],
                        dtype=torch.float32,
                    )
                    lossfn = nn.BCELoss(weight=class_weights_tensor)

                optimizer = optim.Adam(model.parameters(), lr=0.001)

                for _ in range(self.epochs):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = lossfn(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()

                self.fitted_models[name] = model
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X using weighted average of all models.

        Args:
            X: Features to predict

        Returns:
            Weighted average of predicted probabilities
        """
        predictions = []
        weights = []

        for name, model in self.fitted_models.items():
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
            else:
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_tensor)
                    pred = np.array(
                        [
                            [1 - test_output.item(), test_output.item()]
                            for test_output in test_outputs
                        ]
                    )

            predictions.append(pred)
            weights.append(self.weights[name])

        # Stack predictions and calculate weighted average
        predictions = np.stack(predictions)
        weights = np.array(weights).reshape(-1, 1, 1)
        weighted_preds = (predictions * weights).sum(axis=0)
        normalized_preds = weighted_preds / weights.sum()

        return normalized_preds

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X: Features to predict

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
