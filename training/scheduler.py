class ValidationLRScheduler:
    """
    Learning-rate scheduler based on validation RMSE.

    The learning rate is reduced when validation RMSE does not
    improve for a specified number of epochs.
    """

    def __init__(
        self,
        initial_lr,
        patience=5,
        factor=0.5,
        min_lr=1e-5,
    ):
        self.lr = float(initial_lr)
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)

        self.best_metric = float("inf")
        self.wait = 0

    def step(self, validation_rmse):
        """
        Update the learning rate using validation RMSE.

        Returns
        -------
        float
            Current learning rate.
        """

        validation_rmse = float(validation_rmse)

        if validation_rmse < self.best_metric:
            self.best_metric = validation_rmse
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                new_lr = max(
                    self.lr * self.factor,
                    self.min_lr,
                )

                if new_lr < self.lr:
                    self.lr = new_lr

                self.wait = 0

        return self.lr

    @property
    def current_lr(self):
        """Return the current learning rate."""
        return self.lr