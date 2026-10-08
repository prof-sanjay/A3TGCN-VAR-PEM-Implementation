import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class A3TGCNTrainer:
    """
    Training helper for the TensorFlow 1.x A3T-GCN model.

    This class manages the TensorFlow session, optimization,
    training steps, and model checkpointing.

    Model construction remains outside this class.
    """

    def __init__(
        self,
        loss,
        learning_rate,
        optimizer="adam",
    ):
        self.loss = loss
        self.learning_rate = learning_rate

        self.learning_rate_ph = tf.placeholder(
            tf.float32,
            shape=(),
            name="learning_rate",
        )

        if optimizer.lower() == "adam":
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_ph
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: {optimizer}"
            )

        self.train_op = self.optimizer.minimize(self.loss)

        self.session = tf.Session()

    def initialize(self):
        """
        Initialize all TensorFlow variables.
        """
        self.session.run(
            tf.global_variables_initializer()
        )

    def train_batch(
        self,
        feed_dict,
    ):
        """
        Execute one optimization step.

        Parameters
        ----------
        feed_dict : dict
            TensorFlow feed dictionary.

        Returns
        -------
        float
            Batch loss.
        """

        feed_dict = dict(feed_dict)

        feed_dict[self.learning_rate_ph] = self.learning_rate

        _, batch_loss = self.session.run(
            [self.train_op, self.loss],
            feed_dict=feed_dict,
        )

        return float(batch_loss)

    def evaluate_loss(
        self,
        feed_dict,
    ):
        """
        Evaluate the loss without updating model parameters.
        """

        loss_value = self.session.run(
            self.loss,
            feed_dict=feed_dict,
        )

        return float(loss_value)

    def set_learning_rate(self, learning_rate):
        """
        Update the learning rate used by subsequent batches.
        """
        self.learning_rate = float(learning_rate)

    def save(self, checkpoint_path, global_step=None):
        """
        Save the current TensorFlow model.
        """

        saver = tf.train.Saver()

        return saver.save(
            self.session,
            checkpoint_path,
            global_step=global_step,
        )

    def restore(self, checkpoint_path):
        """
        Restore a previously saved TensorFlow model.
        """

        saver = tf.train.Saver()

        saver.restore(
            self.session,
            checkpoint_path,
        )

    def close(self):
        """
        Close the TensorFlow session.
        """

        self.session.close()