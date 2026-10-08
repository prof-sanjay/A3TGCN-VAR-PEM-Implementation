import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def a3tgcn_loss(y_pred, y_true, regularization_loss=0.0):
    """
    Original A3T-GCN loss used in the legacy implementation.

    The legacy code computes:

        tf.reduce_mean(
            tf.nn.l2_loss(y_pred - label) + Lreg
        )

    We preserve that behavior here so that the clean implementation
    can be compared directly against the original baseline.

    Parameters
    ----------
    y_pred : tf.Tensor
        Model predictions.

    y_true : tf.Tensor
        Ground-truth target values.

    regularization_loss : tf.Tensor or float
        Sum of regularization terms.

    Returns
    -------
    tf.Tensor
        Scalar loss tensor.
    """

    prediction_loss = tf.nn.l2_loss(y_pred - y_true)

    total_loss = tf.reduce_mean(
        prediction_loss + regularization_loss
    )

    return total_loss