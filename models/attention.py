import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def self_attention1(
    x,
    weight_att,
    bias_att,
    seq_len,
    num_nodes,
    gru_units,
):
    """
    Temporal attention used in the original A3T-GCN implementation.

    Parameters
    ----------
    x : tf.Tensor
        TGCN output with shape:

            [batch_size, seq_len, num_nodes, gru_units]

    weight_att : dict
        Attention weight variables containing:
            'w1'
            'w2'

    bias_att : dict
        Attention bias variables containing:
            'b1'
            'b2'

    seq_len : int
        Number of historical time steps.

    num_nodes : int
        Number of graph nodes.

    gru_units : int
        Number of TGCN hidden units.

    Returns
    -------
    context : tf.Tensor
        Temporally weighted representation.

    beta : tf.Tensor
        Temporal attention weights.
    """

    # ---------------------------------------------------------
    # Project the GRU/TGCN representation.
    # ---------------------------------------------------------

    x = tf.matmul(
        tf.reshape(x, [-1, gru_units]),
        weight_att["w1"],
    ) + bias_att["b1"]

    # ---------------------------------------------------------
    # Attention projections.
    #
    # These intentionally preserve the original implementation.
    # In the original code, w2/b2 are used for f, g and h.
    # ---------------------------------------------------------

    f = tf.matmul(
        tf.reshape(x, [-1, num_nodes]),
        weight_att["w2"],
    ) + bias_att["b2"]

    g = tf.matmul(
        tf.reshape(x, [-1, num_nodes]),
        weight_att["w2"],
    ) + bias_att["b2"]

    h = tf.matmul(
        tf.reshape(x, [-1, num_nodes]),
        weight_att["w2"],
    ) + bias_att["b2"]

    # ---------------------------------------------------------
    # Reshape so that attention is calculated across time.
    # ---------------------------------------------------------

    f1 = tf.reshape(
        f,
        [-1, seq_len],
    )

    g1 = tf.reshape(
        g,
        [-1, seq_len],
    )

    h1 = tf.reshape(
        h,
        [-1, seq_len],
    )

    # Original attention score.
    s = g1 * f1

    # Temporal attention weights.
    beta = tf.nn.softmax(
        s,
        axis=-1,
    )

    # ---------------------------------------------------------
    # Apply temporal attention.
    # ---------------------------------------------------------

    context = (
        tf.expand_dims(beta, 2)
        * tf.reshape(
            x,
            [-1, seq_len, num_nodes],
        )
    )

    context = tf.transpose(
        context,
        perm=[0, 2, 1],
    )

    return context, beta