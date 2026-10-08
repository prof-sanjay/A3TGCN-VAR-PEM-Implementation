import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from .tgcn_cell import tgcnCell
from .attention import self_attention1


class A3TGCNModel:
    """
    A3T-GCN model.

    This class contains only the TensorFlow model architecture.
    Training, validation, checkpointing, and final evaluation are
    handled separately.
    """

    def __init__(
        self,
        seq_len,
        num_nodes,
        gru_units,
        pre_len,
        adjacency,
        lambda_loss=0.0015,
    ):
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.gru_units = gru_units
        self.pre_len = pre_len
        self.adjacency = adjacency
        self.lambda_loss = lambda_loss

        self._build_placeholders()
        self._build_model()
        self._build_loss()

    def _build_placeholders(self):
        """Create input and target placeholders."""

        self.inputs = tf.placeholder(
            tf.float32,
            shape=[None, self.seq_len, self.num_nodes],
            name="inputs",
        )

        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.pre_len, self.num_nodes],
            name="labels",
        )

    def _build_model(self):
        """Build the A3T-GCN model."""

        self.weights = {
            "out": tf.Variable(
                tf.random_normal(
                    [self.seq_len, self.pre_len],
                    mean=1.0,
                ),
                name="weight_o",
            )
        }

        self.biases = {
            "out": tf.Variable(
                tf.random_normal(
                    [self.pre_len]
                ),
                name="bias_o",
            )
        }

        self.attention_weights = {
            "w1": tf.Variable(
                tf.random_normal(
                    [self.gru_units, 1],
                    stddev=0.1,
                ),
                name="att_w1",
            ),
            "w2": tf.Variable(
                tf.random_normal(
                    [self.num_nodes, 1],
                    stddev=0.1,
                ),
                name="att_w2",
            ),
        }

        self.attention_biases = {
            "b1": tf.Variable(
                tf.random_normal([1]),
                name="att_b1",
            ),
            "b2": tf.Variable(
                tf.random_normal([1]),
                name="att_b2",
            ),
        }

        (
            self.prediction,
            self.rnn_outputs,
            self.states,
            self.attention,
        ) = self._build_tgcn()

        self.y_pred = self.prediction

    def _build_tgcn(self):
        """
        Build:

        Input
          ↓
        TGCN
          ↓
        Temporal attention
          ↓
        Output projection
          ↓
        Prediction
        """

        # -----------------------------------------------------
        # TGCN cell
        # -----------------------------------------------------

        cell_1 = tgcnCell(
            self.gru_units,
            self.adjacency,
            num_nodes=self.num_nodes,
        )

        cell = tf.nn.rnn_cell.MultiRNNCell(
            [cell_1],
            state_is_tuple=True,
        )

        # -----------------------------------------------------
        # Convert [batch, seq_len, nodes] into a sequence
        # of [batch, nodes] tensors.
        # -----------------------------------------------------

        sequence = tf.unstack(
            self.inputs,
            axis=1,
        )

        # -----------------------------------------------------
        # Recurrent graph
        # -----------------------------------------------------

        outputs, states = tf.nn.static_rnn(
            cell,
            sequence,
            dtype=tf.float32,
        )

        # -----------------------------------------------------
        # Arrange TGCN outputs for temporal attention.
        #
        # Original:
        #
        # [seq_len * batch, nodes, gru_units]
        #       ↓
        # [seq_len, batch, nodes, gru_units]
        #       ↓
        # [batch, seq_len, nodes, gru_units]
        # -----------------------------------------------------

        out = tf.concat(
            outputs,
            axis=0,
        )

        out = tf.reshape(
            out,
            shape=[
                self.seq_len,
                -1,
                self.num_nodes,
                self.gru_units,
            ],
        )

        out = tf.transpose(
            out,
            perm=[1, 0, 2, 3],
        )

        # -----------------------------------------------------
        # Temporal attention
        # -----------------------------------------------------

        last_output, attention = self_attention1(
            out,
            self.attention_weights,
            self.attention_biases,
            seq_len=self.seq_len,
            num_nodes=self.num_nodes,
            gru_units=self.gru_units,
        )

        # -----------------------------------------------------
        # Output projection
        # -----------------------------------------------------

        output = tf.reshape(
            last_output,
            shape=[-1, self.seq_len],
        )

        output = tf.matmul(
            output,
            self.weights["out"],
        )

        output = tf.nn.bias_add(
            output,
            self.biases["out"],
        )

        output = tf.reshape(
            output,
            shape=[
                -1,
                self.num_nodes,
                self.pre_len,
            ],
        )

        output = tf.transpose(
            output,
            perm=[0, 2, 1],
        )

        output = tf.reshape(
            output,
            shape=[-1, self.num_nodes],
        )

        return output, outputs, states, attention

    def _build_loss(self):
        """
        Build the original A3T-GCN loss and RMSE tensors.
        """

        self.label = tf.reshape(
            self.labels,
            [-1, self.num_nodes],
        )

        # -----------------------------------------------------
        # L2 regularization from the original implementation.
        # -----------------------------------------------------

        regularization_loss = self.lambda_loss * sum(
            tf.nn.l2_loss(variable)
            for variable in tf.trainable_variables()
        )

        # -----------------------------------------------------
        # Preserve the original loss formulation.
        # -----------------------------------------------------

        self.loss = tf.reduce_mean(
            tf.nn.l2_loss(
                self.y_pred - self.label
            )
            + regularization_loss
        )

        # -----------------------------------------------------
        # Preserve the original RMSE formulation.
        # -----------------------------------------------------

        self.error = tf.sqrt(
            tf.reduce_mean(
                tf.square(
                    self.y_pred - self.label
                )
            )
        )