import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def normalized_adj(adj):
    """
    Normalize an adjacency matrix using the normalization
    used by the original A3T-GCN implementation.
    """

    adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(
        rowsum,
        -0.5,
    ).flatten()

    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = sp.diags(
        d_inv_sqrt
    )

    normalized_adj = (
        adj
        .dot(d_mat_inv_sqrt)
        .transpose()
        .dot(d_mat_inv_sqrt)
        .tocoo()
    )

    normalized_adj = normalized_adj.astype(
        np.float32
    )

    return normalized_adj


def sparse_to_tuple(mx):
    """
    Convert a SciPy sparse matrix to a TensorFlow SparseTensor.
    """

    mx = mx.tocoo()

    coords = np.vstack(
        (
            mx.row,
            mx.col,
        )
    ).transpose()

    sparse_tensor = tf.SparseTensor(
        coords,
        mx.data,
        mx.shape,
    )

    return tf.sparse.reorder(
        sparse_tensor
    )


def calculate_laplacian(
    adj,
    lambda_max=1,
):
    """
    Calculate the graph representation used by T-GCN.

    The implementation intentionally preserves the original
    A3T-GCN preprocessing behavior.
    """

    adj = normalized_adj(
        adj + sp.eye(adj.shape[0])
    )

    adj = sp.csr_matrix(adj)

    adj = adj.astype(
        np.float32
    )

    return sparse_to_tuple(adj)


def weight_variable_glorot(
    input_dim,
    output_dim,
    name="",
):
    """
    Create a Glorot-uniform weight variable.

    Retained for compatibility with the original implementation.
    """

    init_range = np.sqrt(
        6.0 / (input_dim + output_dim)
    )

    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32,
    )

    return tf.Variable(
        initial,
        name=name,
    )