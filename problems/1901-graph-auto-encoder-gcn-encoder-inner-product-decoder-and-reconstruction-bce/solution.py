import numpy as np

def A_normalized(A):
    E = A + np.eye(A.shape[0])
    degree = E.sum(axis=0)
    degree_inv = np.diag(1 / np.sqrt(degree))

    return degree_inv @ E @ degree_inv

def relu(x):
    return np.maximum(0, x)

def gae_encode(A, X, W1, W2):
    A_hat = A_normalized(A)
    B = relu(A_hat @ X @ W1)

    return A_hat @ B @ W2

def gae_logits(Z, pairs):
    return np.array([
        np.dot(Z[i], Z[j]) for i, j in pairs
    ])

def gae_loss(A, Z, pos_weight=None):
    W_hat = Z @ Z.T
    N = A.shape[0]

    if pos_weight is None:
        sigma_A = A.sum()
        pos_weight = (N ** 2 - sigma_A) / sigma_A

    loss = pos_weight * A * np.logaddexp(0.0, -W_hat) + (1.0 - A) * np.logaddexp(0.0, W_hat)

    return np.mean(loss)

def sampled_gae_loss(Z, pos_pairs, neg_pairs):
    pos_pairs = np.asarray(pos_pairs, dtype=int).reshape(-1, 2)
    neg_pairs = np.asarray(neg_pairs, dtype=int).reshape(-1, 2)
    
    pos_u, pos_v = Z[pos_pairs[:, 0]], Z[pos_pairs[:, 1]]
    neg_u, neg_v = Z[neg_pairs[:, 0]], Z[neg_pairs[:, 1]]

    pos_logits = np.sum(pos_u * pos_v, axis=-1)
    neg_logits = np.sum(neg_u * neg_v, axis=-1)

    pos_loss = np.logaddexp(0, -pos_logits)
    neg_loss = np.logaddexp(0, neg_logits)

    N = len(pos_logits) + len(neg_logits)
    return (np.sum(pos_loss) + np.sum(neg_loss)) / N

def vgae_sample(mu, logvar, eps):
    return mu + np.exp(0.5 * logvar) * eps

def vgae_kl(mu, logvar):
    A = mu ** 2 + np.exp(logvar)
    return 0.5 * np.mean(np.sum(A - logvar - 1, axis=-1))
