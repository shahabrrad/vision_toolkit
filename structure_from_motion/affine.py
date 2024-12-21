import torch
from torch import Tensor

def normalize_measurements(X: Tensor) -> Tensor:
    r'''
    x1(1)  x2(1)  x3(1) ... xN(1)
    y1(1)  y2(1)  y3(1) ... yN(1)
    ...
    x1(M)  x2(M)  x3(M) ... xN(M)
    y1(M)  y2(M)  y3(M) ... yN(M)
    '''
    # Number of views (M) and points (N)
    num_views = X.shape[0] // 2

    # Reshape into separate views for x and y coordinates
    reshaped_data = X.view(num_views, 2, -1)

    # Compute centroids for each view and subtract them
    centroids = reshaped_data.mean(dim=2, keepdim=True)
    normalized_data = reshaped_data - centroids

    # Reshape back to the original format
    normalized_data = normalized_data.view(-1, X.shape[1])

    return normalized_data, centroids


def get_structure_and_motion(
    D: Tensor, 
    k: int = 3
) -> tuple[Tensor, Tensor]:
    # SVD decomposition of the normalized data matrix
    U, Sigma, Vt = torch.linalg.svd(D, full_matrices=False)

    # Keep the top 3 singular values
    U_3 = U[:, :3]
    Sigma_3 = torch.diag(Sigma[:3])
    V_3 = Vt[:3, :]

    M = U_3 @ Sigma_3.sqrt()
    S = Sigma_3.sqrt() @ V_3

    return M, S


def get_Q(M) -> Tensor:

    L = torch.zeros((3, 3), dtype=torch.float64)
    for i in range(0, M.shape[0], 2):
        x_row = M[i, :]
        y_row = M[i + 1, :]
        L += torch.outer(x_row, x_row) + torch.outer(y_row, y_row)

    # Solve for Q^T Q
    Q_TQ = torch.linalg.pinv(L)

    Q = torch.linalg.cholesky(Q_TQ)

    return Q
