"""Factory function for creating logistic regression models."""

try:
    import cuml

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


def create_logistic_regression(max_iter=2000, tol=1e-5, C=1.0, n_jobs=-1, logger=None):
    """Create a logistic regression model, using cuML if available, otherwise scikit-learn.

    Args:
        max_iter: Maximum number of iterations for solver
        tol: Tolerance for stopping criteria
        C: Inverse of regularization strength
        n_jobs: Number of CPU cores to use (only for scikit-learn)
        logger: Optional logger to use for logging which implementation is used

    Returns:
        A logistic regression model from either cuML or scikit-learn
    """
    if CUML_AVAILABLE:
        if logger:
            logger.info("Using cuML LogisticRegression for faster GPU-based training")
        return cuml.linear_model.LogisticRegression(
            max_iter=max_iter,
            tol=tol,
            C=C,
        )
    else:
        if logger:
            logger.info("Using scikit-learn LogisticRegression (cuML not available)")
        return SklearnLogisticRegression(
            max_iter=max_iter,
            tol=tol,
            C=C,
            n_jobs=n_jobs,
        )
