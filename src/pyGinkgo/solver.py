# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from pyGinkgo import pyGinkgoBindings as pGB
from .pyGinkgoBindings.solver import *
import pyGinkgo as pg
from . import gko_types
import numpy as np

def _shape(block):
    """Return a dense block shape as a regular Python tuple."""
    return tuple(block.shape)


def _require_shape(block, expected, name):
    expected = tuple(expected)
    actual = _shape(block)
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}.")


def _empty_dense_like(prototype, shape=None):
    """Allocate uninitialized Dense storage on the prototype executor."""
    if shape is None or tuple(shape) == _shape(prototype):
        return prototype.create_with_config_of()
    return prototype.create_with_type_of(tuple(shape))


def _small_dense_from_numpy_like(prototype, values):
    """Copy a small host coefficient/projected matrix to prototype's executor."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(
            "A projected/coefficient matrix must be two-dimensional."
        )
    values = np.ascontiguousarray(values)
    return type(prototype)(prototype.get_executor(), values)


def _small_dense_to_numpy(block):
    """Copy a small Dense matrix to an owning NumPy array on the host."""
    host = block.copy_to_host()
    return np.array(host, copy=True).reshape(_shape(block))


def _copy_block(destination, source):
    """Copy one full block without changing the destination executor."""
    _require_shape(source, _shape(destination), "source")
    destination.copy_from(source)


def _apply_block(operator, block, output):
    """Compute output = operator @ block for all block columns at once."""
    _, block_cols = _shape(block)
    _, output_cols = _shape(output)
    if output_cols != block_cols:
        raise ValueError(
            "Operator input and output blocks must have the same number of "
            f"columns, got {block_cols} and {output_cols}."
        )
    # Do not require operator.shape here: every Ginkgo LinOp supports apply(),
    # while not every derived Python binding currently exposes a shape property.
    operator.apply(block, output)


def _transpose_block(block, output):
    """Write a block transpose into reusable executor-local storage."""
    rows, cols = _shape(block)
    _require_shape(output, (cols, rows), "transpose output")
    block.transpose_into(output)


def _gram(block_x, block_y, output, transpose_workspace):
    """Compute the full column-pair Gram matrix block_x.T @ block_y."""
    x_rows, x_cols = _shape(block_x)
    y_rows, y_cols = _shape(block_y)
    if x_rows != y_rows:
        raise ValueError(
            "Gram matrix inputs must have the same number of rows, "
            f"got {x_rows} and {y_rows}."
        )
    _require_shape(transpose_workspace, (x_cols, x_rows),
                   "transpose workspace")
    _require_shape(output, (x_cols, y_cols), "Gram output")
    block_x.transpose_into(transpose_workspace)
    transpose_workspace.apply(block_y, output)

def _gram_standard(X, Y):
    """Return the full small host Gram matrix X.T @ Y."""
    x_rows, x_cols = _shape(X)
    y_rows, y_cols = _shape(Y)

    if x_rows != y_rows:
        raise ValueError(
            "Gram matrix inputs must have the same number of rows, "
            f"got {x_rows} and {y_rows}."
        )

    transpose = _empty_dense_like(X, (x_cols, x_rows))
    gram = _empty_dense_like(X, (x_cols, y_cols))

    _gram(X, Y, gram, transpose)

    # Only x_cols-by-y_cols data moves to the host.
    return _small_dense_to_numpy(gram)


def _mix_columns(block, coefficients, output):
    """Compute output = block @ coefficients as one dense block product."""
    rows, inner = _shape(block)
    coeff_rows, coeff_cols = _shape(coefficients)
    if inner != coeff_rows:
        raise ValueError(
            "Coefficient rows must match the block column count, "
            f"got {coeff_rows} and {inner}."
        )
    _require_shape(output, (rows, coeff_cols), "mixed-column output")
    block.apply(coefficients, output)


def _mix_columns_advanced(block, coefficients, output, *, alpha=1.0,
                          beta=0.0):
    """Compute output = alpha * block @ coefficients + beta * output."""
    rows, inner = _shape(block)
    coeff_rows, coeff_cols = _shape(coefficients)
    if inner != coeff_rows:
        raise ValueError(
            "Coefficient rows must match the block column count, "
            f"got {coeff_rows} and {inner}."
        )
    _require_shape(output, (rows, coeff_cols), "mixed-column output")
    block.apply(alpha, coefficients, beta, output)


def _add_scaled_block(destination, alpha, source):
    """Compute destination += alpha * source for a scalar or row vector."""
    _require_shape(source, _shape(destination), "source")
    destination.add_scaled(alpha, source)


def _sub_scaled_block(destination, alpha, source):
    """Compute destination -= alpha * source for a scalar or row vector."""
    _require_shape(source, _shape(destination), "source")
    destination.sub_scaled(alpha, source)


def _column_norm2(block, output):
    """Compute one Euclidean norm for every column of a block."""
    _, cols = _shape(block)
    _require_shape(output, (1, cols), "norm output")
    block.compute_norm2(output)

def _as_small_host_matrix(values, *, name):
    """Validate and return an owning, contiguous two-dimensional host matrix."""
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional host matrix.")
    return np.ascontiguousarray(matrix)


def _allocate_blopex_workspace(X0, itmax):
    """Allocate all Phase-2 BLOPEX block and reduction workspaces."""
    n, m = _shape(X0)

    # Large n-by-m blocks always remain Ginkgo Dense objects on X0's executor.
    blocks = {
        "X": X0.clone(),
        "R": _empty_dense_like(X0),
        "Z": _empty_dense_like(X0),
        "P": _empty_dense_like(X0),
        "W": _empty_dense_like(X0),
        "AX": _empty_dense_like(X0),
        "AZ": _empty_dense_like(X0),
        "AP": _empty_dense_like(X0),
    }

    # Reusable device-local storage for block reductions and small products.
    blocks.update({
        "transpose": _empty_dense_like(X0, (m, n)),
        "gram": _empty_dense_like(X0, (m, m)),
        "norms": _empty_dense_like(X0, (1, m)),
        "lambda_row": _empty_dense_like(X0, (1, m)),
    })

    # Only convergence history is host-side at this phase.
    blocks["res"] = np.full(
        (m, itmax + 1),
        np.nan,
        dtype=np.float64,
    )
    return blocks

def _right_multiply(X, C, out):
    """Compute out = X @ C, where C is a small host matrix."""
    C = np.asarray(C)

    if C.ndim != 2:
        raise ValueError("C must be a two-dimensional host matrix.")

    C = np.ascontiguousarray(C)

    if C.shape[0] != X.shape[1]:
        raise ValueError(
            "C.shape[0] must match X.shape[1], "
            f"got {C.shape[0]} and {X.shape[1]}."
        )

    C_gko = _small_dense_from_numpy_like(X, C)
    _mix_columns(X, C_gko, out)

def _linear_combination(terms, out, workspace):
    """Compute out = X1 @ C1 + X2 @ C2 + ... using full blocks."""
    terms = list(terms)

    if not terms:
        raise ValueError(
            "terms must contain at least one block/coefficient pair."
        )

    if workspace is out:
        raise ValueError("workspace must be distinct from out.")

    _require_shape(
        workspace,
        _shape(out),
        "linear-combination workspace",
    )

    out_rows, out_cols = _shape(out)
    prepared = []

    for index, (block, coefficients) in enumerate(terms):
        if block is workspace:
            raise ValueError(
                f"terms[{index}] aliases workspace."
            )

        block_rows, block_cols = _shape(block)
        if block_rows != out_rows:
            raise ValueError(
                f"terms[{index}] must have {out_rows} rows, "
                f"got {block_rows}."
            )

        coefficients = np.asarray(coefficients)
        if coefficients.ndim != 2:
            raise ValueError(
                f"terms[{index}] coefficients must be two-dimensional."
            )

        coefficients = np.ascontiguousarray(coefficients)
        expected = (block_cols, out_cols)

        if coefficients.shape != expected:
            raise ValueError(
                f"terms[{index}] coefficients must have shape "
                f"{expected}, got {coefficients.shape}."
            )

        prepared.append((block, coefficients))

    # First term overwrites workspace.
    first_block, first_coefficients = prepared[0]
    _right_multiply(
        first_block,
        first_coefficients,
        workspace,
    )

    # Remaining terms are accumulated with advanced apply:
    # workspace = block @ coefficients + workspace
    for block, coefficients in prepared[1:]:
        C_gko = _small_dense_from_numpy_like(
            block,
            coefficients,
        )

        _mix_columns_advanced(
            block,
            C_gko,
            workspace,
            alpha=1.0,
            beta=1.0,
        )

    _copy_block(out, workspace)

def _column_norms(X):
    """Return one Euclidean norm per column."""
    _, m = _shape(X)

    norms_gko = _empty_dense_like(X, (1, m))
    _column_norm2(X, norms_gko)

    return _small_dense_to_numpy(norms_gko).reshape(m)

def _compute_standard_residual(
    AX,
    X,
    eigenvalues,
    R,
    workspace,
):
    """Compute R = AX - X * eigenvalues columnwise."""
    n, m = _shape(X)

    _require_shape(AX, (n, m), "AX")
    _require_shape(R, (n, m), "R")
    _require_shape(
        workspace,
        (n, m),
        "residual workspace",
    )

    if workspace is X or workspace is AX or workspace is R:
        raise ValueError(
            "residual workspace must be distinct from X, AX, and R."
        )

    eigenvalues = np.asarray(eigenvalues)

    if eigenvalues.ndim > 2:
        raise ValueError(
            "eigenvalues must be one-dimensional or a row/column vector."
        )

    eigenvalues = eigenvalues.reshape(-1)

    if eigenvalues.size != m:
        raise ValueError(
            f"eigenvalues must contain {m} entries, "
            f"got {eigenvalues.size}."
        )

    lambda_row = _small_dense_from_numpy_like(
        X,
        eigenvalues.reshape(1, m),
    )

    # workspace[:, j] = eigenvalues[j] * X[:, j]
    _copy_block(workspace, X)
    workspace.scale(lambda_row)

    # R = AX - workspace
    _copy_block(R, AX)
    _sub_scaled_block(R, 1.0, workspace)

def _orthonormalize_standard(
    W,
    *,
    operator_products=(),
    work,
):
    """Orthonormalize a standard-problem block by a right transformation.

    Computes ``C = chol(W.T @ W)^{-T}``, replaces ``W`` by ``W @ C``, and
    applies the same transformation to every cached operator product.

    Parameters
    ----------
    W
        The n-by-m block to orthonormalize in place.
    operator_products
        Cached products such as ``AW`` that must remain consistent with W.
    work
        An n-by-m Dense block distinct from W and all operator products.

    Returns
    -------
    numpy.ndarray
        The m-by-m transformation C satisfying ``W_new = W_old @ C``.
    """
    n, m = _shape(W)
    _require_shape(work, (n, m), "orthonormalization workspace")

    products = tuple(operator_products)
    if work is W:
        raise ValueError("orthonormalization workspace must be distinct from W.")

    seen = {id(W), id(work)}
    for index, product in enumerate(products):
        _require_shape(product, (n, m), f"operator_products[{index}]")
        if id(product) in seen:
            raise ValueError(
                "operator_products must be distinct from W, work, and each "
                "other."
            )
        seen.add(id(product))

    gram = _gram_standard(W, W)
    gram = 0.5 * (gram + gram.T.conj())

    try:
        # NumPy returns the lower factor L with gram = L @ L.T.
        lower = np.linalg.cholesky(gram)
        transform = np.linalg.solve(
            lower.T.conj(),
            np.eye(m, dtype=gram.dtype),
        )
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "BLOPEX block is rank deficient during orthonormalization."
        ) from error

    _right_multiply(W, transform, work)
    _copy_block(W, work)

    for product in products:
        _right_multiply(product, transform, work)
        _copy_block(product, work)

    return transform


def _rayleigh_ritz_blopex_standard(
    blocks,
    a_blocks,
    num_vectors,
):
    """Solve the BLOPEX projected generalized eigenproblem.

    The trial basis V is represented by ``blocks`` and AV by ``a_blocks``.
    Only the small projected matrices ``H = V.T @ AV`` and ``G = V.T @ V``
    are materialized on the host.

    Returns
    -------
    coefficients : numpy.ndarray
        A ``sum(block.shape[1])``-by-``num_vectors`` coefficient matrix.
    eigenvalues : numpy.ndarray
        The selected Ritz values in ascending order.
    """
    blocks = tuple(blocks)
    a_blocks = tuple(a_blocks)

    if not blocks:
        raise ValueError("blocks must contain at least one trial block.")
    if len(blocks) != len(a_blocks):
        raise ValueError("blocks and a_blocks must have the same length.")

    num_rows = _shape(blocks[0])[0]
    block_widths = []
    for index, (block, a_block) in enumerate(zip(blocks, a_blocks)):
        rows, cols = _shape(block)
        if rows != num_rows:
            raise ValueError(
                f"blocks[{index}] must have {num_rows} rows, got {rows}."
            )
        _require_shape(a_block, (rows, cols), f"a_blocks[{index}]")
        block_widths.append(cols)

    projected_size = sum(block_widths)
    if not 1 <= num_vectors <= projected_size:
        raise ValueError(
            "num_vectors must satisfy 1 <= num_vectors <= the projected "
            f"dimension ({projected_size})."
        )

    h_rows = []
    g_rows = []
    for left_block in blocks:
        h_rows.append([
            _gram_standard(left_block, right_a_block)
            for right_a_block in a_blocks
        ])
        g_rows.append([
            _gram_standard(left_block, right_block)
            for right_block in blocks
        ])

    projected_h = np.block(h_rows)
    projected_g = np.block(g_rows)

    # Roundoff can destroy exact symmetry by a few ulps.
    projected_h = 0.5 * (projected_h + projected_h.T.conj())
    projected_g = 0.5 * (projected_g + projected_g.T.conj())

    try:
        lower = np.linalg.cholesky(projected_g)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "BLOPEX projected Gram matrix is rank deficient."
        ) from error

    # Convert H c = G c lambda, G = L L.T, with y = L.T c:
    #     (L^{-1} H L^{-T}) y = y lambda.
    left_reduced = np.linalg.solve(lower, projected_h)
    reduced = np.linalg.solve(lower, left_reduced.T.conj()).T.conj()
    reduced = 0.5 * (reduced + reduced.T.conj())

    eigenvalues, reduced_vectors = np.linalg.eigh(reduced)
    order = np.argsort(eigenvalues)[:num_vectors]
    eigenvalues = eigenvalues[order]
    reduced_vectors = reduced_vectors[:, order]
    coefficients = np.linalg.solve(lower.T.conj(), reduced_vectors)

    return coefficients, eigenvalues


def _relative_standard_residual_norms(residual, eigenvalues):
    """Return ||r_j||_2 / |lambda_j| for small host convergence data."""
    norms = _column_norms(residual)
    denominator = np.abs(np.asarray(eigenvalues).reshape(-1))
    if norms.shape != denominator.shape:
        raise ValueError(
            "Residual norms and eigenvalues must contain the same number of "
            "entries."
        )
    return np.divide(
        norms,
        denominator,
        out=np.full(norms.shape, np.inf, dtype=np.result_type(norms, float)),
        where=denominator != 0,
    )

def lobpcg_basic_standard_impl_(A, X0, nev,
                          T, itmax, tol,
                          A_products):
    """
    Knyazev, A. V. (2001)
    Toward the optimal preconditioned eigensolver: Locally optimal block preconditioned conjugate gradient method
    SIAM journal on scientific computing, 23(2), 517-541

    Parameters:
    A          : left  hand-side operator, symmetric positive definite, n-by-n
    B          : right hand-side operator, symmetric positive definite, n-by-n
    X0         : initial iterates, n-by-m (m < n)
    nev        : number of wanted eigenpairs, nev <= m
    T          : precondontioner, symmetric positive definite, n-by-n
    itmax      : maximum number of iterations
    tol        : tolerance used for convergence criterion
    A_products : if :implicit, the matrix products with A are updated implicitly
    B_products : if :implicit, the matrix products with B are updated implicitly

    Returns:
    Lambda : last iterates of least dominant eigenvalues, m-by-1
    X      : last iterates of least dominant eigenvectors, n-by-m
    res    : normalized norms of eigenresiduals, m-by-it
    """
    pass

def blopex_lobpcg_standard_impl_(A, X0, nev, *, 
                           T=None, itmax=200, tol=1e-6,
                           A_products="explicit"):
    """
    Knyazev, A. V., Argentati, M. E., Lashuk, I., & Ovtchinnikov, E. E. (2007)
    Block locally optimal preconditioned eigenvalue Xolvers (BLOPEX) in Hypre and PETSc
    SIAM Journal on Scientific Computing, 29(5), 2224-2239.

    Parameters:
    A          : left  hand-side operator, symmetric positive definite, n-by-n
    X0         : initial iterates, n-by-m (m < n)
    nev        : number of wanted eigenpairs, nev <= m
    T          : precondontioner, symmetric positive definite, n-by-n
    itmax      : maximum number of iterations
    tol        : tolerance used for convergence criterion
    A_products : if :implicit, the matrix products with A are updated implicitly
        
    Returns:
    Lambda : last iterates of least dominant eigenvalues, m-by-1
    X      : last iterates of least dominant eigenvectors, n-by-m
    res    : normalized norms of eigenresiduals, m-by-it
    """

    if T is not None:
        raise NotImplementedError(
            "Preconditioned BLOPEX iterations are not implemented yet."
    )

    n, m = _shape(X0)

    if not 1 <= nev <= m:
        raise ValueError(
            "nev must satisfy 1 <= nev <= X0.shape[1]."
        )

    if m >= n:
        raise ValueError(
            "The block size X0.shape[1] must be smaller than n."
        )

    if itmax < 0:
        raise ValueError("itmax must be non-negative.")

    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    if A_products not in {"explicit", "implicit"}:
        raise ValueError(
            "A_products must be 'explicit' or 'implicit'."
        )

    # Temporary restriction for the first implementation.
    if A_products == "implicit":
        raise NotImplementedError(
            "Implicit A-product updates are not implemented yet."
        )

    workspace = _allocate_blopex_workspace(X0, itmax)
    X = workspace["X"]
    R = workspace["R"]
    Z = workspace["Z"]
    P = workspace["P"]
    W = workspace["W"]
    AX = workspace["AX"]
    AZ = workspace["AZ"]
    AP = workspace["AP"]
    residual_history = workspace["res"]

    identity = np.eye(m)

    # Initialization: orthonormalize X, form AX, and perform RR in range(X).
    _orthonormalize_standard(X, work=W)
    _apply_block(A, X, AX)

    coefficients, eigenvalues = _rayleigh_ritz_blopex_standard(
        blocks=[X],
        a_blocks=[AX],
        num_vectors=m,
    )
    _right_multiply(X, coefficients, W)
    _copy_block(X, W)

    # Explicit-product mode deliberately recomputes every operator product.
    _apply_block(A, X, AX)
    _compute_standard_residual(AX, X, eigenvalues, R, W)
    residual_history[:, 0] = _relative_standard_residual_norms(
        R, eigenvalues
    )

    if np.all(residual_history[:nev, 0] < tol) or itmax == 0:
        return eigenvalues, X, residual_history[:, :1]

    for iteration in range(1, itmax + 1):
        # With no preconditioner, Z is the current residual block.
        _copy_block(Z, R)
        _orthonormalize_standard(Z, work=W)
        _apply_block(A, Z, AZ)

        if iteration == 1:
            coefficients, eigenvalues = _rayleigh_ritz_blopex_standard(
                blocks=[X, Z],
                a_blocks=[AX, AZ],
                num_vectors=m,
            )
            Cx = coefficients[:m, :]
            Cz = coefficients[m:2 * m, :]

            # P_new = Z @ Cz.
            _right_multiply(Z, Cz, P)
            _apply_block(A, P, AP)
        else:
            # P participates as an independently normalized trial block.
            _orthonormalize_standard(P, work=W)
            _apply_block(A, P, AP)

            coefficients, eigenvalues = _rayleigh_ritz_blopex_standard(
                blocks=[X, Z, P],
                a_blocks=[AX, AZ, AP],
                num_vectors=m,
            )
            Cx = coefficients[:m, :]
            Cz = coefficients[m:2 * m, :]
            Cp = coefficients[2 * m:3 * m, :]

            # P_new = Z @ Cz + P_old @ Cp. The separate W accumulator
            # preserves P_old until its contribution has been consumed.
            _linear_combination(
                [(Z, Cz), (P, Cp)],
                P,
                W,
            )
            _apply_block(A, P, AP)

        # X_new = X_old @ Cx + P_new.
        _linear_combination(
            [(X, Cx), (P, identity)],
            X,
            W,
        )
        _apply_block(A, X, AX)

        _compute_standard_residual(AX, X, eigenvalues, R, W)
        residual_history[:, iteration] = _relative_standard_residual_norms(
            R, eigenvalues
        )

        if np.all(residual_history[:nev, iteration] < tol):
            return (
                eigenvalues,
                X,
                residual_history[:, :iteration + 1],
            )

    return eigenvalues, X, residual_history


def lobpcg(A, X0, nev,
           B=None, T=None, itmax=200, tol=1e-6,
           method='BLOPEX',
           A_products='explicit',
           B_products='implicit'):
    """
    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) method.
      
    Parameters:
    A          : left  hand-side operator, symmetric positive definite, n-by-n
    X0         : initial iterates, n-by-m (m < n)
    nev        : number of eigenvalues to compute.
    B          : right hand-side operator, symmetric positive definite, n-by-n
    T          : precondontioner, symmetric positive definite, n-by-n
    itmax      : maximum number of iterations
    tol        : tolerance used for convergence criterion
    method     : type of LOBPCG iterations among ('Basic', 'BLOPEX', 'Ortho', 'Skip_ortho')
    A_products : if "implicit", the matrix products with A are updated implicitly
    B_products : if "implicit", the matrix products with B are updated implicitly

    Returns:
    Lambda : last iterates of least dominant eigenvalues, m-by-1
    X      : last iterates of least dominant eigenvectors, n-by-m
    res    : normalized norms of eigenresiduals, m-by-it
    """
         
    if method == "BLOPEX" and B is None:
        return blopex_lobpcg_standard_impl_(
            A,
            X0,
            nev,
            T=T,
            itmax=itmax,
            tol=tol,
            A_products=A_products,
        )

    raise NotImplementedError(...)
    


def gmres(
    device: gko_types.DeviceType,
    matrix: pGB.LinOp,
    max_iters: int,
    krylov_dim: int,
    reduction_factor: float,
    relative_stop_mode: bool = False,
    preconditioner = None
):
    executor = pg.device(device)

     # TODO: create a better way to check the type of the matrix
    typization = type(matrix).__name__.split('_')[1:]
    if len(typization) > 0:
        gmres_cls = getattr(pGB.solver, "gmres_" + typization[0])
    else:
        raise ValueError(f"Not a known matrix type: {typization}.")
    
    args = [
        executor,
        matrix,
        # Conditionally including the preconditioner, if provided
        *([preconditioner] if preconditioner is not None else []),
        max_iters,
        krylov_dim,
        reduction_factor,
        relative_stop_mode
    ]
    
    return gmres_cls(*args)
