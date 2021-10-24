import os
import numpy as np
import scipy.sparse

import gurobipy as gp 
from gurobipy import GRB

import torch 
import torch.nn as nn
import torch.nn.functional as F

def generate_setcover(nrows=500, ncols=1000, density=0.05, filename=None, rng=np.random.RandomState(), max_coef=100):
    """
    Generates a setcover instance with specified characteristics, and writes
    it to a file in the LP format.

    **The original objective is to find minimum. The problem is modified to be come equivalent maximum problem:**

    $Ax\geq b, \min{cx} \Rightarrow \\ y = -x + 1, Ay\leq A-b, \max{(cy-c)} \Rightarrow \\ Ay\leq A-b, \max{cy}$
    
    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.

    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float in range (0, 1]
        Desired density of the constraint matrix
    filename: str, optinal
        File to which the LP will be written
    rng: numpy.random.RandomState, optinal
        Random number generator, default: `numpy.random.RandomState()` (Means no seed)
    max_coef: int
        Maximum objective coefficient (>=1), default: 100
    """
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows) # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i+n] = rng.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i+n] = rng.choice(remaining_rows, size=i+n-nrows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(nrows, ncols)).tocsr()
    indices = A.indices
    indptr = A.indptr

    b = np.ones((nrows))
    A = np.array(A.todense())

    # Modify the problem to equivalent maximize problem
    b = A.sum(axis=1) - b

    return A, b, c

def setcover_encoding(A, b, c, x):
    dual_coef = b/b.max()
    slack = (b - A @ x)/b.max()
    obj_cos_sim = (A @ c) / (np.linalg.norm(A, ord = 2, axis = 1) * np.linalg.norm(c, ord = 2))
    not_satisfied = A @ x > b 
    not_tight = A @ x < b
    C = torch.tensor(np.stack([dual_coef, slack, obj_cos_sim, not_satisfied, not_tight], axis = 1), dtype=torch.float)

    coef = c / c.max()
    sol_val = x
    V = torch.tensor(np.stack([coef, sol_val], axis = 1), dtype=torch.float)

    E = torch.tensor(A, dtype=torch.float).float().unsqueeze(-1)

    return V, C, E

def random_assignment(A, b, c):
    m, n = A.shape
    num_ones = np.random.randint(n + 1)
    assignment = np.zeros_like(c, dtype = bool)
    assignment[np.random.choice(n, size = num_ones, replace = False)] = True
    return assignment

if __name__ == "__main__":
    A, b, c = generate_setcover(nrows=20, ncols=40, density=0.5)
    x = np.ones_like(c, dtype = bool)
    print(A, b, c)
    print(A.shape, b.shape, c.shape)
    V, C, E = setcover_encoding(A, b, c, x)
    print(V.shape, C.shape, E.shape)
    print(V)
    print(C)
    print(E.squeeze(-1))