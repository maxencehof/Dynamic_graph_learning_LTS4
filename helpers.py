import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.linalg import sqrtm
import time
from scipy import spatial
from scipy import sparse
from sklearn.covariance import graphical_lasso
from pyunlocbox.functions import dummy, _prox_star
from pyunlocbox import acceleration
from pyunlocbox import functions

def signal_to_laplacian(graph_signals, threshold=0.05, verbosity='NONE'):
    W2_kal = log_degree_barrier((graph_signals).T, alpha=1, beta=1, verbosity=verbosity)
    W2_kal_thresholded = W2_kal.copy()
    W2_kal_thresholded[abs(W2_kal_thresholded) < threshold] = 0
    return W_to_L(W2_kal_thresholded)



def signal_to_laplacian_prior(graph_signals, prior, step_prior=0.05, threshold=0.05, verbosity='NONE'):
    W2_kal = log_degree_barrier((graph_signals).T, alpha=1, beta=1, prior=prior, step_prior=step_prior, verbosity=verbosity)
    W2_kal_thresholded = W2_kal.copy()
    W2_kal_thresholded[abs(W2_kal_thresholded) < threshold] = 0
    return W_to_L(W2_kal_thresholded)


def signal_to_laplacian_prior_Frob(graph_signals, prior_Frob, step_prior=0.05, threshold=0.05, verbosity='NONE'):
    W2_kal = log_degree_barrier((graph_signals).T, alpha=1, beta=1, prior_Frob=prior_Frob, step_prior=step_prior, verbosity=verbosity)
    W2_kal_thresholded = W2_kal.copy()
    W2_kal_thresholded[abs(W2_kal_thresholded) < threshold] = 0
    return W_to_L(W2_kal_thresholded)

def laplacian_matrix_to_graph(L):
    """
    Convert a Laplacian matrix to a NetworkX graph.

    Parameters:
        L (numpy.ndarray): Laplacian matrix.

    Returns:
        nx.Graph: NetworkX graph.
    """
    # Get the number of nodes from the size of the Laplacian matrix
    num_nodes = L.shape[0]
    
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # Iterate over the non-zero entries of the Laplacian matrix to add edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if L[i, j] != 0:
                # Add an edge between nodes i and j with weight L[i, j]
                G.add_edge(i, j, weight=-L[i, j])

    return G

def upper_diagonal(matrix):
    upper_diag = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if j >= i: ## maybe change to strictly >
                upper_diag.append(matrix[i][j])
    return upper_diag


def Wasserstein_distance(L1, L2):
    P1 = np.linalg.pinv(L1)
    P2 = np.linalg.pinv(L2)
    return np.trace(P1+P2) - 2*np.trace(np.real(sqrtm(P1@P2)))

def print_W2_L2_norm(L1, L2):
    w2 = Wasserstein_distance(L1, L2)
    l2 = np.linalg.norm(L1-L2)
    print("Wasserstein distance:", w2, "Frobenius norm:", l2)


def plot_matrix(W):
    plt.figure(figsize=(18, 10))
    sns.heatmap(W, annot=True, fmt=".2g", cbar=False)
    plt.show()

def L_to_vec_w(L):
    # Get the shape of the Laplacian matrix
    n, _ = L.shape
    
    # Initialize edge vector
    edge_vector = []
    
    # Iterate over upper triangular part of the Laplacian matrix
    for i in range(n):
        for j in range(i+1, n):
            # If there's an edge, append its weight to the edge vector
            if L[i][j] != 0:
                edge_vector.append(L[i][j] * -1)  # Edge weight is negative of Laplacian value
            elif L[j][i] != 0:
                edge_vector.append(L[j][i] * -1)  # Edge weight is negative of Laplacian value
            else:
                edge_vector.append(0)  # No edge, append 0
                
    return edge_vector

def W_to_L(W):
    return np.diag(W @ np.ones(W.shape[0]))-W


def log_degree_barrier(X, dist_type='sqeuclidean', alpha=1, beta=1, step=0.5,
                       w0=None, prior=None, prior_Frob=None, step_prior=0, maxit=1000, rtol=1e-6, retall=False,
                       verbosity='NONE'):
    r"""
    Learn graph by imposing a log barrier on the degrees

    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1} - \alpha 1^{T} \log{W1} + \beta \| W \|_{F}^{2}`,
    where :math:`Z` is a pairwise distance matrix, and :math:`\mathcal{W}_m`
    is the set of valid symmetric weighted adjacency matrices.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    dist_type : string
        Type of pairwise distance between variables. See
        :func:`spatial.distance.pdist` for the possible options.
    alpha : float, optional
        Regularization parameter acting on the log barrier
    beta : float, optional
        Regularization parameter controlling the density of the graph
    step : float, optional
        A number between 0 and 1 defining a stepsize value in the admissible
        stepsize interval (see [Komodakis & Pesquet, 2015], Algorithm 6)
    w0 : array_like, optional
        Initialization of the edge weights. Must be an N(N-1)/2-dimensional
        vector.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. Relative tolerance between successive updates.
    retall : boolean
        Return solution and problem details. See output of
        :func:`pyunlocbox.solvers.solve`.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        Level of verbosity of the solver. See :func:`pyunlocbox.solvers.solve`.

    Returns
    -------
    W : array_like
        Learned weighted adjacency matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.

    Notes
    -----
    This is the solver proposed in [Kalofolias, 2016] :cite:`kalofolias2016`.


    Examples
    --------
    >>> import learn_graph as lg
    >>> import networkx as nx
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import spatial
    >>> G_gt = nx.waxman_graph(100)
    >>> pos = nx.random_layout(G_gt)
    >>> coords = np.array(list(pos.values()))
    >>> def s1(x, y):
            return np.sin((2 - x - y)**2)
    >>> def s2(x, y):
            return np.cos((x + y)**2)
    >>> def s3(x, y):
            return (x - 0.5)**2 + (y - 0.5)**3 + x - y
    >>> def s4(x, y):
            return np.sin(3 * ( (x - 0.5)**2 + (y - 0.5)**2 ) )
    >>> X = np.array((s1(coords[:,0], coords[:,1]),
                      s2(coords[:,0], coords[:,1]),
                      s3(coords[:,0], coords[:,1]),
                      s4(coords[:,0], coords[:,1]))).T
    >>> z = 25 * spatial.distance.pdist(X, 'sqeuclidean')
    >>> W = lg.log_degree_barrier(z)
    >>> W[W < np.percentile(W, 96)] = 0
    >>> G_learned = nx.from_numpy_matrix(W)
    >>> plt.figure(figsize=(12, 6))
    >>> plt.subplot(1,2,1)
    >>> nx.draw(G_gt, pos=pos)
    >>> plt.title('Ground Truth')
    >>> plt.subplot(1,2,2)
    >>> nx.draw(G_learned, pos=pos)
    >>> plt.title('Learned')
    """

    # Parse X
    N = X.shape[0]
    z = spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Parse stepsize
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    K, Kt = weight2degmap(N)
    norm_K = np.sqrt(2 * (N - 1))

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: - alpha * np.sum(np.log(np.maximum(
        np.finfo(np.float64).eps, K(w))))
    f2._prox = lambda d, gamma: np.maximum(
        0, 0.5 * (d + np.sqrt(d**2 + (4 * alpha * gamma))))

    f3 = functions.func()
    f3._eval = lambda w: beta * np.sum(w**2)
    f3._grad = lambda w: 2 * beta * w
    lipg = 2 * beta

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

    # Solve problem
    ## use the following part if you want to use the previous cell

    solver = mlfbf(L=K, Lt=Kt, step=stepsize)
    problem = solve([f1, f2, f3], x0=w0, solver=solver, prior=prior, prior_Frob=prior_Frob, step_prior=step_prior, maxit=maxit,
                            rtol=rtol, verbosity=verbosity)
    
    ## Use the following part if you want to use the pyunlocbox library
    #solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize)
    #problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
    #                        rtol=rtol, verbosity=verbosity)

    # Transform weight matrix from vector form to matrix form
    W = spatial.distance.squareform(problem['sol'])

    if retall:
        return W, problem
    else:
        return W
    
def l2_degree_reg(X, dist_type='sqeuclidean', alpha=1, s=None, step=0.5,
                  w0=None, maxit=1000, rtol=1e-5, retall=False,
                  verbosity='NONE'):
    r"""
    Learn graph by regularizing the l2-norm of the degrees.

    This is done by solving
    :math:`\tilde{W} = \underset{W \in \mathcal{W}_m}{\text{arg}\min} \,
    \|W \odot Z\|_{1,1} + \alpha \|W1}\|^2 + \alpha \| W \|_{F}^{2}`, subject
    to :math:`\|W\|_{1,1} = s`, where :math:`Z` is a pairwise distance matrix,
    and :math:`\mathcal{W}_m`is the set of valid symmetric weighted adjacency
    matrices.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    dist_type : string
        Type of pairwise distance between variables. See
        :func:`spatial.distance.pdist` for the possible options.
    alpha : float, optional
        Regularization parameter acting on the l2-norm.
    s : float, optional
        The "sparsity level" of the weight matrix, as measured by its l1-norm.
    step : float, optional
        A number between 0 and 1 defining a stepsize value in the admissible
        stepsize interval (see [Komodakis & Pesquet, 2015], Algorithm 6)
    w0 : array_like, optional
        Initialization of the edge weights. Must be an N(N-1)/2-dimensional
        vector.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. Relative tolerance between successive updates.
    retall : boolean
        Return solution and problem details. See output of
        :func:`pyunlocbox.solvers.solve`.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        Level of verbosity of the solver. See :func:`pyunlocbox.solvers.solve`.

    Returns
    -------
    W : array_like
        Learned weighted adjacency matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.

    Notes
    -----
    This is the problem proposed in [Dong et al., 2015].


    Examples
    --------

    """

    # Parse X
    N = X.shape[0]
    E = int(N * (N - 1.) / 2.)
    z = spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Parse s
    s = N if s is None else s

    # Parse step
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    one_vec = np.ones(E)

    def K(w):
        return np.array([2 * np.dot(one_vec, w)])

    def Kt(n):
        return 2 * n * one_vec

    norm_K = 2 * np.sqrt(E)

    # Get weight-to-degree map
    S, St = weight2degmap(N)

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: 0.
    f2._prox = lambda d, gamma: s

    f3 = functions.func()
    f3._eval = lambda w: alpha * (2 * np.sum(w**2) + np.sum(S(w)**2))
    f3._grad = lambda w: alpha * (4 * w + St(S(w)))
    lipg = 2 * alpha * (N + 1)

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

    # Solve problem
    ## use the following part if you want to use the previous cell
    solver = mlfbf(L=K, Lt=Kt, step=stepsize)
    problem = solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
                            rtol=rtol, verbosity=verbosity)
    
    ## Use the following part if you want to use the pyunlocbox library

    #solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize)
    #problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
    #                        rtol=rtol, verbosity=verbosity)

    # Transform weight matrix from vector form to matrix form
    W = spatial.distance.squareform(problem['sol'])

    if retall:
        return W, problem
    else:
        return W


def glasso(X, alpha=1, w0=None, maxit=1000, rtol=1e-5, retall=False,
           verbosity='NONE'):
    r"""
    Learn graph by imposing promoting sparsity in the inverse covariance.

    This is done by solving
    :math:`\tilde{W} = \underset{W \succeq 0}{\text{arg}\min} \,
    -\log \det W - \text{tr}(SW) + \alpha\|W \|_{1,1},
    where :math:`S` is the empirical (sample) covariance matrix.

    Parameters
    ----------
    X : array_like
        An N-by-M data matrix of N variable observations in an M-dimensional
        space. The learned graph will have N nodes.
    alpha : float, optional
        Regularization parameter acting on the l1-norm
    w0 : array_like, optional
        Initialization of the inverse covariance. Must be an N-by-N symmetric
        positive semi-definite matrix.
    maxit : int, optional
        Maximum number of iterations.
    rtol : float, optional
        Stopping criterion. If the dual gap goes below this value, iterations
        are stopped. See :func:`sklearn.covariance.graphical_lasso`.
    retall : boolean
        Return solution and problem details.
    verbosity : {'NONE', 'ALL'}, optional
        Level of verbosity of the solver.
        See :func:`sklearn.covariance.graphical_lasso`/

    Returns
    -------
    W : array_like
        Learned inverse covariance matrix
    problem : dict, optional
        Information about the solution of the optimization. Only returned if
        retall == True.

    Notes
    -----
    This function uses the solver :func:`sklearn.covariance.graphical_lasso`.

    Examples
    --------

    """

    # Parse X
    S = np.cov(X)

    # Parse initial point
    w0 = np.ones(S.shape) if w0 is None else w0
    if (w0.shape != S.shape):
        raise ValueError("w0 must be of dimension N-by-N.")

    # Solve problem
    tstart = time.time()
    res = graphical_lasso(emp_cov=S,
                      alpha=alpha,
                      cov_init=w0,
                      mode='cd',
                      tol=rtol,
                      max_iter=maxit,
                      verbose=(verbosity == 'ALL'),
                      return_costs=True,
                      return_n_iter=True)

    problem = {'sol':       res[1],
               'dual_sol':  res[0],
               'solver':    'sklearn.covariance.graph_lasso',
               'crit':      'dual_gap',
               'niter':     res[3],
               'time':      time.time() - tstart,
               'objective': np.array(res[2])[:, 0]}

    W = problem['sol']

    if retall:
        return W, problem
    else:
        return W



def solve(functions, x0, solver=None, prior=None, prior_Frob=None, step_prior=0, atol=None, dtol=None, rtol=1e-3,
          xtol=None, maxit=200, verbosity='LOW'):
    r"""
    Solve an optimization problem whose objective function is the sum of some
    convex functions.

    This function minimizes the objective function :math:`f(x) =
    \sum\limits_{k=0}^{k=K} f_k(x)`, i.e. solves
    :math:`\operatorname{arg\,min}\limits_x f(x)` for :math:`x \in
    \mathbb{R}^{n \times N}` where :math:`n` is the dimensionality of the data
    and :math:`N` the number of independent problems. It returns a dictionary
    with the found solution and some informations about the algorithm
    execution.

    Parameters
    ----------
    functions : list of objects
        A list of convex functions to minimize. These are objects who must
        implement the :meth:`pyunlocbox.functions.func.eval` method. The
        :meth:`pyunlocbox.functions.func.grad` and / or
        :meth:`pyunlocbox.functions.func.prox` methods are required by some
        solvers. Note also that some solvers can only handle two convex
        functions while others may handle more. Please refer to the
        documentation of the considered solver.
    x0 : array_like
        Starting point of the algorithm, :math:`x_0 \in \mathbb{R}^{n \times
        N}`. Note that if you pass a numpy array it will be modified in place
        during execution to save memory. It will then contain the solution. Be
        careful to pass data of the type (int, float32, float64) you want your
        computations to use.
    solver : solver class instance, optional
        The solver algorithm. It is an object who must inherit from
        :class:`pyunlocbox.solvers.solver` and implement the :meth:`_pre`,
        :meth:`_algo` and :meth:`_post` methods. If no solver object are
        provided, a standard one will be chosen given the number of convex
        function objects and their implemented methods.
    atol : float, optional
        The absolute tolerance stopping criterion. The algorithm stops when
        :math:`f(x^t) < atol` where :math:`f(x^t)` is the objective function at
        iteration :math:`t`. Default is None.
    dtol : float, optional
        Stop when the objective function is stable enough, i.e. when
        :math:`\left|f(x^t) - f(x^{t-1})\right| < dtol`. Default is None.
    rtol : float, optional
        The relative tolerance stopping criterion. The algorithm stops when
        :math:`\left|\frac{ f(x^t) - f(x^{t-1}) }{ f(x^t) }\right| < rtol`.
        Default is :math:`10^{-3}`.
    xtol : float, optional
        Stop when the variable is stable enough, i.e. when :math:`\frac{\|x^t -
        x^{t-1}\|_2}{\sqrt{n N}} < xtol`. Note that additional memory will be
        used to store :math:`x^{t-1}`. Default is None.
    maxit : int, optional
        The maximum number of iterations. Default is 200.
    verbosity : {'NONE', 'LOW', 'HIGH', 'ALL'}, optional
        The log level : ``'NONE'`` for no log, ``'LOW'`` for resume at
        convergence, ``'HIGH'`` for info at all solving steps, ``'ALL'`` for
        all possible outputs, including at each steps of the proximal operators
        computation. Default is ``'LOW'``.

    Returns
    -------
    sol : ndarray
        The problem solution.
    solver : str
        The used solver.
    crit : {'ATOL', 'DTOL', 'RTOL', 'XTOL', 'MAXIT'}
        The used stopping criterion. See above for definitions.
    niter : int
        The number of iterations.
    time : float
        The execution time in seconds.
    objective : ndarray
        The successive evaluations of the objective function at each iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers

    Define a problem:

    >>> y = [4, 5, 6, 7]
    >>> f = functions.norm_l2(y=y)

    Solve it:

    >>> x0 = np.zeros(len(y))
    >>> ret = solvers.solve([f], x0, atol=1e-2, verbosity='ALL')
    INFO: Dummy objective function added.
    INFO: Selected solver: forward_backward
        norm_l2 evaluation: 1.260000e+02
        dummy evaluation: 0.000000e+00
    INFO: Forward-backward method
    Iteration 1 of forward_backward:
        norm_l2 evaluation: 1.400000e+01
        dummy evaluation: 0.000000e+00
        objective = 1.40e+01
    Iteration 2 of forward_backward:
        norm_l2 evaluation: 2.963739e-01
        dummy evaluation: 0.000000e+00
        objective = 2.96e-01
    Iteration 3 of forward_backward:
        norm_l2 evaluation: 7.902529e-02
        dummy evaluation: 0.000000e+00
        objective = 7.90e-02
    Iteration 4 of forward_backward:
        norm_l2 evaluation: 5.752265e-02
        dummy evaluation: 0.000000e+00
        objective = 5.75e-02
    Iteration 5 of forward_backward:
        norm_l2 evaluation: 5.142032e-03
        dummy evaluation: 0.000000e+00
        objective = 5.14e-03
    Solution found after 5 iterations:
        objective function f(sol) = 5.142032e-03
        stopping criterion: ATOL

    Verify the stopping criterion (should be smaller than atol=1e-2):

    >>> np.linalg.norm(ret['sol'] - y)**2  # doctest:+ELLIPSIS
    0.00514203...

    Show the solution (should be close to y w.r.t. the L2-norm measure):

    >>> ret['sol']
    array([ 4.02555301,  5.03194126,  6.03832952,  7.04471777])

    Show the used solver:

    >>> ret['solver']
    'forward_backward'

    Show some information about the convergence:

    >>> ret['crit']
    'ATOL'
    >>> ret['niter']
    5
    >>> ret['time']  # doctest:+SKIP
    0.0012578964233398438
    >>> ret['objective']  # doctest:+NORMALIZE_WHITESPACE,+ELLIPSIS
    [[126.0, 0], [13.99999999..., 0], [0.29637392..., 0], [0.07902528..., 0],
    [0.05752265..., 0], [0.00514203..., 0]]

    """

    if verbosity not in ['NONE', 'LOW', 'HIGH', 'ALL']:
        raise ValueError('Verbosity should be either NONE, LOW, HIGH or ALL.')

    # Add a second dummy convex function if only one function is provided.
    if len(functions) < 1:
        raise ValueError('At least 1 convex function should be provided.')
    elif len(functions) == 1:
        functions.append(dummy())
        if verbosity in ['LOW', 'HIGH', 'ALL']:
            print('INFO: Dummy objective function added.')

    # Choose a solver if none provided.
    if not solver:
        if len(functions) == 2:
            fb0 = 'GRAD' in functions[0].cap(x0) and \
                  'PROX' in functions[1].cap(x0)
            fb1 = 'GRAD' in functions[1].cap(x0) and \
                  'PROX' in functions[0].cap(x0)
            dg0 = 'PROX' in functions[0].cap(x0) and \
                  'PROX' in functions[1].cap(x0)
            
        if verbosity in ['LOW', 'HIGH', 'ALL']:
            name = solver.__class__.__name__
            print('INFO: Selected solver: {}'.format(name))

    # Set solver and functions verbosity.
    translation = {'ALL': 'HIGH', 'HIGH': 'HIGH', 'LOW': 'LOW', 'NONE': 'NONE'}
    solver.verbosity = translation[verbosity]
    translation = {'ALL': 'HIGH', 'HIGH': 'LOW', 'LOW': 'NONE', 'NONE': 'NONE'}
    functions_verbosity = []
    for f in functions:
        functions_verbosity.append(f.verbosity)
        f.verbosity = translation[verbosity]

    tstart = time.time()
    crit = None
    niter = 0
    objective = [[f.eval(x0) for f in functions]]
    rtol_only_zeros = True

    # Solver specific initialization.
    solver.pre(functions, x0)

    while not crit:

        niter += 1

        if xtol is not None:
            last_sol = np.array(solver.sol, copy=True)

        if verbosity in ['HIGH', 'ALL']:
            name = solver.__class__.__name__
            print('Iteration {} of {}:'.format(niter, name))


        ### Here add a part to improve the distribution regarding the prior
        #
        #
        #
        if prior is not None:
            W = spatial.distance.squareform(solver.sol)
            L = np.diag(W @ np.ones(W.shape[0]))-W
            pinv_L = np.linalg.pinv(L)
            pinv_prior = np.linalg.pinv(prior)
            sqrt_pinv_prior = np.real(sqrtm(pinv_prior))
            sqrt_prior = np.real(sqrtm(prior))
            #new_L = step_prior*
            S_t = sqrt_prior@np.linalg.matrix_power((step_prior)*pinv_prior+(1-step_prior)*np.real(sqrtm(sqrt_pinv_prior@pinv_L@sqrt_pinv_prior)), 2)@sqrt_prior
            new_L = np.linalg.pinv(S_t)
            solver.sol = L_to_vec_w(new_L)
        elif prior_Frob is not None:
            W = spatial.distance.squareform(solver.sol)
            L = np.diag(W @ np.ones(W.shape[0]))-W
            new_L = step_prior*prior_Frob+(1-step_prior)*L
            solver.sol = L_to_vec_w(new_L)





        # Solver iterative algorithm.
        solver.algo(objective, niter)

        objective.append([f.eval(solver.sol) for f in functions])
        current = np.sum(objective[-1])
        last = np.sum(objective[-2])

        # Verify stopping criteria.
        if atol is not None and current < atol:
            crit = 'ATOL'
        if dtol is not None and np.abs(current - last) < dtol:
            crit = 'DTOL'
        if rtol is not None:
            div = current  # Prevent division by 0.
            if div == 0:
                if verbosity in ['LOW', 'HIGH', 'ALL']:
                    print('WARNING: (rtol) objective function is equal to 0 !')
                if last != 0:
                    div = last
                else:
                    div = 1.0  # Result will be zero anyway.
            else:
                rtol_only_zeros = False
            relative = np.abs((current - last) / div)
            if relative < rtol and not rtol_only_zeros:
                crit = 'RTOL'
        if xtol is not None:
            err = np.linalg.norm(solver.sol - last_sol)
            err /= np.sqrt(last_sol.size)
            if err < xtol:
                crit = 'XTOL'
        if maxit is not None and niter >= maxit:
            crit = 'MAXIT'

        if verbosity in ['HIGH', 'ALL']:
            print('    objective = {:.2e}'.format(current))

    # Restore verbosity for functions. In case they are called outside solve().
    for k, f in enumerate(functions):
        f.verbosity = functions_verbosity[k]

    if verbosity in ['LOW', 'HIGH', 'ALL']:
        print('Solution found after {} iterations:'.format(niter))
        print('    objective function f(sol) = {:e}'.format(current))
        print('    stopping criterion: {}'.format(crit))

    # Returned dictionary.
    result = {'sol':       solver.sol,
              'solver':    solver.__class__.__name__,  # algo for consistency ?
              'crit':      crit,
              'niter':     niter,
              'time':      time.time() - tstart,
              'objective': objective}
    try:
        # Update dictionary for primal-dual solvers
        result['dual_sol'] = solver.dual_sol
    except AttributeError:
        pass

    # Solver specific post-processing (e.g. delete references).
    solver.post()

    return result


class solver(object):
    r"""
    Defines the solver object interface.

    This class defines the interface of a solver object intended to be passed
    to the :func:`pyunlocbox.solvers.solve` solving function. It is intended to
    be a base class for standard solvers which will implement the required
    methods. It can also be instantiated by user code and dynamically modified
    for rapid testing. This class also defines the generic attributes of all
    solver objects.

    Parameters
    ----------
    step : float
        The gradient-descent step-size. This parameter is bounded by 0 and
        :math:`\frac{2}{\beta}` where :math:`\beta` is the Lipschitz constant
        of the gradient of the smooth function (or a sum of smooth functions).
        Default is 1.
    accel : pyunlocbox.acceleration.accel
        User-defined object used to adaptively change the current step size
        and solution while the algorithm is running. Default is a dummy
        object that returns unchanged values.

    """

    def __init__(self, step=1., accel=None):
        if step < 0:
            raise ValueError('Step should be a positive number.')
        self.step = step
        self.accel = acceleration.dummy() if accel is None else accel

    def pre(self, functions, x0):
        """
        Solver-specific pre-processing. See parameters documentation in
        :func:`pyunlocbox.solvers.solve` documentation.

        Notes
        -----
        When preprocessing the functions, the solver should split them into
        two lists:
        * `self.smooth_funs`, for functions involved in gradient steps.
        * `self.non_smooth_funs`, for functions involved proximal steps.
        This way, any method that takes in the solver as argument, such as the
        methods in :class:`pyunlocbox.acceleration.accel`, can have some
        context as to how the solver is using the functions.

        """
        self.sol = np.asarray(x0)
        self.smooth_funs = []
        self.non_smooth_funs = []
        self._pre(functions, self.sol)
        self.accel.pre(functions, self.sol)

    def _pre(self, functions, x0):
        raise NotImplementedError("Class user should define this method.")

    def algo(self, objective, niter):
        """
        Call the solver iterative algorithm and the provided acceleration
        scheme. See parameters documentation in
        :func:`pyunlocbox.solvers.solve`

        Notes
        -----
        The method :meth:`self.accel.update_sol` is called before
        :meth:`self._algo` because the acceleration schemes usually involves
        some sort of averaging of previous solutions, which can add some
        unwanted artifacts on the output solution. With this ordering, we
        guarantee that the output of solver.algo is not corrupted by the
        acceleration scheme.

        Similarly, the method :meth:`self.accel.update_step` is called after
        :meth:`self._algo` to allow the step update procedure to act directly
        on the solution output by the underlying algorithm, and not on the
        intermediate solution output by the acceleration scheme in
        :meth:`self.accel.update_sol`.

        """
        self.sol[:] = self.accel.update_sol(self, objective, niter)
        self.step = self.accel.update_step(self, objective, niter)
        self._algo()

    def _algo(self):
        raise NotImplementedError("Class user should define this method.")

    def post(self):
        """
        Solver-specific post-processing. Mainly used to delete references added
        during initialization so that the garbage collector can free the
        memory. See parameters documentation in
        :func:`pyunlocbox.solvers.solve`.

        """
        self._post()
        self.accel.post()
        del self.sol, self.smooth_funs, self.non_smooth_funs

    def _post(self):
        raise NotImplementedError("Class user should define this method.")

class primal_dual(solver):
    r"""
    Parent class of all primal-dual algorithms.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.solver` base class.

    Parameters
    ----------
    L : function or ndarray, optional
        The transformation L that maps from the primal variable space to the
        dual variable space. Default is the identity, :math:`L(x)=x`. If `L` is
        an ``ndarray``, it will be converted to the operator form.
    Lt : function or ndarray, optional
        The adjoint operator. If `Lt` is an ``ndarray``, it will be converted
        to the operator form. If `L` is an ``ndarray``, default is the
        transpose of `L`. If `L` is a function, default is `L`,
        :math:`Lt(x)=L(x)`.
    d0: ndarray, optional
        Initialization of the dual variable.

    """

    def __init__(self, L=None, Lt=None, d0=None, *args, **kwargs):
        super(primal_dual, self).__init__(*args, **kwargs)

        if L is None:
            self.L = lambda x: x
        else:
            if callable(L):
                self.L = L
            else:
                # Transform matrix form to operator form.
                self.L = lambda x: L.dot(x)

        if Lt is None:
            if L is None:
                self.Lt = lambda x: x
            elif callable(L):
                self.Lt = L
            else:
                self.Lt = lambda x: L.T.dot(x)
        else:
            if callable(Lt):
                self.Lt = Lt
            else:
                self.Lt = lambda x: Lt.dot(x)

        self.d0 = d0

    def _pre(self, functions, x0):
        # Dual variable.
        if self.d0 is None:
            self.dual_sol = self.L(x0)
        else:
            self.dual_sol = self.d0

    def _post(self):
        self.d0 = None
        del self.dual_sol


class mlfbf(primal_dual):
    r"""
    Monotone+Lipschitz Forward-Backward-Forward primal-dual algorithm.

    This algorithm solves convex optimization problems with objective of the
    form :math:`f(x) + g(Lx) + h(x)`, where :math:`f` and :math:`g` are proper,
    convex, lower-semicontinuous functions with easy-to-compute proximity
    operators, and :math:`h` has Lipschitz-continuous gradient with constant
    :math:`\beta`.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.primal_dual` base class.

    Notes
    -----
    The order of the functions matters: set :math:`f` first on the list,
    :math:`g` second, and :math:`h` third.

    This algorithm requires the first two functions to implement the
    :meth:`pyunlocbox.functions.func.prox` method, and the third function to
    implement the :meth:`pyunlocbox.functions.func.grad` method.

    The step-size should be in the interval :math:`\left] 0, \frac{1}{\beta +
    \|L\|_{2}}\right[`.

    See :cite:`komodakis2015primaldual`, Algorithm 6, for details.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
    >>> y = np.array([294, 390, 361])
    >>> L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
    >>> x0 = np.zeros(len(y))
    >>> f = functions.dummy()
    >>> f._prox = lambda x, T: np.maximum(np.zeros(len(x)), x)
    >>> g = functions.norm_l2(lambda_=0.5)
    >>> h = functions.norm_l2(y=y, lambda_=0.5)
    >>> max_step = 1/(1 + np.linalg.norm(L, 2))
    >>> solver = solvers.mlfbf(L=L, step=max_step/2.)
    >>> ret = solvers.solve([f, g, h], x0, solver, maxit=1000, rtol=0)
    Solution found after 1000 iterations:
        objective function f(sol) = 1.833865e+05
        stopping criterion: MAXIT
    >>> ret['sol']
    array([ 1.,  1.,  1.])

    """

    def _pre(self, functions, x0):
        super(mlfbf, self)._pre(functions, x0)

        if len(functions) != 3:
            raise ValueError('MLFBF requires 3 convex functions.')

        self.non_smooth_funs.append(functions[0])   # f
        self.non_smooth_funs.append(functions[1])   # g
        self.smooth_funs.append(functions[2])       # h

    def _algo(self):
        # Forward steps (in both primal and dual spaces)
        y1 = self.sol - self.step * (self.smooth_funs[0].grad(self.sol) +
                                     self.Lt(self.dual_sol))
        y2 = self.dual_sol + self.step * self.L(self.sol)

        # Backward steps (in both primal and dual spaces)
        p1 = self.non_smooth_funs[0].prox(y1, self.step)
        p2 = _prox_star(self.non_smooth_funs[1], y2, self.step)

        # Forward steps (in both primal and dual spaces)
        q1 = p1 - self.step * (self.smooth_funs[0].grad(p1) + self.Lt(p2))
        q2 = p2 + self.step * self.L(p1)

        # Update solution (in both primal and dual spaces)
        self.sol[:] = self.sol - y1 + q1
        self.dual_sol[:] = self.dual_sol - y2 + q2


class projection_based(primal_dual):
    r"""
    Projection-based primal-dual algorithm.

    This algorithm solves convex optimization problems with objective of the
    form :math:`f(x) + g(Lx)`, where :math:`f` and :math:`g` are proper,
    convex, lower-semicontinuous functions with easy-to-compute proximity
    operators.

    See generic attributes descriptions of the
    :class:`pyunlocbox.solvers.primal_dual` base class.

    Parameters
    ----------
    lambda_ : float, optional
        The update term weight. It should be between 0 and 2. Default is 1.

    Notes
    -----
    The order of the functions matters: set :math:`f` first on the list, and
    :math:`g` second.

    This algorithm requires the two functions to implement the
    :meth:`pyunlocbox.functions.func.prox` method.

    The step-size should be in the interval :math:`\left] 0, \infty \right[`.

    See :cite:`komodakis2015primaldual`, Algorithm 7, for details.

    Examples
    --------
    >>> import numpy as np
    >>> from pyunlocbox import functions, solvers
    >>> y = np.array([294, 390, 361])
    >>> L = np.array([[5, 9, 3], [7, 8, 5], [4, 4, 9], [0, 1, 7]])
    >>> x0 = np.array([500, 1000, -400])
    >>> f = functions.norm_l1(y=y)
    >>> g = functions.norm_l1()
    >>> solver = solvers.projection_based(L=L, step=1.)
    >>> ret = solvers.solve([f, g], x0, solver, maxit=1000, rtol=None, xtol=.1)
    Solution found after 996 iterations:
        objective function f(sol) = 1.045000e+03
        stopping criterion: XTOL
    >>> ret['sol']
    array([0, 0, 0])

    """

    def __init__(self, lambda_=1., *args, **kwargs):
        super(projection_based, self).__init__(*args, **kwargs)
        self.lambda_ = lambda_

    def _pre(self, functions, x0):
        super(projection_based, self)._pre(functions, x0)

        if self.lambda_ <= 0 or self.lambda_ > 2:
            raise ValueError('Lambda is bounded by 0 and 2.')

        if len(functions) != 2:
            raise ValueError('projection_based requires 2 convex functions.')

        self.non_smooth_funs.append(functions[0])   # f
        self.non_smooth_funs.append(functions[1])   # g

    def _algo(self):
        a = self.non_smooth_funs[0].prox(self.sol - self.step *
                                         self.Lt(self.dual_sol), self.step)
        ell = self.L(self.sol)
        b = self.non_smooth_funs[1].prox(ell + self.step * self.dual_sol,
                                         self.step)
        s = (self.sol - a) / self.step + self.Lt(ell - b) / self.step
        t = b - self.L(a)
        tau = np.sum(s**2) + np.sum(t**2)
        if tau == 0:
            self.sol[:] = a
            self.dual_sol[:] = self.dual_sol + (ell - b) / self.step
        else:
            theta = self.lambda_ * (np.sum((self.sol - a)**2) / self.step +
                                    np.sum((ell - b)**2) / self.step) / tau
            self.sol[:] = self.sol - theta * s
            self.dual_sol[:] = self.dual_sol - theta * t


def weight2degmap(N, array=False):
    r"""
    Generate linear operator K such that W @ 1 = K @ vec(W).

    Parameters
    ----------
    N : int
        Number of nodes on the graph

    Returns
    -------
    K : function
        Operator such that K(w) is the vector of node degrees
    Kt : function
        Adjoint operator mapping from degree space to edge weight space
    array : boolean, optional
        Indicates if the maps are returned as array (True) or callable (False).

    Examples
    --------
    >>> import learn_graph
    >>> K, Kt = learn_graph.weight2degmap(10)

    Notes
    -----
    Used in :func:`learn_graph.log_degree_barrier method`.

    """
    import numpy as np

    Ne = int(N * (N - 1) / 2)  # Number of edges
    row_idx1 = np.zeros((Ne, ))
    row_idx2 = np.zeros((Ne, ))
    count = 0
    for i in np.arange(1, N):
        row_idx1[count: (count + (N - i))] = i - 1
        row_idx2[count: (count + (N - i))] = np.arange(i, N)
        count = count + N - i
    row_idx = np.concatenate((row_idx1, row_idx2))
    col_idx = np.concatenate((np.arange(0, Ne), np.arange(0, Ne)))
    vals = np.ones(len(row_idx))
    K = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(N, Ne))
    if array:
        return K, K.transpose()
    else:
        return lambda w: K.dot(w), lambda d: K.transpose().dot(d)


def plot_objectives(objective, labels=None, fig=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm

    objective = np.asarray(objective)

    n = objective.shape[0]
    try:
        m = objective.shape[1]
        obj = objective
    except IndexError:
        m = 1
        obj = np.reshape(objective, (n, 1))

    if labels is None:
        labels = []
        for i in range(m):
            labels.append('obj. ' + str(i + 1))
    assert m == len(labels), "Must have same number of labels as obj. fun."

    if fig is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax = fig.axes[-1]

    fig.set_figheight(6)
    fig.set_figwidth(12)

    stride = 1. / (m - 1) if m > 1 else 1.
    cmap = matplotlib.cm.get_cmap('viridis')

    for i in range(m):
        color = cmap(i * stride)
        ax.plot(obj[:, i], color=color, linewidth=2, label=labels[i])

    ax.legend(loc='best')
    ax.set_xlim([-1, n + 1])
    ax.set_ylim([np.min(obj) - 1, np.max(obj) + 1])

    return fig, ax