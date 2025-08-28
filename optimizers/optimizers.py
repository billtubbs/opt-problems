import numpy as np
import lpfgopt
import scipy

from sklearn.utils import check_random_state


def scipy_minimize_rand_init(
    fun,
    init_range,
    random_state=None,
    args=(),
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None
):
    """Wrapper to the scipy.optimize.minimize function to provide an additional
    keyword argument init_range that allows the initial point to be generated
    randomly.
    """

    # Generate random initial point using init_range
    rng = check_random_state(random_state)
    init_range = np.asarray(init_range)
    x0 = rng.uniform(init_range[:, 0], init_range[:, 1])

    return scipy.optimize.minimize(
        fun,
        x0,
        args=args,
        method=method,
        jac=jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        callback=callback,
        options=options
    )


def lpfgopt_minimize_rand_init(
    fun,
    bounds,
    args=(),
    points=20,
    fconstraint=None,
    discrete=None,
    maxit=10000,
    tol=1e-5,
    seedval=None,
    pointset=None,
    callback=None,
    cdll_ptr=None,
    init_range=None,
    **kwargs
):
    """Wrapper to the lpfgopt.minimize function to provide an additional
    keyword argument init_range that allows initial points to be generated
    randomly.
    """

    if discrete is None:
        discrete = []

    if init_range is not None:
        if pointset is not None:
            raise ValueError("provide either pointset or init_range, not both")

        # Generate random initial points using init_range
        rng = check_random_state(seedval)
        init_range = np.asarray(init_range)
        n = init_range.shape[0]
        pointset = rng.uniform(
            init_range[:, 0], init_range[:, 1], size=(points, n)
        )

    return lpfgopt.minimize(
        fun,
        bounds,
        args=args,
        points=points,
        fconstraint=fconstraint,
        discrete=discrete,
        maxit=maxit,
        tol=tol,
        seedval=seedval,
        pointset=pointset,
        callback=callback,
        cdll_ptr=cdll_ptr,
        **kwargs
    )
