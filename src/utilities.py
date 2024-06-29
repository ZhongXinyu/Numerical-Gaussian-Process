####################################################################################################
# This function should be manually added to the ConjugatePosteriors class in the gpjax/gps.py file.

def predict_with_prev_cov(
    self,
    test_inputs: Num[Array, "N D"],
    train_data: Dataset,
    prev_cov: Num[Array, "N N"],
) -> GaussianDistribution:
    r"""Query the predictive posterior distribution with a previous covariance matrix.

    Conditional on a training data set, compute the GP's posterior
    predictive distribution for a given set of parameters, which takes into consideration of the previous covariance matrix passed in as a parameter.
    The returned function can be evaluated at a set of test inputs to compute the corresponding
    predictive density.

    The predictive distribution of a conjugate GP with previous covariance is given by
    \begin{equation}
        \mu^n = q^T K^{-1} \begin{bmatrix} u_n^a \\ \mu_{n-1}^a \end{bmatrix}
    \end{equation}

    \begin{equation}
        \Sigma^{n,a}(x^n, x^*) = k_u^n(x^n, x^*) - q^T K^{-1} q + q^T K^{-1} \begin{bmatrix} 0 & 0 \\ 0 & \Sigma_{n-1}^{a,a} \end{bmatrix} K^{-1} q
    \end{equation}

    The conditioning set is a GPJax `Dataset` object, whilst predictions
    are made on a regular Jax array.

    Args:
        test_inputs (Num[Array, "N D"]): A Jax array of test inputs at which the
            predictive distribution is evaluated.
        train_data (Dataset): A `gpx.Dataset` object that contains the input and
            output data used for training dataset.
        prev_cov (Num[Array, "N N"]): The covariance matrix of the previous posterior distribution.

    Returns
    -------
        GaussianDistribution: A function that accepts an input array and
            returns the predictive distribution as a `GaussianDistribution`.
    """
    # Unpack training data
    x, y = train_data.X, train_data.y

    # Unpack test inputs
    t = test_inputs

    # mx = mean(x)
    mx = self.prior.mean_function(x)

    # Precompute Gram matrix, Kxx, at training inputs, x
    Kxx = self.prior.kernel.gram(x)
    Kxx += cola.ops.I_like(Kxx) * self.jitter

    Sigma = Kxx
    Sigma = cola.PSD(Sigma)

    mean_t = self.prior.mean_function(t)
    Ktt = self.prior.kernel.gram(t)
    Kxt = self.prior.kernel.cross_covariance(x, t)
    Sigma_inv_Kxt = cola.solve(Sigma, Kxt)

    # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
    mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

    # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
    covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
    covariance += cola.ops.I_like(covariance) * self.prior.jitter
    covariance = cola.PSD(covariance)

    # Get the number of training and test data
    n_test = t.shape[0]
    n_boundary = x.shape[0] - n_test

    # A matrix with right bottom corner as the previous covariance
    Kxx_zero = jnp.zeros((n_boundary, n_boundary))
    Kxt_zero = jnp.zeros((n_boundary, n_test))
    Ktx_zero = jnp.zeros((n_test, n_boundary))
    
    # add the term that is dependent on the previous covariance
    prev_cov = jnp.block([[prev_cov, Ktx_zero],
                        [Kxt_zero, Kxx_zero]])

    covariance += Sigma_inv_Kxt.T @ prev_cov @ Sigma_inv_Kxt

    return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)
