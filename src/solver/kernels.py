"""
\page kernels.py

This page contains the implementation of the kernels used in the GP model. The kernels are used to model the covariance between the data points in the GP model.
"""

# functions that define the kernels used in the GP model
import jax
import jax.numpy as jnp
import gpjax as gpx

from dataclasses import dataclass
from gpjax.base import param_field, static_field
from jaxtyping import (
    Float,
)
from gpjax.typing import (
    Array,
    ScalarFloat,
)

@dataclass
class arcSin(gpx.kernels.AbstractKernel):
    """
    The arcSin kernel is a non-stationary kernel that is used as an alternative to the RBF kernel.
    
    The kernel is defined as:
    k(x, x') = 2/pi * arcsin(2 * (sigma_0^2 + sigma^2 * x * x')/sqrt((1 + 2 * (sigma_0^2 + sigma^2 * x * x)) * (1 + 2* (sigma_0^2 + sigma^2 * x' * x')))

    Methods:
    __call__: Calculate the kernel value
    dX: Calculate the derivative of the kernel with respect to X
    dXp: Calculate the derivative of the kernel with respect to Xp
    dX_dXp: Calculate the derivative of the kernel with respect to X and Xp
    dX_dX: Calculate the derivative of the kernel with respect to X twice
    dXp_dXp: Calculate the derivative of the kernel with respect to Xp twice
    dXp2_dX: Calculate the derivative of the kernel with respect to Xp twice and X
    dX2_dXp: Calculate the derivative of the kernel with respect to X twice and Xp
    dX2_dXp2: Calculate the derivative of the kernel with respect to X twice and Xp twice

    """

    # define parameters
    sigma_0: ScalarFloat = param_field(jnp.array(2.0), trainable = True)
    sigma: ScalarFloat = param_field(jnp.array(2.0), trainable = True)
    
    # @jax.jit
    def __call__(
        self, 
        X: Float[Array, "1 D"], 
        Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """
        Calculate the kernel value at X and Xp.
        """
        # calculate the kernel
        k = (2/jnp.pi * jnp.arcsin(2 * (self.sigma_0**2 + self.sigma**2 * X * Xp)/
                                jnp.sqrt((1 + 2 * (self.sigma_0**2 + self.sigma **2 * X * X)) * (1 + 2* (self.sigma_0**2 + self.sigma **2 * Xp * Xp))))).squeeze()
        return (k).squeeze()
        
    def dX(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        """
        Calculate the derivative of the kernel with respect to X.
        """
        numerator = 4 * self.sigma**2 * (Xp - 2 * (X - Xp) * self.sigma_0**2)
        term1 = 1 + 2 * X**2 * self.sigma**2 + 2 * self.sigma_0**2
        term2 = 1 + 2 * Xp**2 * self.sigma**2 + 2 * self.sigma_0**2
        denominator = jnp.pi * (term1)**(3/2) * jnp.sqrt(term2) * jnp.sqrt(
            1 - (4 * (X * Xp * self.sigma**2 + self.sigma_0**2)**2) / (term1 * term2)
        )
        return (numerator / denominator).squeeze()

    def dXp(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        """
        Calculate the derivative of the kernel with respect to Xp.
        """
        numerator = 4 * self.sigma**2 * (X + 2 * (X - Xp) * self.sigma_0**2)
        term1 = 1 + 2 * X**2 * self.sigma**2 + 2 * self.sigma_0**2
        term2 = 1 + 2 * Xp**2 * self.sigma**2 + 2 * self.sigma_0**2
        denominator = jnp.pi * jnp.sqrt(term1) * (term2)**(3/2) * jnp.sqrt(
            1 - (4 * (X * Xp * self.sigma**2 + self.sigma_0**2)**2) / (term1 * term2)
        )
        return (numerator / denominator).squeeze()
    
    def dX_dXp(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        """
        Calculate the derivative of the kernel with respect to X and Xp.
        """
        numerator = 4 * self.sigma**2 * (1 + 4 * self.sigma_0**2)
        
        # Calculate the terms used in the denominator
        term1 = 1 + 2 * X**2 * self.sigma**2 + 2 * self.sigma_0**2
        term2 = 1 + 2 * Xp**2 * self.sigma**2 + 2 * self.sigma_0**2
        adjustment = 1 + 2 * X**2 * self.sigma**2 + 2 * Xp**2 * self.sigma**2 + 4 * (1 + X**2 * self.sigma**2 - 2 * X * Xp * self.sigma**2 + Xp**2 * self.sigma**2) * self.sigma_0**2
        
        # Calculate the root terms
        root1 = jnp.sqrt(term1)
        root2 = jnp.sqrt(term2)
        term3 = 1 - (4 * (X * Xp * self.sigma**2 + self.sigma_0**2)**2) / (term1 * term2)
        root3 = jnp.sqrt(term3)

        # Combine everything into the denominator
        denominator = jnp.pi * root1 * root2 * adjustment * root3

        # Return the calculated derivative
        return (numerator / denominator).squeeze()

    def dX_dX(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        """
        Calculate the derivative of the kernel with respect to X twice.
        """

        # Calculate components of the numerator
        term1 = -X * Xp * self.sigma**2 * (3 + 6 * X**2 * self.sigma**2 + 4 * Xp**2 * self.sigma**2)
        term2 = (-1 + 8 * X**4 * self.sigma**4 - 24 * X**3 * Xp * self.sigma**4 - 2 * X * Xp * self.sigma**2 * (9 + 8 * Xp**2 * self.sigma**2) + 2 * X**2 * (self.sigma**2 + 12 * Xp**2 * self.sigma**4)) * self.sigma_0**2
        term3 = (2 * (-3 + 8 * X**4 * self.sigma**4 - 24 * X**3 * Xp * self.sigma**4 - 4 * X * Xp * self.sigma**2 * (3 + 2 * Xp**2 * self.sigma**2) + 4 * X**2 * (self.sigma**2 + 6 * Xp**2 * self.sigma**4)) * self.sigma_0**4 - 8 * self.sigma_0**6)
        numerator = 8 * self.sigma**2 * (term1 + term2 + term3)

        # Calculate components of the denominator
        denom_term1 = 1 + 2 * X**2 * self.sigma**2 + 2 * self.sigma_0**2
        denom_term2 = 1 + 2 * Xp**2 * self.sigma**2 + 2 * self.sigma_0**2
        adjustment = 1 + 2 * X**2 * self.sigma**2 + 2 * Xp**2 * self.sigma**2 + 4 * (1 + X**2 * self.sigma**2 - 2 * X * Xp * self.sigma**2 + Xp**2 * self.sigma**2) * self.sigma_0**2
        correction_factor = 1 - (4 * (X * Xp * self.sigma**2 + self.sigma_0**2)**2) / (denom_term1 * denom_term2)
        
        # Combine terms for the full denominator
        denominator = jnp.pi * (denom_term1**2.5) * jnp.sqrt(denom_term2) * adjustment * jnp.sqrt(correction_factor)
        
        return (numerator / denominator).squeeze()
    
    def dXp_dXp(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        """
        Calculate the derivative of the kernel with respect to Xp twice.
        """
        
        # Components of the numerator
        part1 = X * Xp * self.sigma**2 * (3 + 4 * X**2 * self.sigma**2 + 6 * Xp**2 * self.sigma**2)
        part2 = (1 - 2 * Xp**2 * self.sigma**2 + 16 * X**3 * Xp * self.sigma**4 - 24 * X**2 * Xp**2 * self.sigma**4 - 8 * Xp**4 * self.sigma**4 + 6 * X * Xp * self.sigma**2 * (3 + 4 * Xp**2 * self.sigma**2)) * self.sigma_0**2
        part3 = (2 * (3 - 4 * Xp**2 * self.sigma**2 + 8 * X**3 * Xp * self.sigma**4 - 24 * X**2 * Xp**2 * self.sigma**4 - 8 * Xp**4 * self.sigma**4 + 12 * X * (Xp * self.sigma**2 + 2 * Xp**3 * self.sigma**4)) * self.sigma_0**4 + 8 * self.sigma_0**6)
        numerator = -8 * self.sigma**2 * (part1 + part2 + part3)

        # Components of the denominator
        denom_term1 = 1 + 2 * X**2 * self.sigma**2 + 2 * self.sigma_0**2
        denom_term2 = 1 + 2 * Xp**2 * self.sigma**2 + 2 * self.sigma_0**2
        adjustment = 1 + 2 * X**2 * self.sigma**2 + 2 * Xp**2 * self.sigma**2 + 4 * (1 + X**2 * self.sigma**2 - 2 * X * Xp * self.sigma**2 + Xp**2 * self.sigma**2) * self.sigma_0**2
        correction_factor = 1 - (4 * (X * Xp * self.sigma**2 + self.sigma_0**2)**2) / (denom_term1 * denom_term2)
        
        # Combine terms for the full denominator
        denominator = jnp.pi * jnp.sqrt(denom_term1) * (denom_term2**(5/2)) * adjustment * jnp.sqrt(correction_factor)

        # Calculate and return the derivative
        return (numerator / denominator).squeeze()
    
    def dXp2_dX(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        """
        Calculate the derivative of the kernel with respect to Xp twice and X.
        """
        
        # Calculation of the numerator
        numerator = -24 * self.sigma**4 * (1 + 4 * self.sigma_0**2) * (Xp - 2 * (X - Xp) * self.sigma_0**2)

        # Components of the denominator
        term1 = 1 + 2 * X**2 * self.sigma**2 + 2 * self.sigma_0**2
        term2 = 1 + 2 * Xp**2 * self.sigma**2 + 2 * self.sigma_0**2
        adjustment = 1 + 2 * X**2 * self.sigma**2 + 2 * Xp**2 * self.sigma**2 + 4 * (1 + X**2 * self.sigma**2 - 2 * X * Xp * self.sigma**2 + Xp**2 * self.sigma**2) * self.sigma_0**2
        correction_factor = 1 - (4 * (X * Xp * self.sigma**2 + self.sigma_0**2)**2) / (term1 * term2)
        
        # Combine terms for the full denominator
        denominator = jnp.pi * jnp.sqrt(term1) * jnp.sqrt(term2) * adjustment**2 * jnp.sqrt(correction_factor)

        # Calculate and return the derivative
        return (numerator / denominator).squeeze()

    def dX2_dXp(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        """
        Calculate the derivative of the kernel with respect to X twice and Xp.
        """
        
        # Components of the numerator
        numerator = -24 * self.sigma**4 * (1 + 4 * self.sigma_0**2) * (X + 2 * (X - Xp) * self.sigma_0**2)

        # Components of the denominator
        term1 = 1 + 2 * X**2 * self.sigma**2 + 2 * self.sigma_0**2
        term2 = 1 + 2 * Xp**2 * self.sigma**2 + 2 * self.sigma_0**2
        adjustment = 1 + 2 * X**2 * self.sigma**2 + 2 * Xp**2 * self.sigma**2 + 4 * (1 + X**2 * self.sigma**2 - 2 * X * Xp * self.sigma**2 + Xp**2 * self.sigma**2) * self.sigma_0**2
        correction_factor = 1 - (4 * (X * Xp * self.sigma**2 + self.sigma_0**2)**2) / (term1 * term2)
        
        # Combine terms for the full denominator
        denominator = jnp.pi * jnp.sqrt(term1) * jnp.sqrt(term2) * adjustment**2 * jnp.sqrt(correction_factor)

        # Calculate and return the derivative
        return (numerator / denominator).squeeze()
    
    def dX2_dXp2(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        """
        Calculate the derivative of the kernel with respect to X twice and Xp twice.
        """
        # Calculation of the numerator
        part1 = -5 * X * Xp * self.sigma**2
        part2 = (-1 + 8 * X**2 * self.sigma**2 - 20 * X * Xp * self.sigma**2 + 8 * Xp**2 * self.sigma**2) * self.sigma_0**2
        part3 = 4 * (-1 + 4 * X**2 * self.sigma**2 - 8 * X * Xp * self.sigma**2 + 4 * Xp**2 * self.sigma**2) * self.sigma_0**4
        numerator = -48 * self.sigma**4 * (1 + 4 * self.sigma_0**2) * (part1 + part2 + part3)

        # Components of the denominator
        term1 = 1 + 2 * X**2 * self.sigma**2 + 2 * self.sigma_0**2
        term2 = 1 + 2 * Xp**2 * self.sigma**2 + 2 * self.sigma_0**2
        adjustment = 1 + 2 * X**2 * self.sigma**2 + 2 * Xp**2 * self.sigma**2 + 4 * (1 + X**2 * self.sigma**2 - 2 * X * Xp * self.sigma**2 + Xp**2 * self.sigma**2) * self.sigma_0**2
        correction_factor = 1 - (4 * (X * Xp * self.sigma**2 + self.sigma_0**2)**2) / (term1 * term2)
        
        # Combine terms for the full denominator
        denominator = jnp.pi * jnp.sqrt(term1) * jnp.sqrt(term2) * adjustment**3 * jnp.sqrt(correction_factor)

        return (numerator / denominator).squeeze()


@dataclass(frozen=False)
class WaveKernel(gpx.kernels.AbstractKernel):
    """
    The WaveKernel is a kernel that is used to solve 2D wave equation using the GP model, with RBFs as base kernels for each dimension.

    The kernel is defined as:
        k(x, x') = k_u(x, x') + k_v(x, x') 
    where:
        k_u(x, x') = k_u(x_1, x_1') + k_u(x_2, x_2')
        k_v(x, x') = k_v(x_1, x_1') + k_v(x_2, x_2')

    Methods:

    __call__: Calculate the kernel value
    """


    # define the kernel for the u and v values
    kernel_u: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0])
    kernel_v: gpx.kernels.AbstractKernel = gpx.kernels.RBF(active_dims=[0])

    # define the delta_t parameter
    delta_t: ScalarFloat = param_field(jnp.array(0.001), trainable = False)
    
    def __call__(
        self, 
        X: Float[Array, "1 D"], 
        Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """
        Calculate the kernel value at X and Xp.
        """
        # use t to track the time
        # i.e. t = 0 is when we are looking at the n_1th time
        # and t = 1 is when we are looking at the nth time
        t = jnp.array(X[1], dtype=int)
        tp = jnp.array(Xp[1], dtype=int)

        # use d to track the spatial dimension
        # i.e. d = 0 is when we are looking at the 1st dimension
        # and d = 1 is when we are looking at the 2nd dimension
        d = jnp.array(X[2], dtype=int)
        dp = jnp.array(Xp[2], dtype=int)

        # only use the first element of the array in calculation
        X = jnp.array(X[0:1],dtype=jnp.float64)
        Xp = jnp.array(Xp[0:1],dtype=jnp.float64) 

        # evaluate the kernel
        k_u = self.kernel_u(X, Xp)
        k_v = self.kernel_v(X, Xp)

        # gradient of the kernel
        # We do not need the gradient of the kernel for the current implementation
        
        # hessian of the kernel
        hess_kernel_u = k_u * ((X-Xp)**2/self.kernel_u.lengthscale**4 - 1/self.kernel_u.lengthscale**2)
        
        # hessian of the hessian of the kernel
        hess_hess_kernel_u = k_u * (3/self.kernel_u.lengthscale**4 - 6*(X - Xp)**2/self.kernel_u.lengthscale**6 + (X - Xp)**4/self.kernel_u.lengthscale**8)


        k_n_n_u_u = k_u + 1/4 * self.delta_t**2 * k_v
        k_n_n_u_v = (1/2 * self.delta_t * hess_kernel_u + 
                    1/2 * self.delta_t * k_v)
        k_n_n_v_u = (1/2 * self.delta_t * hess_kernel_u + 
                    1/2 * self.delta_t * k_v)
        
        k_n_n_1_u_u = k_u - 1/4 * self.delta_t**2 * k_v
        k_n_1_n_u_u = k_u - 1/4 * self.delta_t**2 * k_v
        
        k_n_n_1_u_v = (-1/2 * self.delta_t * hess_kernel_u + 
                    1/2 * self.delta_t * k_v)
        k_n_1_n_v_u = (-1/2 * self.delta_t * hess_kernel_u + 
                    1/2 * self.delta_t * k_v)
    
        k_n_n_1_v_u = (1/2 * self.delta_t * hess_kernel_u -
                    1/2 * self.delta_t * k_v)
        k_n_1_n_u_v = (1/2 * self.delta_t * hess_kernel_u -
                    1/2 * self.delta_t * k_v)
        
        k_n_n_v_v = k_v + 1/4 * self.delta_t**2+ hess_hess_kernel_u

        k_n_1_n_v_v = k_v - 1/4 * self.delta_t**2 * hess_hess_kernel_u
        k_n_n_1_v_v = k_v - 1/4 * self.delta_t**2 * hess_hess_kernel_u
        
        k_n_1_n_1_u_u = k_u + 1/4 * self.delta_t**2 * k_v
        k_n_1_n_1_u_v = (-1/2 * self.delta_t * hess_kernel_u -
                        1/2 * self.delta_t * k_v)
        k_n_1_n_1_v_u = (-1/2 * self.delta_t * hess_kernel_u -
                        1/2 * self.delta_t * k_v)
        k_n_1_n_1_v_v = k_v + 1/4 * self.delta_t**2 * hess_hess_kernel_u
        
        
        switch_n_n_u_u = jnp.where((t == 1) & (tp == 1) & (d == 0) & (dp == 0), 1, 0)
        
        switch_n_n_u_v = jnp.where((t == 1) & (tp == 1) & (d == 0) & (dp == 1), 1, 0)
        switch_n_n_v_u = jnp.where((t == 1) & (tp == 1) & (d == 1) & (dp == 0), 1, 0)

        switch_n_n_1_u_u = jnp.where((t == 1) & (tp == 0) & (d == 0) & (dp == 0), 1, 0)
        switch_n_1_n_u_u = jnp.where((t == 0) & (tp == 1) & (d == 0) & (dp == 0), 1, 0)

        switch_n_n_1_u_v = jnp.where((t == 1) & (tp == 0) & (d == 0) & (dp == 1), 1, 0)
        switch_n_1_n_v_u = jnp.where((t == 0) & (tp == 1) & (d == 1) & (dp == 0), 1, 0)
        
        switch_n_n_v_v = jnp.where((t == 1) & (tp == 1) & (d == 1) & (dp == 1), 1, 0)
        
        switch_n_n_1_v_u = jnp.where((t == 1) & (tp == 0) & (d == 1) & (dp == 0), 1, 0)
        switch_n_1_n_u_v = jnp.where((t == 0) & (tp == 1) & (d == 0) & (dp == 1), 1, 0)
        
        switch_n_n_1_v_v = jnp.where((t == 1) & (tp == 0) & (d == 1) & (dp == 1), 1, 0)
        switch_n_1_n_v_v = jnp.where((t == 0) & (tp == 1) & (d == 1) & (dp == 1), 1, 0)

        switch_n_1_n_1_u_u = jnp.where((t == 0) & (tp == 0) & (d == 0) & (dp == 0), 1, 0)
        switch_n_1_n_1_u_v = jnp.where((t == 0) & (tp == 0) & (d == 0) & (dp == 1), 1, 0)
        switch_n_1_n_1_v_u = jnp.where((t == 0) & (tp == 0) & (d == 1) & (dp == 0), 1, 0)
        switch_n_1_n_1_v_v = jnp.where((t == 0) & (tp == 0) & (d == 1) & (dp == 1), 1, 0)

        return_value = (k_n_n_u_u*switch_n_n_u_u+
                k_n_n_1_u_u*switch_n_n_1_u_u+
                k_n_1_n_u_u*switch_n_1_n_u_u+
                k_n_n_1_u_v*switch_n_n_1_u_v+
                k_n_1_n_v_u*switch_n_1_n_v_u+
                k_n_1_n_1_u_u*switch_n_1_n_1_u_u+
                k_n_1_n_1_u_v*switch_n_1_n_1_u_v+
                k_n_1_n_1_v_u*switch_n_1_n_1_v_u+
                k_n_1_n_1_v_v*switch_n_1_n_1_v_v+
                k_n_n_v_v*switch_n_n_v_v+
                k_n_n_v_u*switch_n_n_v_u+
                k_n_n_u_v*switch_n_n_u_v+
                k_n_n_1_v_u*switch_n_n_1_v_u+
                k_n_1_n_u_v*switch_n_1_n_u_v+
                k_n_n_1_v_v*switch_n_n_1_v_v+
                k_n_1_n_v_v*switch_n_1_n_v_v+
                k_n_n_1_v_u*switch_n_n_1_v_u
                ).squeeze()
        return return_value


