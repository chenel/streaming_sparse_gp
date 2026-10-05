import tensorflow as tf
import numpy as np

import gpflow
from gpflow.inducing_variables import InducingPoints
from gpflow.models import GPModel, InternalDataTrainingLossMixin
from gpflow import covariances


class SGPR_PEP(GPModel, InternalDataTrainingLossMixin):
    """
    Sparse GP regression using Power-EP.

    A unifying framework for Gaussian process pseudo-point
    approximations using Power Expectatin Propagation
    Thang D. Bui, Josiah Yan, Richard E. Turner
    JMLR 2017
    
    """

    def __init__(self, data, kernel, Z, alpha, mean_function=None,
                 noise_pull_dims=None, noise_floor=1e-2, noise_exponent=2.0,
                 noise_amplitude=None, fit_noise_exponent=False):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate gpflow objects

        This method only works with a Gaussian likelihood.
        """
        self.X, self.Y = self.data = gpflow.models.util.data_input_to_tensor(data)
        likelihood = gpflow.likelihoods.Gaussian()
        num_latent_gps = GPModel.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.inducing_variable = InducingPoints(Z)
        self.num_data = self.X.shape[0]

        self.alpha = alpha
        # Optional heteroscedastic noise; see _noise_variance().  Default None reproduces
        # the scalar-noise behaviour exactly.
        self.noise_pull_dims = noise_pull_dims
        self.noise_floor = noise_floor
        self.noise_exponent = noise_exponent
        # Per-row amplitude, supplied rather than fitted.  See _noise_variance().
        self.noise_amplitude = (None if noise_amplitude is None else
                                tf.constant(np.reshape(np.asarray(noise_amplitude,
                                                                  dtype=np.float64), [-1]),
                                            dtype=tf.float64))
        if noise_pull_dims is not None:
            self.noise_pull_weights = gpflow.Parameter(
                np.ones(len(noise_pull_dims)), transform=gpflow.utilities.positive())
            # One exponent per knob when fitted: the measured growth of the conditional
            # variance with |theta| is not 2 for every knob (2.6 for one of the three in
            # studies/weight-distributions/), and a shared value has to compromise.
            if fit_noise_exponent:
                self.noise_exponents = gpflow.Parameter(
                    np.full(len(noise_pull_dims), float(noise_exponent)),
                    transform=gpflow.utilities.positive())
            else:
                self.noise_exponents = None
            # The scalar likelihood variance plays no part once the noise is
            # per-observation; leaving it trainable would add a parameter the objective
            # never sees.
            gpflow.set_trainable(self.likelihood.variance, False)

    def _noise_variance(self):
        """Per-observation noise variance.

        A single scalar noise is provably wrong for reweighting targets.  At zero pulls
        every event's weight is exactly 1 by construction, so the conditional variance
        vanishes identically on that hyperplane and grows away from it; a constant cannot
        represent that.  Here

            sigma2_i = a_i * sum_j w_j * (noise_floor + |theta_ij| ** p_j)

        a_i is an optional per-row amplitude, supplied by the caller rather than fitted.
        It exists because this form is otherwise a function of the pulls alone, with no
        dependence on the observables at all -- and the observable dependence is the
        largest measured feature of the noise (see studies/weight-distributions/).  Being
        data rather than a parameter, it does not reintroduce the degeneracy described
        below: there is nothing for the optimizer to trade against w_j.  The caller is
        responsible for normalizing it, or the overall scale is split arbitrarily.

        one weight per systematic and nothing per observation: an observation's scale is
        set by its own pull vector, a known input rather than an estimated quantity.

        Note what this deliberately does NOT have: a separate overall noise level
        multiplying the sum.  Observations away from zero pull constrain only the product
        of such a level with the per-axis weights, so the split between them would be
        fixed by the zero-pull rows alone -- and those are rows whose residual is exactly
        zero by construction.  A parameter answerable only to rows the model already gets
        right is driven to zero, taking the fitted noise with it.  Folding the floor
        inside the sum welds the zero-pull variance to the per-axis weights, which the
        scattering observations hold up.

        noise_floor is therefore a dimensionless ratio, not a variance: it says how much
        more reliable a zero-pull row is than a unit-pull one.  It must stay strictly
        positive, or the log(Dff / sigma2) term in the objective diverges on those rows.

        The exponent is the leading behaviour of a variance that must vanish at the
        origin; exact where the response is linear in the pull, approximate otherwise.
        noise_exponent makes it explicit rather than implied, since nothing about the
        method requires it to be 2.
        """
        if self.noise_pull_dims is None:
            return self.likelihood.variance
        theta = tf.abs(tf.gather(self.X, self.noise_pull_dims, axis=1))
        p = (self.noise_exponent if self.noise_exponents is None
             else self.noise_exponents)
        # |theta| ** p is 0 at theta = 0, but its derivative with respect to p involves
        # log(theta), so a fitted exponent produces NaN gradients on exactly the rows
        # where theta vanishes -- the zero-pull anchors, thousands of them per chunk.
        # The doubled tf.where keeps the forward value and gives the gradient a finite
        # path: the inner one hides the zeros from tf.pow, the outer one restores them.
        nz = theta > 0.0
        safe = tf.where(nz, theta, tf.ones_like(theta))
        powed = tf.where(nz, tf.pow(safe, p), tf.zeros_like(theta))
        sigma2 = tf.reduce_sum(self.noise_pull_weights * (self.noise_floor + powed), axis=1)
        if self.noise_amplitude is not None:
            sigma2 = sigma2 * self.noise_amplitude
        return sigma2

    def _common_terms(self):
        Mb = self.inducing_variable.num_inducing
        # jitter = gpflow.default_jitter()
        jitter = gpflow.utilities.to_default_float(1e-4)
        sigma2 = self._noise_variance()
        alpha = self.alpha

        # b is inducing points
        # f is training points
        Kfdiag = self.kernel(self.X, full_cov=False)
        Kbf = covariances.Kuf(self.inducing_variable, self.kernel, self.X)
        Kbb = covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter)

        err = self.Y - self.mean_function(self.X)
        Lb = tf.linalg.cholesky(Kbb)
        Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True)

        Qff_diag = tf.reduce_sum(tf.square(Lbinv_Kbf), axis=0)
        Dff = sigma2 + alpha * (Kfdiag - Qff_diag)
        Lbinv_Kbf_LDff = Lbinv_Kbf / tf.sqrt(Dff)
        d1 = tf.matmul(Lbinv_Kbf_LDff, Lbinv_Kbf_LDff, transpose_b=True)
        D = tf.eye(Mb, dtype=d1.dtype) + d1
        LD = tf.linalg.cholesky(D)

        # err, not self.Y: the objective pairs -0.5*sum(err^2/Dff) against
        # +0.5*sum(LDinv_c^2), so both must be built from the same residual.  With a
        # zero mean function they agree and their divergences as Dff -> 0 cancel; with
        # a non-zero one, using raw Y here leaves the bound unbounded above, and the
        # optimizer drives the noise to its floor and the kernel variance to zero.
        Sinv_y = err / tf.reshape(Dff, [self.num_data, 1])
        c = tf.matmul(Lbinv_Kbf, Sinv_y)
        LDinv_c = tf.linalg.triangular_solve(LD, c, lower=True)

        return (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff)

    def maximum_log_likelihood_objective(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. 
        """

        sigma2 = self._noise_variance()
        N = self.num_data
        alpha = self.alpha

        # a is old inducing points, b is new
        # f is training points
        (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff) = self._common_terms()

        # constant term
        bound = -0.5 * N * np.log(2 * np.pi)
        # quadratic term
        bound += -0.5 * tf.reduce_sum(tf.square(err) / tf.reshape(Dff, [N, 1]))
        bound += 0.5 * tf.reduce_sum(tf.square(LDinv_c))

        # log det term
        bound += -0.5 * tf.reduce_sum(tf.math.log(Dff))
        bound += - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))

        # trace-like term
        bound += - 0.5 * (1 - alpha) / alpha * \
            tf.reduce_sum(tf.math.log(Dff / sigma2))

        return bound

    def predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. 
        """
        # jitter = gpflow.default_jitter()
        jitter = gpflow.utilities.to_default_float(1e-4)

        # b is inducing points
        # f is training points
        # s is test points
        Kbs = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        (Kbf, Kbb, Lb, D, LD, LDinv_c, err, Dff) = self._common_terms()

        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, LDinv_c, transpose_a=True)

        if full_cov:
            Kss = self.kernel(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=gpflow.default_float())
            var1 = Kss
            var2 = - tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
            var3 = tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_Kbs, transpose_a=True)
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(Xnew, full_cov=False) + jitter
            var2 = -tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
            var3 = tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            var = var1 + var2 + var3

        return mean + self.mean_function(Xnew), var
