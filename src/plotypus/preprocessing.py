"""
Light curve space transformation preprocessors for regressing upon.
"""
import numpy
from numpy import pi
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from .utils import autocorrelation, rowvec

__all__ = [
    'Fourier'
]


class Fourier():
    r"""
    Transforms observed data from phase-space to Fourier-space.

    In order to represent a light curve as a Fourier series of the form

    .. math::
        m(t) = A_0 + \sum_{k=1}^n (a_k \sin(k \omega t) + b_k \cos(k \omega t)),

    phased time observations are transformed into a design matrix
    :math:`\mathbf{X}` by :func:`Fourier.design_matrix`, such that linear
    regression can be used to solve for coefficients

    .. math::
        \hat{b} = \begin{bmatrix}
                    A_0    \\
                    a_1    \\
                    b_1    \\
                    \vdots \\
                    a_n    \\
                    b_n
                  \end{bmatrix}

    in the matrix equation

    .. math::
        \mathbf{X} \hat{b} = \hat{y}

    where :math:`\vec{y}` is the vector of observed magnitudes

    .. math::
        \hat{y} = \begin{bmatrix}
                    m_0    \\
                    m_1    \\
                    \vdots \\
                    m_n
                  \end{bmatrix}

    If *degree_range* is not None, *degree* is selected via
    :func:`baart_criteria`. Otherwise the provided *degree* is used.

    **Parameters**

    degree : positive int, optional
        Degree of Fourier series to use, assuming *degree_range* is None
        (default 3).
    degree_range : 2-tuple or None, optional
        Range of allowed *degree*\s to search via :func:`baart_criteria`, or
        None if single provided *degree* is to be used (default None).
    regressor : object with "fit" and "transform" methods, optional
        Regression object used for fitting light curve when selecting *degree*
        via :func:`baart_criteria`. Not used otherwise
        (default
        ``sklearn.linear_model.LinearRegression(fit_intercept=False)``).
    """
    def __init__(self, degree=3, degree_range=None,
                 regressor=LinearRegression(fit_intercept=False)):
        self.degree = degree
        self.degree_range = degree_range
        self.regressor = regressor

    def fit(self, X, y=None):
        """
        Sets *self.degree* according to :func:`baart_criteria` if *degree_range*
        is not None, otherwise does nothing.

        **Parameters**

        X : array-like, shape = [n_samples, 1]
            Column vector of phases.
        y : array-like or None, shape = [n_samples], optional
            Row vector of magnitudes (default None).

        **Returns**

        self : returns an instance of self
        """
        if self.degree_range is not None:
            self.degree = self.baart_criteria(X, y)
        return self

    def transform(self, X, y=None, **params):
        """
        Transforms *X* from phase-space to Fourier-space, returning the design
        matrix produced by :func:`Fourier.design_matrix` for input to a
        regressor.

        **Parameters**

        X : array-like, shape = [n_samples, 1]
            Column vector of phases.
        y : None, optional
            Unused argument for conformity (default None).

        **Returns**

        design_matrix : array-like, shape = [n_samples, 2*degree+1]
            Fourier design matrix produced by :func:`Fourier.design_matrix`.
        """
        data = numpy.dstack((numpy.array(X).T[0], range(len(X))))[0]
        phase, order = data[data[:,0].argsort()].T
        design_matrix = self.design_matrix(phase, self.degree)
        return design_matrix[order.argsort()]

    def get_params(self, deep=False):
        """
        Get parameters for this preprocessor.

        **Parameters**

        deep : boolean, optional
            Only here for scikit-learn compliance. Ignore it (default False).

        **Returns**

        params : dict
            Mapping of parameter name to value.
        """
        return {'degree': self.degree}

    def set_params(self, **params):
        """
        Set parameters for this preprocessor.

        **Returns**

        self : returns an instance of self
        """
        if 'degree' in params:
            self.degree = params['degree']

        return self

    def baart_criteria(self, X, y):
        """
        Returns the optimal Fourier series degree as determined by
        `Baart's Criteria <http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1986A%26A...170...59P&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf>`_ [JOP]_.

        **Citations**

        .. [JOP] J. O. Petersen, 1986,
                 "Studies of Cepheid type variability. IV.
                 The uncertainties of Fourier decomposition parameters.",
                 A&A, Vol. 170, p. 59-69
        """
        try:
            min_degree, max_degree = self.degree_range
        except ValueError:
            raise ValueError("Degree range must be a length two sequence")

        cutoff = self.baart_tolerance(X)
        pipeline = Pipeline([('Fourier', Fourier()),
                             ('Regressor', self.regressor)])
        sorted_X = numpy.sort(X, axis=0)
        X_sorting = numpy.argsort(rowvec(X))
        for degree in range(min_degree, max_degree):
            pipeline.set_params(Fourier__degree=degree)
            pipeline.fit(X, y)
            lc = pipeline.predict(sorted_X)
            residuals = y[X_sorting] - lc
            p_c = autocorrelation(residuals)
            if abs(p_c) <= cutoff:
                return degree
        # reached max_degree without reaching cutoff
        return max_degree

    @staticmethod
    def baart_tolerance(X):
        r"""
        Returns the autocorrelation cutoff of *X* for :func:`baart_criteria`,
        as given by

        .. math::

            \frac{1}{\sqrt{2 (\operatorname{card}(\mathbf{X}) - 1)}}


        **Parameters**

        X : array-like, shape = [n_samples, 1]
            Column vector of phases

        **Returns**


        """
        return (2 * (len(X) - 1))**(-1/2)

    @staticmethod
    def design_matrix(phases, degree):
        r"""
        Constructs an :math:`N \times 2n+1` matrix of the form:

        .. math::

            \begin{bmatrix}
              1
            & \sin(1 \cdot 2\pi \cdot \phi_0)
            & \cos(1 \cdot 2\pi \cdot \phi_0)
            & \ldots
            & \sin(n \cdot 2\pi \cdot \phi_0)
            & \cos(n \cdot 2\pi \cdot \phi_0)
            \\
              \vdots
            & \vdots
            & \vdots
            & \ddots
            & \vdots
            & \vdots
            \\
              1
            & \sin(1 \cdot 2\pi \cdot \phi_N)
            & \cos(1 \cdot 2\pi \cdot \phi_N)
            & \ldots
            & \sin(n \cdot 2\pi \cdot \phi_N)
            & \cos(n \cdot 2\pi \cdot \phi_N)
            \end{bmatrix}

        where :math:`n =` *degree*, :math:`N =` *n_samples*, and
        :math:`\phi_i =` *phases[i]*.

        Parameters
        ----------
        phases : array-like, shape = [n_samples]

        """
        n_samples = phases.size
        # initialize coefficient matrix
        M = numpy.empty((n_samples, 2*degree+1))
        # indices
        i = numpy.arange(1, degree+1)
        # initialize the Nxn matrix that is repeated within the
        # sine and cosine terms
        x = numpy.empty((n_samples, degree))
        # the Nxn matrix now has N copies of the same row, and each row is
        # integer multiples of pi counting from 1 to the degree
        x[:,:] = i*2*numpy.pi
        # multiply each row of x by the phases
        x.T[:,:] *= phases
        # place 1's in the first column of the coefficient matrix
        M[:,0]    = 1
        # the odd indices of the coefficient matrix have sine terms
        M[:,1::2] = numpy.sin(x)
        # the even indices of the coefficient matrix have cosine terms
        M[:,2::2] = numpy.cos(x)
        return M

    @staticmethod
    def phase_shifted_coefficients(amplitude_coefficients, form='cos',
                                   shift=0.0):
        r"""
        Converts Fourier coefficients from the amplitude form to the
        phase-shifted form, as either a sine or cosine series.

        Amplitude form:

        .. math::
            m(t) = A_0 + \sum_{k=1}^n (a_k \sin(k \omega t)
                                     + b_k \cos(k \omega t))

        Sine form:

        .. math::
            m(t) = A_0 + \sum_{k=1}^n A_k \sin(k \omega t + \Phi_k)

        Cosine form:

        .. math::
            m(t) = A_0 + \sum_{k=1}^n A_k \cos(k \omega t + \Phi_k)

        **Parameters**

        amplitude_coefficients : array-like, shape = [:math:`2n+1`]
            Array of coefficients
            :math:`[ A_0, a_1, b_1, \ldots a_n, b_n ]`.
        form : str, optional
            Form of output coefficients, must be one of 'sin' or 'cos'
            (default 'cos').
        shift : number, optional
            Shift to apply to light curve (default 0.0).

        **Returns**

        out : array-like, shape = [:math:`2n+1`]
            Array of coefficients
            :math:`[ A_0, A_1, \Phi_1, \ldots, A_n, \Phi_n ]`.
        """
        if form != 'sin' and form != 'cos':
            raise NotImplementedError(
                'Fourier series must have form sin or cos')

        # separate array of coefficients into respective parts
        A_0 = amplitude_coefficients[0]
        a_k = amplitude_coefficients[1::2]
        b_k = amplitude_coefficients[2::2]

        degree = a_k.size
        k = numpy.arange(1, degree+1)
        # A_k and Phi_k are the angle and hypotenuse in the right triangles
        # pictured below. A_k is obtained with the Pythagorean theorem, and
        # Phi_k is obtained with the 2-argument inverse tangent.
        # The positions of a_k and b_k depend on whether it is a sin or cos
        # series.
        #
        # Cos series                Sin series
        #
        #    b_k                          /|
        # ---------                      / |
        # \ Φ_k |_|                     /  |
        #  \      |                A_k /   |
        #   \     |                   /    | b_k
        #    \    | a_k              /     |
        # A_k \   |                 /     _|
        #      \  |                / Φ_k | |
        #       \ |                ---------
        #        \|                   a_k
        #
        A_k   = numpy.sqrt(a_k**2 + b_k**2)
        # phase coefficients are shifted to the left by optional ``shift``
        if form == 'cos':
            Phi_k = numpy.arctan2(-a_k, b_k) + 2*pi*k*shift
        elif form == 'sin':
            Phi_k = numpy.arctan2(b_k, a_k) + 2*pi*k*shift
        # constrain Phi between 0 and 2*pi
        Phi_k %= 2*pi

        phase_shifted_coefficients_ = numpy.empty(amplitude_coefficients.shape,
                                                  dtype=float)
        phase_shifted_coefficients_[0]    = A_0
        phase_shifted_coefficients_[1::2] = A_k
        phase_shifted_coefficients_[2::2] = Phi_k

        return phase_shifted_coefficients_

    @staticmethod
    def fourier_ratios(phase_shifted_coeffs):
        r"""
        Returns the :math:`R_{j1}` and :math:`\phi_{j1}` values for the given
        phase-shifted coefficients.

        .. math::

            R_{j1} = A_j / A_1

        .. math::

            \phi_{j1} = \phi_j - j \phi_1

        **Parameters**

        phase_shifted_coeffs : array-like, shape = [:math:`2n+1`]
            Fourier sine or cosine series coefficients.
            :math:`[ A_0, A_1, \Phi_1, \ldots, A_n, \Phi_n ]`.

        **Returns**

        out : array-like, shape = [:math:`2n+1`]
            Fourier ratios
            :math:`[ R_{21}, \phi_{21}, \ldots, R_{n1}, \phi_{n1} ]`.
        """


        n_coeff = phase_shifted_coeffs.size
        # n_coeff = 2*degree + 1 => degree = (n_coeff-1)/2
        degree = (n_coeff - 1) / 2

        amplitudes = phase_shifted_coeffs[1::2]
        phases = phase_shifted_coeffs[2::2]

        # there are degree-1 amplitude ratios, and degree-1 phase deltas,
        # so altogether there are 2*(degree-1) values
        ratios = numpy.empty(2*(degree-1), dtype=float)
        amplitude_ratios = ratios[::2]
        phase_deltas = ratios[1::2]

        # amplitudes may be zero, so suppress division by zero warnings
        with numpy.errstate(divide="ignore"):
            amplitude_ratios[:] = amplitudes[1:]
            amplitude_ratios   /= amplitudes[0]

        # indices for phase deltas
        i = numpy.arange(2, degree+1)
        phase_deltas[:] = phases[1:]
        phase_deltas   -= i*phases[0]
        # constrain phase_deltas between 0 and 2*pi
        phase_deltas   %= 2*pi

        return ratios
