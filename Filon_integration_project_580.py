import jax
import jax.numpy as jnp
# from Phi_A78_jax import phi_A78 # Only if you are using this in your f(x) function
from jax.scipy.linalg import solve_triangular
jax.config.update("jax_enable_x64", True)

e = 1
a0 = 5.29177210903e-11
HartE = 27.21139

''' 
Using Filon method to integrate f(x) alongside the approximation for the bessel function 
larger than x = 18.0711 of the J0(x). That is where the asymtotic relation works very well.
'''

# ---------- analytic integrals ----------
# Integrand: polynomial p(x) = c0 + c1 x + c2 x^2 + c3 x^3
# We compute definite integral of p(x) * cos(k x + phi) from xa to xb.
# Use closed-form antiderivatives for x^n cos(kx+phi) (n=0..3).
# For sin integrals, reuse cos integrator by shifting phi -> phi - pi/2.
def poly_cos_definite(c, xa, xb, k, phi):
    """
    Definite integral of polynomial (coeffs c[0..3]) times cos(kx+phi) on [xa,xb].
    Computed directly by evaluating antiderivative at both endpoints and subtracting
    to avoid catastrophic cancellation by computing differences directly.
    """
    c0, c1, c2, c3 = c[0], c[1], c[2], c[3]
    invk = 1.0 / k
    invk2 = invk * invk
    invk3 = invk2 * invk
    invk4 = invk3 * invk
    
    # Evaluate trig functions at both endpoints - This can be updated to work in the t-basis
    sa = jnp.sin(k * xa + phi)
    ca = jnp.cos(k * xa + phi)
    sb = jnp.sin(k * xb + phi)
    cb = jnp.cos(k * xb + phi)

    # Changes variables to evaluate polynomial fit with better condition number 
    xa = 0
    xb = 1
    
    # Compute definite integral by computing [F(xb) - F(xa)] directly
    # This avoids catastrophic cancellation since we compute differences in place
    # ∫ c0 cos: [c0/k * sin]_a^b
    I0 = c0 * invk * (sb - sa)
    
    # ∫ c1 * x * cos: [c1 * (x*sin/k + cos/k²)]_a^b
    I1 = c1 * (xb * sb * invk + cb * invk2 - xa * sa * invk - ca * invk2)
    
    # ∫ c2 * x² * cos: [c2 * (x²*sin/k + 2*x*cos/k² - 2*sin/k³)]_a^b
    I2 = c2 * (xb**2 * sb * invk + 2*xb*cb*invk2 - 2*sb*invk3 
             - xa**2 * sa * invk - 2*xa*ca*invk2 + 2*sa*invk3)
    
    # ∫ c3 * x³ * cos: [c3 * (x³*sin/k + 3*x²*cos/k² - 6*x*sin/k³ - 6*cos/k⁴)]_a^b
    I3 = c3 * (xb**3 * sb * invk + 3*xb**2*cb*invk2 - 6*xb*sb*invk3 - 6*cb*invk4
             - xa**3 * sa * invk - 3*xa**2*ca*invk2 + 6*xa*sa*invk3 + 6*ca*invk4)
    
    return I0 + I1 + I2 + I3

def poly_sin_definite(c, xa, xb, k, phi):
    """Definite integral of polynomial times sin(kx+phi) on [xa,xb].
       Use cos integral with shifted phase: sin(kx+phi) = cos(kx + phi - pi/2).
    """
    return poly_cos_definite(c, xa, xb, k, phi - jnp.pi / 2.0)


# rescaling the QR decomposition to give us a better covariance matrix
def fit_cubic_rescaled_qr(xs, ys):
    # Set up t values for the t-basis
    t = jnp.linspace(0, 1 ,len(xs))

    # computes X everytime, however in the t basis this can be created only once (Future work)
    X = jnp.stack([jnp.ones_like(t), t, t**2, t**3], axis=1)   # (n,4)

    # QR decomposition (faster than other methods) - Make only once (future work)
    Q, R = jnp.linalg.qr(X, mode='reduced')
    
    # Solve for coefficents
    c = solve_triangular(R, Q.T @ ys)
    return c

# ---------- one-period integrator ----------
def one_period_contribution(f, x0, period, m, phi, n_samples):
    """
    Compute contribution from [x0, x0 + period]:
      I = ∫ g(x) cos(m x + phi) dx + ∫ h(x) sin(m x + phi) dx,
    where f returns a pair (g(x), h(x)) when called.
    n_samples: number of sample points in period (>=4). default 5.
    """
    # sample points (include both endpoints); use n_samples evenly spaced
    xs = x0 + jnp.linspace(0.0, period, n_samples)  # shape (n_samples,)

    # evaluate f to get g and h - The y-axis components   
    gh = f(xs)  # expect shape (n_samples, 2) or tuple; we'll support returning tuple
    
    # seperates the components
    if isinstance(gh, tuple) or isinstance(gh, list):
        g_vals = gh[0]
        h_vals = gh[1]
    else:
        # assume array shape (n_samples, 2)
        g_vals = gh[:, 0]
        h_vals = gh[:, 1]

    # Finds te cubic fit in the t-basis
    cg = fit_cubic_rescaled_qr(xs, g_vals)  # (4,)
    ch = fit_cubic_rescaled_qr(xs, h_vals)  # (4,)

    # analytic integrals - Bounds only needed for the sin and cosine part
    xa = x0
    xb = x0 + period
    Ic = poly_cos_definite(cg, xa, xb, m, phi)
    Is = poly_sin_definite(ch, xa, xb, m, phi)
    return Ic + Is

# ---------- vectorized integrator for multiple b values ----------
def compute_integral(b, x0, m, phi, n_samples, f_gh):
    """
    Compute the Filon integration from x0 to b.
    All parameters are traced (vmap-compatible).
    """
    # determine period length
    period = 2 * jnp.pi / m

    # Determine integer number of periods
    n_periods = jnp.floor((b - x0) / period).astype(jnp.int64)
    
    # Integral of a single period
    def one_period(x_start):
        return one_period_contribution(f_gh, x_start, period, m, phi, n_samples)
    
    # Function to sum up all of the periods
    def body_fun(i, acc):
        x_start = x0 + i * period
        I = one_period(x_start)
        return acc + I
    
    # needed if you don't want jax.jit notation
    # val = 0
    # for i in range(0, n_periods):
    #     val = body_fun(i, val)
    # return val    

    # Goes through and integrates every period and sums them up
    return jax.lax.fori_loop(0, n_periods, body_fun, 0.0)


# values in front of the cos approximation
def A_of_x(x):
    # three-term asymptotic A(x)
    return jnp.sqrt(2.0 / (jnp.pi * x)) * (1 - 9.0 / (128.0 * x**2) + 75.0 / (1024.0 * x**4))

# values in fron of the sin approximation
def B_of_x(x):
    return jnp.sqrt(2.0 / (jnp.pi * x)) * (-1.0 / (8.0 * x) + 75.0 / (3072.0 * x**3) - 3675.0 / (393216.0 * x**5))


def make_f_gh(r, zee, zhh, params):
    def f_gh(x):
        f_base_value = 1 #-e/(2*jnp.pi)*phi_A78(zee, zhh, x, params)*x # Enter function here <---------
        g = f_base_value * A_of_x(x*r)
        h = f_base_value * B_of_x(x*r)
        return (g, h)
    return f_gh


def filon_integration(r, zee, zhh, params, q_up):
    f_gh_func = make_f_gh(r, zee, zhh, params)

    # parameters
    phi = -jnp.pi / 4.0
    x0 = 18.07106 / r     # start at 6th zero
    n_samples = 5

    # Choose ONE value of b
    num_periods = 1000
    b = x0 + num_periods*2*jnp.pi/r + 0.1/r    # for example one full period away

    # JIT the single integrator
    compute_integral_single = jax.jit(
        lambda b: compute_integral(b, x0, r, phi, n_samples, f_gh_func))

    # Warm-up
    _ = compute_integral_single(b)

    # Compute the integral for one b
    I_single = compute_integral_single(b)
    print("Integral for one b =", I_single)


# ---------- Example usage ----------
if __name__ == "__main__":
    r = 600 #40e-9/a0
    qend = 10
    zee = 1
    zhh = 1
    params = jnp.array([6.15, 2.8, 6.41, 7.4, 0.111, 1.91/HartE])
    filon_integration(r, zee, zhh, params, qend)
