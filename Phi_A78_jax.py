import jax.numpy as jnp
from jax import jit, vmap, lax, config
from jax.numpy import sinh, cosh, arccosh, pi
import matplotlib.pyplot as plt
config.update("jax_enable_x64", True)

# from Muljarov_Potential import parameters

e = 1.0  # using float for JAX compatibility

# Here, we will find "phi(z,z0,q)" or equations A7 and A8 from the Muljarov paper
@jit
def phi_A78(z, z0, q, params):
    """JAX-compatible implementation of phi_A78"""
    ew, eb, lw, lb = params[0], params[1], params[2], params[3]

    # Define cases
    case1 = (z <= 0) & (z0 < 0)
    case2 = (z > 0) & (z0 >= 0)
    case3 = (((z <=0) & (z0 > 0)) | ((z0 <= 0) & (z > 0))) # Which is neither case1 nor case2

    # #----- Case 1: Both in well -----#
    def case1_branch(_):
    # if case1:
        #----- Define constants -----#

        n = eb/ew
        alpha = (1+n)/2
        beta = (1-n)/2
        
        rhs = (alpha**2*cosh(q*(lb+lw)) - beta**2*cosh(q*(lb-lw)))
        q0 = arccosh(rhs/n)

        t1 = -(2*pi*e/(ew*q))*sinh(q*jnp.abs(z-z0))
        t2 = (2*alpha*beta)*sinh(q*lb)*cosh(q*(z+z0+lw))
        t3 = (alpha**2*sinh(q*(lb+lw)) + beta**2*sinh(q*(lb-lw)))*cosh(q*(z-z0))
        pf = (2*pi*e/(eb*q*sinh(q0)))
        p = t1 + pf*(t2 + t3)
        return p
    
    #----- Case 2: Both in barrier -----#
    def case2_branch(_):
        # elif case2:
        #----- Make substitutions -----#

        lbs, lws, ews, ebs = lw, lb, eb, ew
        
        n = ebs/ews
        alpha = (1+n)/2
        beta = (1-n)/2
        
        rhs = (alpha**2*cosh(q*(lbs+lws)) - beta**2*cosh(q*(lbs-lws)))
        q0 = arccosh(rhs/n)
        
        t1 = -(2*pi*e/(ews*q))*sinh(q*abs(-z+z0))
        t2 = (2*alpha*beta)*sinh(q*lbs)*cosh(q*(-z-z0+lws))
        t3 = (alpha**2*sinh(q*(lbs+lws)) + beta**2*sinh(q*(lbs-lws)))*cosh(q*(-z+z0))
        pf = (2*pi*e/(ebs*q*sinh(q0)))

        p = t1 + pf*(t2 + t3)
        return p

    #----- Case 3: Different layers -----#
    def case3_branch(_):
        # elif (((z <=0) & (z0 > 0)) | ((z0 <= 0) & (z > 0))):

        #----- Define constants -----# 
        n = eb/ew
        alpha = (1+n)/2
        beta = (1-n)/2

        rhs = (alpha**2*cosh(q*(lb+lw)) - beta**2*cosh(q*(lb-lw)))
        q0 = arccosh(rhs/n)

        pf = (2*pi*e/(eb*q*sinh(q0)))
        t1 = alpha*sinh(q*(lb+lw))*cosh(q*(z-z0))
        t2 = beta*sinh(q*(lb-lw))*cosh(q*(z+z0))
        t3 = beta*((2*alpha**2)*sinh(q*lb)*sinh(q*lw) - sinh(q0))*sinh(q*(z+z0))
        t4 = alpha*((2*beta**2)*sinh(q*lb)*sinh(q*lw) - sinh(q0))*sinh(q*abs(z-z0))

        p = pf*(t1 + t2 + t3 + t4)
        return p
            

    # This is like a nested if statement or switch case
    p = lax.cond(
        case1,
        lambda _: case1_branch(_),
        lambda _: lax.cond(
            case2,
            lambda _: case2_branch(_),
            lambda _: case3_branch(_),
            operand=None
        ),
        operand=None
    )
    return p
    

