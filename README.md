# Physics580
This is a repository for Physics 580 final project

In this repository there are a few files that are used to help determine the image potential for the schrodinger equation. This project uses the filon method (https://en.wikipedia.org/wiki/Filon_quadrature) to solve for the integral analytically to better get results for fast oscillations.

Code Dependencies:

pip install jax

# Project Poster

[![Preview of the PDF](Carter%20Shirley%20Poster%20Resized%2036x48.png)](Carter%20Shirley%20Poster%20Resized%2036x48.pdf)


# Phi_A78_jax.py 
This is a file that makes up the equations that are mentioned inside the Muljarov paper equations 7-8 in the Appendix thus the name A78. This paper talks about image charges in a perovskite superlattices. Here is a link to the paper so that anybody who is interested in the paper can referance it here. 

Muljarov, E. A., Tikhodeev, S. G., Gippius, N. A., & Ishihara, T. (1995). Excitons in self-organized semiconductor/insulator superlattices: PbI-based perovskite compounds. Physical Review B, 51(20), 14370.

The program contains one function that looks over different boundaries through a jax condition statement. and gives back a result depending on the value of z and z0 which is a variable passed into the function.

# Filon_integration_project_580.py
This file contains most of the entire project.

The order of code usage goes as followed:
 - Filon_integration()
   - make_f_gh()
     - f_gh()                         # enter desired function here
       - A_of_x()                     # Bessel function approximation
       - B_of_x()                     # Bessel function approximation
   - compute_integral()               # base function to compute the integral
     - body_fun()                     # sums up all of one_period()
       - one_period()                 # calls one_period_contribution
         - one_period_contribution()  # backbone to the analytic solution for the integral 
         - fit_cubic_rescaled_qr()    # fits the data to a cubic fit rnging from 0 to 1
         - poly_cos_definite()        # performs the analytic soution for cos terms
         - poly_sin_definite()        # performs the analytic soution for sin terms













