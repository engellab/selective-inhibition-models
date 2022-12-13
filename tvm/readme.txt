MatLab code to run 2 variable selective inhibition decision making model.

Start with example script for making post quickly.

The functions dsdt_noAMPA, ds1dt_noAMPA, ds2dt_noAMPA are for finding fixed points and null clines.

The function noampa_jac determines the stability of the fixed points.

The script manifolds finds the stable and unstable manifolds, only run after example so that parameters are defined.

Function gen_alphas calculates the circuit parameters alpha1, alpha2, and I0E1(2)

Function get_nulls_fps2 finds fixed point, null clines, and vector field

Function single_sim_run returns the simulation on a single trial

Function gen_psychometric will simulate a number or trials for a provided set of stimulus strengths.