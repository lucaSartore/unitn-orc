

ex 2: the final position is not extrimelly similar, trying to increase the importance of the final weigth task result in the solver not converging


ex 3: the ciclical task is still not perfect, but a bit better. this time increasing the weights does result in a sizable improvement 

ex 4 part 1: the time actually does not decrease wrt the baseline (0.2)
however increasing the relative tasks's weight does decries the time to a more reasonable 0.14

by default the time is set to 0.24s
changing the weight of the time task from e-1 to e-0 decrease the time to 0.14

ex 4 part 2: following the original instructions provided by the sk+1 = sk + dt * wk result in having to 
set wi to 1/N/dt.
this lead to 2 issues:
 1) dt was initialized at 0, and the solver crashed for a division by zero in the frit iteration (solved easily
 by providing an initial condition)
 2) that division made the problem more complex, and resulted in way poorer solutions (with bad tracking, and time that was above the baseline of 0.2)
 to solve this changed the equations in:
 ....
 and we got good tracking with a substantial time reduction (0.06s)
