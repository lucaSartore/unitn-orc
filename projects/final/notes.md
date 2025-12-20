# step one: OCP

problem with bias toward the solution in zero:

initializing the state variable vector with random value helped remove this bias,
but solution still went at random in one of the 3 pits.

solution: take 10 run with the same input, and pick the one with the lowest score,
in this way the issue was solved (or at least highly mitigated)

# step two: critic

critic nn was selected by gradually increasing the size of the network
until we saw some diminishing return on the improvement in accuracy.
(we also made a train-test split to verify that we were not over-fitting)

with this process we obtained a network with approximately 2000 parameters (1 x 30 x 30 x 30 x 1)
for the critic of the simple system. and a network of approximately 5000 parameters
(2 x 50 x 50 x 50 x 1) for the system with inertia.

For the training we choose hubble loss, the reason we did not use something more traditional
as L2 norm (i.e. MSE) is because of the 3 pits issue 
this resulted in a final loss of approximately 2-5 and 7-11 for the two systems respectively


the optimizer we chose was AdamW with a learning rate of 5e-4.

