# 3

## 3.1

Decent tracking but definitely not perfect.
Error on the center of mass among the x and z axis are pretty minor (1mm and 0.3mm at most) but the error on the z axis 
(the one with the sine wave applied to it) is a bit wider at about 3cm at the peak

The error in the position cah be seen in the image:

[!image](./images/3.1.png)

The most likely cause of the tracking error is that there are errors in the model make it so that there are some unknown forces,
that are not compensated for by the controller.
The most likely sources of this error are joint friction and dumping
Other common sources of error are the motor's controller not being precise, as well as teh model of the robot (i.e. mass and
inertia matrix of the joint) not being perfect, however given that we are in a simulation this shouldn't be a problem.



## 3.2
we try improove the tracking by either updating the weight of the center of mas task, or to update
the proportional gains of the center of mass, and see which one yields better results

### 3.2.1 updating the weight of the center of mass task

In the first scenario (A) I increased the weight of the center of mass task from 1 to 10
and these are the results:
[!image](./images/3.2.1.png)



### 3.2.2 updating the proportional coefficients of the center of mass task

In the second scenario (B) I increased the proportional coefficients of the center of mass
task from 10 to 100 and these are the results
[!image](./images/3.2.2.png)


### difference between the two.
We can note that the scenario A perform worst than the scenario B at the beginning,
where we can note a larger gap between the desired position and the actual position
(note that we are only locking at CoM2, aka the Y axis, as there are not significant
differences in the other two axis, as they are required to stay still)

however wen we let a few seconds pass we can note that scenario A start to perform better.
The differences are minor, but we can observe that the scenario A is able to keep the 
actual trajectory closer to the desired one, while with scenario B it looks like there 
is a minor delay between the desired trajectory and the actual one.

The reason why scenario A perform so poorly in the beginning, is that probably that
we are giving a bad control signal. As we can see in the velocity graph

[!image](./images/3.2.vel.png)

the robot start with a zero velocity, however we instantly request a "high" velocity.
The controller be immediately have an high error, that get multiply by a high proportional
gain, and therefore can get up to speed more quickly. However the controller A cant' do the same.

In the real world I would assume that we would give a better control signal to the robot, and 
let the velocity start from zero at time zero, and therefore I would prefer the controller A.

If however we found ourself in a situation where the ostentatious change in speed where the norm
for some reason, then we may consider controller B.

## 3.3 increasing sign-wave frequency

in this third scenario we increase the sign-wave frequency
to one hertz and se how te two controllers (A and B) behave

### 3.3.1 Controller A performance
here is the center of mass tracking for controller A
[!image](./images/3.3.1.png)

we can see how the tracking here is really poor, and we can see that there 
is a divergence from the desired trajectory and the actual one as times goes on.

We can understand why by locking at the position of the robot in the last step:
[!image](./images/3.3.1.robot.png)

we can see how the robot is tiled sideways, and ths can be attributed to the fact
that we have increase the weights of the center of mass task, therefore 
decreasing the relative weight of the postural task.
The controller is then ignoring the fact that the robot is not straight
and allowing it to tilt sideways, and this is resulting in a really bad tracking.

furthermore if we let the simulation run for a few more step we can see that the 
divergence increase, and at a certain point the robot is no longer able
to stand and fall on itself.

Needless to say that in this circumstance controller B would be preferred
over controller A


### 3.3.2 Controller B performance

controller B also perform poorly. We can see in the chart that the
difference between the desired trajectory and the actual one is much higher
with respect to the lower frequency test.

This is simple to understand as, if the frequency of the center of mass increase 
the first derivative (the velocity) also increase, to get higher velocity we 
need higher torques, and to get higher torques we need an higher error (assuming
we let the proportional coefficient unchanged)

[!image](./images/3.3.2.png)

