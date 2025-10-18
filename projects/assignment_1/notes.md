

# Test on steady robot


## Q1: baseline

The baseline controller (the one with the default weights) has decent tracking but definitely not perfect.
Error on the center of mass among the x and z axis are pretty minor (1mm and 0.3mm at most) but the error on the y axis 
(the one with the sine wave applied to it) is a bit wider at about 3cm at the peak as you can see in the image below

[!image](./images/3.1.png)

The most likely cause of the tracking error is that there are errors in the model make it so that there are some unknown forces,
that are not compensated for by the controller.
The most likely sources of this error are joint friction and dumping.
Other common sources of error are the motor's controller not being precise, as well as teh model of the robot (i.e. mass and
inertia matrix of the joint) not being perfect, however given that we are in a simulation this shouldn't be a problem.



## Q2: tuning the parameters

### Increasing the weight of the center of mass task.

In the first scenario (A) I increased the weight of the center of mass task from 1 to 10
and these are the results:
[!image](./images/3.2.1.png)

### Increasing the proportional coefficients of the center of mass task

In the second scenario (B) I increased the proportional coefficients of the center of mass
task from 10 to 100 and these are the results:
[!image](./images/3.2.2.png)


### Difference between the two.

**Initial behaveor**

We can note that the scenario A perform worst than the scenario B at the beginning,
where we can note a larger gap between the desired position and the actual position.

Note that we are only locking at the Y axis (Com1), as there are not significant
differences in the other two axis, as they are required to stay still.


**Long therm/Steady state behaveor**
Wen we let a few seconds pass we can note that scenario A start to perform better.
The differences are minor, but we can observe that the scenario A is able to keep the 
actual trajectory closer to the desired one, while with scenario B it looks like there 
is a minor delay between the desired trajectory and the actual one.

The reason why scenario A perform so poorly in the beginning, is that probably that
we are giving a bad control signal. As we can see in the velocity graph

[!image](./images/3.2.vel.png)

the robot start with a zero velocity, however we instantly request a "high" velocity.
The controller B immediately have an high error, that get multiply by a high proportional
gain, and therefore can get up to speed more quickly. However the controller A does a bit
works (due to the lower proportional gain).

In the real world I would assume that we would give a better control signal to the robot, and 
let the velocity start from zero at time zero, and therefore I would prefer the controller A.

If however we found ourself in a situation where the ostentatious change in speed where the norm
for some reason, then we may consider controller B.

## Q3: increasing sinewave frequency

in this third scenario we increase the sign-wave frequency
to one hertz and se how te two controllers (A and B) behave

### Controller A performance
here is the center of mass tracking for controller A
[!image](./images/3.3.1.png)

we can see how the tracking here is really poor, and we can see that there 
is a divergence from the desired trajectory and the actual one as times goes on.

We can understand why by locking at the position of the robot in the last step:
[!image](./images/3.3.1.robot.png)

we can see how the robot is tiled sideways, and this can be attributed to the fact
that we have increase the weights of the center of mass task, therefore 
decreasing the relative weight of the postural task.
The controller is then ignoring the fact that the robot is not straight
and allowing it to tilt sideways, and this is resulting in a really bad tracking.

furthermore if we let the simulation run for a few more step we can see that the 
divergence increase, and at a certain point the robot is no longer able
to stand and fall on itself.

Needless to say that in this circumstance controller B would be preferred
over controller A


### Controller B performance

controller B also perform poorly. We can see in the chart that the
difference between the desired trajectory and the actual one is much higher
with respect to the lower frequency test.

This is simple to understand as, if the frequency of the center of mass increase 
the first derivative (the velocity) also increase. Then get higher velocity we 
need higher torques, and to get higher torques we need an higher error (assuming
we let the proportional coefficient unchanged)

[!image](./images/3.3.2.png)


# Test on waling robot

## Q1: Experiment on 15cm steps

### Tuning the postural task weight

At the first run the robot did not manage to stay standing, as it tilted sideways and ultimately fall.
(not dissimilar to what the controller A did in the previous step)
The solution was easy enough, updating the postural task from zero to some were between 1e-2 and 1e-3 
worked fairly well.

ultimately I saddled on 1e-3 and this is the tracking achieved:
[!image](./images/4.1.png)

So as you can see in the picture, the tracking is really good.

### Effect of the different tasks on the robot behaveor

To evaluate the effect that each task has on the robot's walking ability
I disabled each single task by setting the relative weight to zero, and observe how the robot
behave

**Setting the Center of mass task to zero**

By setting the center of mass task to zero we can observe the robot tilting forward
and ultimately falling. This is expected behaver, as if the robot's center of mass fall outside
the foot's projection on the terrain the system will become unstable and the robot will fall.
unless corrective actions are taken.


**Setting the angular momentum task to zero**

Setting the angular momentum task to zero did not result in significant changes in the robot behavior.

In theory the task should try to keep the angular momentum of the robot close to zero, by doing
things such as moving the left arm forward when the right leg is moving forward (not dissimilar
to what human do when running).
In theory this should result a robot that is easier to control, however in practice
we are moving at such a slow speed that this does not make a substation difference.

I also tried to set the weight higher and the results is a robot that moves the arms
extensively to balance the angular momentum, but is overall less table than the default
configuration as the high weight on the angular momentum task end up undermining the other
tasks.

**Setting the foot motion task to zero**

The foot motion task is perhaps the most intuitive one, If there is no control signal
that tels the robot to move the feet than it simply won't start walking.
And this intuition is confirmed with a simple run

**Setting the foot contact task to zero**

The foot contact task is another easy to explain one.
If we set the task weight to zero we can see the robot
moving downward in the beginning of the simulation as if it was
placed in a soft floor. The task is therefore necessary to guaranteed
the thoughtfulness of the simulation (otherwise we would see the robot 
with the feet inside the floor) and is therefore understandable that the 
weight of the task are so high


**Setting the joint posture task to zero**

The posture task is important as it incentivize the robot to keep a "natural" posture
instead of allowing it to do whatever it wants as long as it respect the other tasks.

Setting it to zero results in some really unnatural leg and arm movements, with the 
ultimate result of the robot falling. The problem here is that keeping
a natural posture allow the robot to be more flexible with the joint given
that they are far from singularity, as well as keeping the robot posture up straight
in a more stable position.


## Q2: Experiment on 30cm steps

The first run of the robot walking with a 30cm set size resulted in the 
robot tilting backward (as if it was doing the limbo), this and this
ultimately resulted in the robot falling.

To fix this issue I gradually increased the posture task weight
from 1e-3 all the way up to 5e-2.
After this minor change the robot was no longer tilting backward
but it started falling forward in the last step of the trajectory.

To fix this I set increased the weight of the center of mass task
from one to 10 and ultimately solved the issue.
