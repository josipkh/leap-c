# Active steering of a vehicle using linear MPC

This example implements a linear MPC for the active steering of an autonomous vehicle. The goal is to track a desired trajectory (i.e., stay close to the road centerline and keep the right heading), defined by the upcoming road curvature.

The example uses a Frenet-Serret coordinate frame, based on a simplified presentation given in Section 2.5 of [R. Rajamani - Vehicle Dynamics and Control](https://link.springer.com/book/10.1007/978-1-4614-1433-9).
