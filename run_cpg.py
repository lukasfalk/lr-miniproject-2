# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Run CPG """

import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# initialize data structures to save CPG and robot states
cpg_states = np.zeros((4, 4, TEST_STEPS)) # plot for (r, theta, r_dot, theta_dot) for each leg [info, legId, timeStep]

############## Sample Gains
# joint PD gains
kp = np.array([100,100,100])
kd = np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 

  # get desired foot positions from CPG 
  xs, zs = cpg.update()

  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()
  q = np.reshape(q, (4,3)) # reshape to leg x joint
  dq = np.reshape(dq, (4,3)) # reshape to leg x joint

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz_des = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q_des = env.robot.ComputeInverseKinematics(i, leg_xyz_des)

    # Add joint PD contribution to tau for leg i (Equation 4)
    dq_des = np.zeros(3) # We want the foot to stand still
    tau += kp @ (leg_q_des - q[i]) + kd @ (dq_des - dq[i])

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      _, pos_des = env.robot.ComputeJacobianAndPosition(i, leg_q_des)

      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      J, pos = env.robot.ComputeJacobianAndPosition(i, q[i])

      # Get current foot velocity in leg frame (Equation 2)
      vel = J @ dq[i]

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      vel_des = np.zeros(3)
      tau += J.T @ (kp * (pos_des - pos) + kd * (vel_des - vel))
      

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO][ ] save any CPG or robot states
  cpg_states[:,0,j] = cpg.get_r()
  cpg_states[:,1,j] = cpg.get_theta()
  cpg_states[:,2,j] = cpg.get_dr()
  cpg_states[:,3,j] = cpg.get_dtheta()

##################################################### 
# PLOTS
#####################################################
# [TODO][ ] Create your plots

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
fig.suptitle("CPG States per Leg")

for leg in range(4):
    ax = axs[leg]

    # Left axis: amplitude r
    ax.plot(t, cpg_states[leg, 0, :], label="r (amplitude)")
    ax.set_ylabel(f"Leg {leg}  |  r", color="tab:blue")
    ax.tick_params(axis='y', labelcolor="tab:blue")

    # Right axis: theta
    ax2 = ax.twinx()
    ax2.plot(t, cpg_states[leg, 1, :], color="tab:red", label="\u03b8 (angle)")
    ax2.set_ylabel("\u03b8", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    # Optional: also plot r_dot and theta_dot
    # ax.plot(t, cpg_states[leg, 2, :], '--', color="tab:cyan", label="r_dot")
    # ax2.plot(t, cpg_states[leg, 3, :], '--', color="tab:orange", label="theta_dot")

    ax.grid(True)

axs[-1].set_xlabel("Time [s]")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()