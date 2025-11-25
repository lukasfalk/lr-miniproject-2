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
                    record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(2.5 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO][x] initialize data structures to save CPG and robot states
cpg_states = np.zeros((4, 4, TEST_STEPS)) # plot for (r, theta, r_dot, theta_dot) for each leg
feet_positions = np.zeros((4, 3, TEST_STEPS)) # foot positions for each leg (x, y, z)
feet_desired_positions = np.zeros((4, 3, TEST_STEPS)) # desired foot positions for each leg (x, y, z)
joint_angles = np.zeros((4, 3, TEST_STEPS)) # joint angles for each leg (hip, thigh, knee)
joint_angles_des = np.zeros((4, 3, TEST_STEPS)) # desired joint angles for each leg (hip, thigh, knee)

############## Sample Gains
# joint PD gains
kp = np.array([100,100,100])
kd = np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([1200]*3)
kdCartesian = np.diag([40]*3)

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
      tau += J.T @ (kpCartesian @ (pos_des - pos) + kdCartesian @ (vel_des - vel))
      
    feet_positions[i,:,j] = pos
    feet_desired_positions[i,:,j] = pos_des

    joint_angles[i,:,j] = q[i]
    joint_angles_des[i,:,j] = leg_q_des


    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO][x] save any CPG or robot states
  cpg_states[:,0,j] = cpg.get_r()
  cpg_states[:,1,j] = cpg.get_theta()
  cpg_states[:,2,j] = cpg.get_dr()
  cpg_states[:,3,j] = cpg.get_dtheta()

# [TODO][x] Create your plots
leg = 0  # choose which leg to plot

### CPG STATE PLOTS
r         = cpg_states[leg, 0, :]
theta     = cpg_states[leg, 1, :]
r_dot     = cpg_states[leg, 2, :]
theta_dot = cpg_states[leg, 3, :]

# Optional: wrap theta into [-pi, pi] for readability
theta = (theta + np.pi) % (2*np.pi) - np.pi

fig, axs = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"CPG States for Leg {leg}", fontsize=16)

# r
axs[0].plot(t, r, color="tab:blue", linewidth=1.5)
axs[0].set_ylabel(r"$r$")
axs[0].grid(True, alpha=0.3)

# theta
axs[1].plot(t, theta, color="tab:red", linewidth=1.5)
axs[1].set_ylabel(r"$\theta$ [rad]")
axs[1].grid(True, alpha=0.3)
axs[1].set_ylim([-np.pi, np.pi])

# r_dot
axs[2].plot(t, r_dot, color="tab:green", linewidth=1.5)
axs[2].set_ylabel(r"$\dot{r}$")
axs[2].grid(True, alpha=0.3)

# theta_dot
axs[3].plot(t, theta_dot, color="tab:purple", linewidth=1.5)
axs[3].set_ylabel(r'$\dot{\theta}$ [rad/s]')
axs[3].set_xlabel("Time [s]")
axs[3].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

### FOOT POSITION TRACKING PLOT
pos     = feet_positions[leg, :, :]          # shape (3, TEST_STEPS)
pos_des = feet_desired_positions[leg, :, :]  # shape (3, TEST_STEPS)

labels = ["x", "y", "z"]
colors = ["tab:blue", "tab:green", "tab:red"]

fig, axs = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
fig.suptitle(f"Foot Position Tracking for Leg {leg}", fontsize=16)

for k in range(3):
    axs[k].plot(t, pos[k],     linewidth=1.5, color=colors[k], label=f"{labels[k]} actual")
    axs[k].plot(t, pos_des[k], linewidth=1.2, color="black", linestyle="--", label=f"{labels[k]} desired")

    axs[k].set_ylabel(f"{labels[k]} [m]")
    axs[k].grid(True, alpha=0.3)
    axs[k].legend(loc="upper right")

axs[-1].set_xlabel("Time [s]")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

### JOINT ANGLE TRACKING PLOT
q     = joint_angles[leg, :, :]       # shape (3, TEST_STEPS)
q_des = joint_angles_des[leg, :, :]   # shape (3, TEST_STEPS)

joint_labels = ["Hip", "Thigh", "Calf"]
colors = ["tab:blue", "tab:orange", "tab:green"]

fig, axs = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
fig.suptitle(f"Joint Angle Tracking for Leg {leg}", fontsize=16)

for k in range(3):
    axs[k].plot(t, q[k],     linewidth=1.5, color=colors[k], label=f"{joint_labels[k]} actual")
    axs[k].plot(t, q_des[k], linewidth=1.2, color="black", linestyle="--", label=f"{joint_labels[k]} desired")

    axs[k].set_ylabel(f"{joint_labels[k]} [rad]")
    axs[k].grid(True, alpha=0.3)
    axs[k].legend(loc="upper right")

axs[-1].set_xlabel("Time [s]")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()