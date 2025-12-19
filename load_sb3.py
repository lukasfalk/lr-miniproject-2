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

import os, sys
import gymnasium as gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results

LEARNING_ALG = "PPO" #"SAC"
interm_dir = "./logs/intermediate_models/121925104416"
# path to saved models, i.e. interm_dir + '102824115106'
log_dir = interm_dir + ''

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 

env_config["motor_control_mode"] = "CARTESIAN_PD"
env_config["task_env"] = "LR_COURSE_TASK"
env_config["observation_space_mode"] = "LR_COURSE_OBS"
env_config['randomise_commanded_velocity'] = False
env_config['commanded_velocity'] = np.array([1.0, 0, 0])  


# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0
print(obs)

# =========================
# Data logging containers
# =========================
base_lin_vel = []   # list of [vx, vy, vz]
time_log = []
# =========================
# Data logging containers
# =========================
base_lin_vel_log = []   # list of [vx, vy, vz]
base_pos_log = []       # list of [x, y, z]
base_rpy_log = []       # list of [roll, pitch, yaw]
time_log = []

dt = env.envs[0].env._time_step  # simulation timestep
t = 0.0
frame = env.render()


for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test if the outputs make sense)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    # =========================
    # Log robot states
    # =========================
    # Access the robot instance from the environment
    robot = env.envs[0].env.robot
    
    # 1. Base Linear Velocity
    v = robot.GetBaseLinearVelocity()
    base_lin_vel_log.append(v)
    
    # 2. Base Position
    pos = robot.GetBasePosition()
    base_pos_log.append(pos)
    
    # 3. Base Orientation (Roll, Pitch, Yaw)
    rpy = robot.GetBaseOrientationRollPitchYaw()
    base_rpy_log.append(rpy)
    # =========================
    # Log base linear velocity
    # =========================
    v = env.envs[0].env.robot.GetBaseLinearVelocity()
    base_lin_vel.append(v)
    time_log.append(t)
    t += dt
    
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 

base_lin_vel = np.array(base_lin_vel)  # shape: (T, 3)
time_log = np.array(time_log) * 10
    
# =========================
# Plot base linear velocity
# =========================
cmd_vx = env_config['commanded_velocity']

plt.figure(figsize=(10, 5))
plt.plot(time_log, base_lin_vel[:, 0]- cmd_vx[0], label="v_x")
plt.plot(time_log, base_lin_vel[:, 1]- cmd_vx[1], label="v_y")

plt.xlabel("Time [s]")
plt.ylabel("Base Linear Velocity Absolute Error [m/s]")
plt.title("Quadruped Base Linear Velocity Error During Policy Execution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert lists to numpy arrays so we can slice them (e.g., [:, 0])
base_lin_vel_log = np.array(base_lin_vel_log)
base_pos_log = np.array(base_pos_log)
base_rpy_log = np.array(base_rpy_log)
time_log = np.array(time_log)

plt.figure(figsize=(10, 5))
plt.plot(time_log, base_pos_log[:, 0], label="x")
plt.plot(time_log, base_pos_log[:, 1], label="y")
plt.plot(time_log, base_pos_log[:, 2], label="z")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Base Position vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time_log, np.degrees(base_rpy_log[:, 0]), label="Roll")
plt.plot(time_log, np.degrees(base_rpy_log[:, 1]), label="Pitch")
plt.plot(time_log, np.degrees(base_rpy_log[:, 2]), label="Yaw")
plt.xlabel("Time [s]")
plt.ylabel("Angle [degrees]")
plt.title("Base Orientation vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()