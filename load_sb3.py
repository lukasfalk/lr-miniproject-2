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
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

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
#interm_dir = "./logs/intermediate_models/121325222414"
#best model so far previously: 121225020524
#interm_dir = "./logs/intermediate_models/121325222414"
#weighted fully (1, 0.5, 0.3)
#interm_dir = "./logs/intermediate_models/121325222414"
#unweighted
#interm_dir = "./logs/intermediate_models/121425002429"
#new weighted (0.4, 1.3, 1.3)
#interm_dir = "./logs/intermediate_models/121425014514"
#slope run with base 121425014514
#interm_dir = "./logs/intermediate_models/121425134857"
#random slope run with base 121425134857 (1/3)
#interm_dir = "./logs/intermediate_models/121425155310"
#random slope run with base 121425155310 (2/3)
#interm_dir = "./logs/intermediate_models/121425190350"
#random slope run with base 121425190350 (3/3) 
#interm_dir = "./logs/intermediate_models/121425210800"
#random slope run with base 121425210800  fixed slope 0.3
#interm_dir = "./logs/intermediate_models/121525010206"
#random slope run with base 121425210800 (4/4) random slope again
#interm_dir = "./logs/intermediate_models/121525154102"
#
#
#
#
#
#
#new run 1 run full obs
#interm_dir = "./logs/intermediate_models/121625220200"
#new run 2 run med obs
#interm_dir = "./logs/intermediate_models/121625233724"
#new run 3 run med obs SLOPES
#interm_dir = "./logs/intermediate_models/121725012451" 
#new run 4 run med obs world frame
#interm_dir = "./logs/intermediate_models/121725125512"
#new run 5 run full obs world frame
#interm_dir = "./logs/intermediate_models/121725143400"
#new run 5 runn med obs world frame more noise (0.04)
interm_dir = "./logs/intermediate_models/121725200411"
# path to saved models, i.e. interm_dir + '102824115106'
#
log_dir = interm_dir + ''

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 

env_config["motor_control_mode"] = "CPG"
env_config["task_env"] = "LR_COURSE_TASK"
env_config["observation_space_mode"] = "LR_COURSE_OBS"
#env_config["terrain"] = "SLOPES" 
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

# =========================
# Data logging containers
# =========================
base_lin_vel = []   # list of [vx, vy, vz]
time_log = []

dt = env.envs[0].env._time_step  # simulation timestep
t = 0.0

for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test if the outputs make sense)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards

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
time_log = np.array(time_log)
    
# =========================
# Plot base linear velocity
# =========================
cmd_vx = env_config['commanded_velocity']

plt.figure(figsize=(10, 5))
plt.plot(time_log, base_lin_vel[:, 0]- cmd_vx[0], label="v_x")
plt.plot(time_log, base_lin_vel[:, 1]- cmd_vx[1], label="v_y")
plt.plot(time_log, base_lin_vel[:, 2]- cmd_vx[2], label="v_z")

plt.xlabel("Time [s]")
plt.ylabel("Base Linear Velocity Error [m/s]")
plt.title("Quadruped Base Linear Velocity Error During Policy Execution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()