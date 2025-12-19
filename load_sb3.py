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
import pybullet as pb
import matplotlib.pyplot as plt
import time
import matplotlib
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


from stable_baselines3.common.vec_env import VecVideoRecorder

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
#flat with constant velocity (1,0,0), CPG, minimal observation
interm_dir = "./logs/intermediate_models/121725172313"
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
#new run 6 run med obs world frame more noise (0.04)
#interm_dir = "./logs/intermediate_models/121725200411"
#new run 7 run med obs world frame random velocity
#interm_dir = "./logs/intermediate_models/121725234805"
#new run 8 run med obs world frame fixed slopes based on run 4
#interm_dir = "./logs/intermediate_models/121825102644"
#new run 9 run med obs world frame random slopes (0.05, 0.3) based on run 4 PPO9
#interm_dir = "./logs/intermediate_models/121825163357"
#new run 10 run med obs world frame random velocity PPO10
interm_dir = "./logs/intermediate_models/121825181753"


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
env_config['randomise_commanded_velocity'] = True
env_config['commanded_velocity'] = np.array([0, 0, 0])  


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

####
#
#
#
#
#
#
#


obs = env.reset()
episode_reward = 0

g = 9.81
SIM_DT = env.envs[0].env._time_step  # simulation timestep (per sim substep)

# robot mass (robust)
try:
    mass = float(np.sum(env.envs[0].env.robot.GetTotalMassFromURDF()))
except Exception:
    mass = 12.0  # fallback
print(f"Robot mass [kg] = {mass:.3f}")

# work accumulator: ∫ |tau · qdot| dt   (Joules)
work_J = 0.0

# distance reference
start_pos = np.array(env.envs[0].env.robot.GetBasePosition(), dtype=float)



omega_swing  = 5 * 2 * np.pi
omega_stance = 2 * 2 * np.pi

T_swing  = np.pi / omega_swing
T_stance = np.pi / omega_stance
T_step   = T_swing + T_stance
duty     = T_stance / T_step

print("CPG timing parameters:")
print(f"  Swing time   : {T_swing:.3f} s")
print(f"  Stance time  : {T_stance:.3f} s")
print(f"  Step time    : {T_step:.3f} s")
print(f"  Duty cycle   : {duty:.2f}")


obs = env.reset()
episode_reward = 0

# =========================
# Data logging containers
# =========================
base_lin_vel = []   # list of [vx, vy, vz]
time_log = []

dt = env.envs[0].env._time_step  # simulation timestep
t = 0.0

# =========================
# Duty cycle logging (single rollout)
# =========================
contacts_log = []   # [FR, FL, RR, RL] per step
time_log_dc = []
DT = env.envs[0].env._time_step * env.envs[0].env._action_repeat


for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test if the outputs make sense)
    obs, rewards, dones, info = env.step(action)
    contacts = env.envs[0].env.get_foot_contacts()  # shape (4,)
    contacts_log.append(np.array(contacts, dtype=int))
    time_log_dc.append(i * DT)
    taus = env.envs[0].env._dt_motor_torques
    qdots = env.envs[0].env._dt_motor_velocities
    for tau, qdot in zip(taus, qdots):
        tau = np.asarray(tau, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        work_J += abs(float(np.dot(tau, qdot))) * SIM_DT

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

end_pos = np.array(env.envs[0].env.robot.GetBasePosition(), dtype=float)

contacts_log = np.array(contacts_log)  # (T, 4)
time_log_dc = np.array(time_log_dc)

foot_names = ["FR", "FL", "RR", "RL"]
foot = 0  # choose one foot (0 = FR)

c = contacts_log[:, foot]  # 1 = stance, 0 = swing

# detect touchdown events: 0 -> 1 transitions
touchdowns = np.where((c[1:] == 1) & (c[:-1] == 0))[0] + 1

if len(touchdowns) < 2:
    print("Not enough steps detected to compute duty cycle.")
else:
    td0, td1 = touchdowns[0], touchdowns[1]

    idx = np.arange(td0, td1)
    stance_time = np.sum(c[idx] == 1) * DT
    swing_time  = np.sum(c[idx] == 0) * DT
    step_time   = stance_time + swing_time
    duty_cycle  = stance_time / step_time

    print(f"Duty cycle ({foot_names[foot]} foot):")
    print(f"  Stance time = {stance_time:.3f} s")
    print(f"  Swing time  = {swing_time:.3f} s")
    print(f"  Step time   = {step_time:.3f} s")
    print(f"  Duty cycle  = {duty_cycle:.2f}")


# forward distance in x (simple + typical for velocity tracking)
distance = float(end_pos[0] - start_pos[0])

if distance <= 1e-6:
    print("Cost of Transport: not defined (distance too small).")
else:
    CoT = work_J / (mass * g * distance)
    print(f"Cost of Transport = {CoT:.3f}  (work={work_J:.1f} J, dist={distance:.2f} m)")


# ============================================================
# CONFIG
# ============================================================
N_RUNS = 10                 # number of evaluation rollouts
MAX_STEPS = 2000            # max steps per rollout (safety cap)
CONFIDENCE = 0.90           # 90% confidence band
#DT = env.envs[0].env._time_step
DT = env.envs[0].env._time_step * env.envs[0].env._action_repeat
cmd_v = np.array(env_config["commanded_velocity"], dtype=float)  # shape (3,)

# ============================================================
# Helper: Student-t critical value (with a safe fallback)
# ============================================================
def t_critical_value(confidence: float, df: int) -> float:
    """
    Returns t_{1-alpha/2, df} for a two-sided CI.
    Uses scipy if available; falls back to normal approx if not.
    """
    alpha = 1.0 - confidence
    try:
        from scipy.stats import t
        return float(t.ppf(1 - alpha / 2, df=df))
    except Exception:
        # Normal approx (good when df is large). Still usable if scipy missing.
        from math import erf, sqrt

        # Inverse CDF of normal: use numpy if available (it is), else fallback is messy.
        # We'll use a numeric approximation via scipy if present, otherwise this:
        # We'll just hardcode common z for 90% two-sided: 1.6448536
        # (because alpha/2 = 0.05 => z = 1.64485)
        if abs(confidence - 0.90) < 1e-9:
            return 1.6448536269514722
        if abs(confidence - 0.95) < 1e-9:
            return 1.959963984540054
        if abs(confidence - 0.99) < 1e-9:
            return 2.5758293035489004
        # Default fallback
        return 1.6448536269514722

# ============================================================
# 1) Collect errors across runs
# ============================================================
# We'll store each rollout as an array of shape (T_i, 3)
rollout_errors = []

# log only once (first rollout)
pos_log = []
t_pos = []
rpy_log = []   # roll, pitch, yaw for ONE run

record_once = True

for run_idx in range(N_RUNS):
    obs = env.reset()

    run_err = []
    deterministic = True

    # only for the first rollout
    log_this_run = (run_idx == 0)

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        # --- velocity error for CI (every run) ---
        v = env.envs[0].env.robot.GetBaseLinearVelocity()
        v = np.array(v, dtype=float)
        run_err.append(v - cmd_v)

        # --- position log ONLY once ---
        if log_this_run:
            p = np.array(env.envs[0].env.robot.GetBasePosition(), dtype=float)  # (x,y,z)
            pos_log.append(p)
            t_pos.append(step * DT)
            
            # orientation -> roll, pitch, yaw
            quat = env.envs[0].env.robot.GetBaseOrientation()  # quaternion (x,y,z,w)
            rpy = np.array(env.envs[0].env.robot.GetBaseOrientation(), dtype=float)  # (roll,pitch,yaw) in rad
            rpy_log.append(rpy)

        if bool(done):
            break

    rollout_errors.append(np.asarray(run_err))


# ============================================================
# 2) Align runs to a common length
#    (truncate all to the shortest rollout length)
# ============================================================
min_T = min(r.shape[0] for r in rollout_errors)
if min_T < 5:
    raise RuntimeError(f"Rollouts are too short (min_T={min_T}). Check termination / done flags.")

errors = np.stack([r[:min_T] for r in rollout_errors], axis=0)  # (N, T, 3)

# time axis in seconds
t = np.arange(min_T) * DT

# ============================================================
# 3) Compute mean error and 90% confidence band
# ============================================================
N = errors.shape[0]
T = errors.shape[1]

mean_err = errors.mean(axis=0)                 # (T, 3)
std_err = errors.std(axis=0, ddof=1)           # (T, 3) sample std

# standard error of the mean
sem = std_err / np.sqrt(N)                     # (T, 3)

# t critical value for two-sided CI
tval = t_critical_value(CONFIDENCE, df=N - 1)  # scalar

ci_half = tval * sem                           # (T, 3)

lower = mean_err - ci_half
upper = mean_err + ci_half

# ============================================================
# 4) Plot: 3 subplots (x/y/z), line=mean error, shade=90% CI band
# ============================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axis_names = ["x", "y", "z"]
for i, ax in enumerate(axs):
    ax.plot(t, mean_err[:, i], label=f"Mean error v_{axis_names[i]}")
    ax.fill_between(t, lower[:, i], upper[:, i], alpha=0.2, label=f"{int(CONFIDENCE*100)}% CI band")

    ax.set_ylabel("Velocity error [m/s]")
    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel("Normalized Episode Progress")
fig.suptitle("Base Linear Velocity Error: Mean ± 90% Confidence Band (across rollouts)")
plt.tight_layout()
plt.show()

pos_log = np.array(pos_log)   # (T0, 3)
t_pos = np.array(t_pos)       # (T0,)

plt.figure(figsize=(10, 5))
plt.plot(t_pos, pos_log[:, 0], label="x")
plt.plot(t_pos, pos_log[:, 1], label="y")
plt.plot(t_pos, pos_log[:, 2], label="z")
plt.xlabel("Time [s]")
plt.ylabel("Base position [m]")
plt.title("Base Position (single rollout)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


rpy_log = np.array(rpy_log)  # (T0, 3)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = ["roll", "pitch", "yaw"]

for i, ax in enumerate(axs):
    ax.plot(t_pos, rpy_log[:, i], label=labels[i])
    ax.set_ylabel("Angle [rad]")
    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel("Time [s]")
fig.suptitle("Base Orientation (single rollout)")
plt.tight_layout()
plt.show()


video_env = VecVideoRecorder(
    env,
    video_folder="./videos",
    record_video_trigger=lambda step: step == 0,
    video_length=MAX_STEPS,
    name_prefix="eval_run0",
)

# run ONE rollout on video_env (run_idx==0), then close it





# base_lin_vel = np.array(base_lin_vel)  # shape: (T, 3)
# time_log = np.array(time_log)
    
# =========================
# Plot base linear velocity
# =========================
# cmd_vx = env_config['commanded_velocity']

# plt.figure(figsize=(10, 5))
# plt.plot(time_log, base_lin_vel[:, 0]- cmd_vx[0], label="v_x")
# plt.plot(time_log, base_lin_vel[:, 1]- cmd_vx[1], label="v_y")
# plt.plot(time_log, base_lin_vel[:, 2]- cmd_vx[2], label="v_z")

# plt.xlabel("Time [s]")
# plt.ylabel("Base Linear Velocity Error [m/s]")
# plt.title("Quadruped Base Linear Velocity Error During Policy Execution")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



omega_swing_log  = np.array(omega_swing_log)   # (T,) or (T,4)
omega_stance_log = np.array(omega_stance_log)  # (T,) or (T,4)
t = np.array(t_pos)  # time axis (T,)

# avoid divide-by-zero
eps = 1e-6
omega_swing_log  = np.maximum(omega_swing_log, eps)
omega_stance_log = np.maximum(omega_stance_log, eps)

T_swing = np.pi / omega_swing_log
T_stance = np.pi / omega_stance_log
T_step = T_swing + T_stance

# ---- Plot swing duration (and optionally step duration) ----
plt.figure(figsize=(10, 5))

if T_swing.ndim == 1:
    plt.plot(t, T_swing, label="Swing duration")
    # optional:
    plt.plot(t, T_step, label="Step duration (swing+stance)")
else:
    leg_names = ["FR", "FL", "RR", "RL"]
    for i in range(T_swing.shape[1]):
        plt.plot(t, T_swing[:, i], label=f"T_swing {leg_names[i]}")
    # optional: step duration
    # for i in range(T_step.shape[1]):
    #     plt.plot(t, T_step[:, i], label=f"T_step {leg_names[i]}")

plt.xlabel("Time [s]")
plt.ylabel("Duration [s]")
plt.title("Swing Duration (computed from ω_swing)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
