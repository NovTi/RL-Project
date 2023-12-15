import os
import pdb
import torch
import datetime
import argparse
import matplotlib.pyplot as plt

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def write_list_to_file(file_path, float_list):
    with open(file_path, 'w') as file:
        for value in float_list:
            file.write(f"{value}\n")


def handle_exp_id(exp_id, env, seed=None):
    # exp_id: ppo_lr05_ent15_cp2
    base_lr = 1e-5
    base_clip_range = 0.1
    base_ent_coef = 0.001

    config_lst = exp_id.split('_')[1:]  # ['lr05', 'tau05]

    # non-tunable hyperparameters
    buffer_size=int(1e5)   # Replay buffer size
    n_steps = 1024    # The number of steps to run for each environment per update
    batch_size = 64
    n_epochs = 4       # Number of epoch when optimizing the surrogate loss
    gae_lambda = 0.95  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    sde_sample_freq=4. # Sample a new noise matrix every n steps when using gSDE
    gamma = 0.99       # Reward discount
    # tunable hyperparameters
    learning_rate=1e-3 # lr: [5e-5, 1e-4]   base_lr = 1e-5    , 5e-4
    ent_coef = 0.005   # Entropy coefficient for the loss calculation
    clip_range=0.2    # Clipping parameter [0.2, 0.4, 0.6]

    # learning rate
    learning_rate = float(config_lst[0][2:]) * base_lr
    # entropy coefficient
    ent_coef = float(config_lst[1][3:]) * base_ent_coef
    # clip_range
    clip_range = int(config_lst[2][2:]) * base_clip_range

    model_seed = 0 if seed is None else seed

    model =  PPO(
        policy = 'MlpPolicy',
        env = env,
        learning_rate=learning_rate,
        n_steps = n_steps,
        batch_size = batch_size,
        n_epochs = n_epochs,
        gamma = gamma,
        use_sde= True,
        gae_lambda = gae_lambda,
        ent_coef = ent_coef,
        sde_sample_freq=sde_sample_freq,
        verbose=1,
        clip_range=clip_range,
        seed=model_seed,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', required=True, type=str, help='tune hyperprameters')
    parser.add_argument('--seed', type=int, default=None, help='seed for runing average experiments')
    args = parser.parse_args()

    # print exp_id
    print("Exp ID: ", args.exp_id)

    # Create the environment
    env = gym.make("BipedalWalker-v3")
    env.reset()

    # Wrap the environment
    log_dir = f"./results/seed_average/{datetime.date.today()}/{args.exp_id}"
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir)

    # Create the model
    ppo_model = handle_exp_id(args.exp_id, env, seed=args.seed)

    # Train the agent
    ppo_model.learn(total_timesteps=1e7, log_interval=100, progress_bar=True)  # around 4 hours

    # Save the model
    save_model_path = os.path.join(log_dir, 'ppo_model')
    ppo_model.save(save_model_path)
    print(ppo_model.policy)

    file_path = os.path.join(log_dir, 'episode_rewards.txt')
    file_path_length = os.path.join(log_dir, 'episode_lengths.txt')

    # save the files
    write_list_to_file(file_path=file_path, float_list=env.get_episode_rewards())
    write_list_to_file(file_path=file_path_length, float_list=env.get_episode_lengths())

    # plot the image into the folder
    plt.figure(figsize=(10, 6))
    episode_rew = env.get_episode_rewards()
    episode_rew_ma = (np.convolve(episode_rew, np.ones(100), "valid") / 100)
    plt.plot(np.arange(100, len(episode_rew)+1), episode_rew_ma)
    plt.ylabel('Average Reward for the past 100 episodes')
    plt.xlabel('Episode #')
    plt.savefig(os.path.join(log_dir, 'episode_rewards.png'))

    plt.figure(figsize=(10, 6))
    episode_len = env.get_episode_lengths()
    episode_len_ma = (np.convolve(episode_len, np.ones(100), "valid") / 100)
    plt.plot(np.arange(100, len(episode_len)+1), episode_len_ma)
    plt.ylabel('Average Episode Length for the past 100 episodes')
    plt.xlabel('Episode #')
    plt.savefig(os.path.join(log_dir, 'episode_lengths.png'))



