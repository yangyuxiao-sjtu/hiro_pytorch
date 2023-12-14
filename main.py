import argparse
import copy
import os

import numpy as np
import torch
from tqdm import tqdm
from hiro.config import initialize_cuda_device

initialize_cuda_device(1)


from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.models_back import HiroAgent, TD3Agent
from hiro.utils import Logger, _is_update


try:
    import wandb

    _wandb_installed = True
except ImportError:
    _wandb_installed = False


def run_evaluation(args, env, agent):
    agent.load(args.load_episode)

    rewards, success_rate = agent.evaluate_policy(
        env, args.eval_episodes, args.render, args.save_video, args.sleep
    )

    print(
        "mean:{mean:.2f}, \
            std:{std:.2f}, \
            median:{median:.2f}, \
            success:{success:.2f}".format(
            mean=np.mean(rewards),
            std=np.std(rewards),
            median=np.median(rewards),
            success=success_rate,
        )
    )


class Trainer:
    def __init__(self, args, env, agent, experiment_name):
        self.args = args
        self.env = env
        self.agent = agent
        log_path = os.path.join(args.log_path, experiment_name)
        self.logger = Logger(log_path=log_path, use_wandb=args.use_wandb)
        if args.use_wandb:
            assert _wandb_installed, "Wandb not installed"
            wandb.init(
                project="HIRO",
                dir=log_path,
                config=args,
                name=experiment_name,
            )
            # `train/` metrics use global step as x-axis
            # `eval/` metrics use episode as x-axis
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/step")
            wandb.define_metric("eval/*", step_metric="eval/step")

    def train(self):
        global_step = 0

        for e in tqdm(np.arange(self.args.num_episode) + 1):
            obs = self.env.reset()
            fg = obs["desired_goal"]
            s = obs["observation"]
            done = False

            step = 0
            episode_reward = 0

            self.agent.set_final_goal(fg)

            while not done:
                # Take action
                a, r, n_s, done = self.agent.step(
                    s, self.env, step, global_step, explore=True
                )

                # Append
                self.agent.append(step, s, a, n_s, r, done)

                # Train
                losses, td_errors = self.agent.train(global_step)

                # Log
                self.log(global_step, [losses, td_errors])

                # Updates
                s = n_s
                episode_reward += r
                step += 1
                global_step += 1
                self.agent.end_step()

            self.agent.end_episode(e, self.logger)
            self.logger.write("eval/Exploration Reward", episode_reward, e)
            self.evaluate(e)

    def log(self, global_step, data):
        losses, td_errors = data[0], data[1]

        # Logs
        if global_step >= self.args.start_training_steps and _is_update(
            global_step, args.writer_freq
        ):
            for k, v in losses.items():
                self.logger.write("train/%s" % (k), v, global_step)

            for k, v in td_errors.items():
                self.logger.write("train/%s" % (k), v, global_step)

    def evaluate(self, e):
        # Print
        if _is_update(e, args.print_freq):
            #agent = copy.deepcopy(self.agent)
            rewards, success_rate = agent.evaluate_policy(self.env)
            # rewards, success_rate = self.agent.evaluate_policy(self.env)
            self.logger.write("eval/Success Rate", success_rate, e)
            self.logger.write_csv({"eval/Success Rate": success_rate})

            print(
                "episode:{episode:05d}, mean:{mean:.2f}, std:{std:.2f}, median:{median:.2f}, success:{success:.2f}".format(
                    episode=e,
                    mean=np.mean(rewards),
                    std=np.std(rewards),
                    median=np.median(rewards),
                    success=success_rate,
                )
            )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # make all file use same device
    

    # Across All
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--sleep", type=float, default=-1)
    parser.add_argument("--eval_episodes", type=float, default=5, help="Unit = Episode")
    parser.add_argument("--env", default="AntMaze", type=str)
    parser.add_argument("--td3", action="store_true")

    # Training
    parser.add_argument("--num_episode", default=25000, type=int)
    parser.add_argument(
        "--start_training_steps", default=2500, type=int, help="Unit = Global Step"
    )
    parser.add_argument(
        "--writer_freq", default=25, type=int, help="Unit = Global Step"
    )
    # Training (Model Saving)
    parser.add_argument("--subgoal_dim", default=15, type=int)
    parser.add_argument("--load_episode", default=-1, type=int)
    parser.add_argument(
        "--model_save_freq", default=2000, type=int, help="Unit = Episodes"
    )
    parser.add_argument("--print_freq", default=250, type=int, help="Unit = Episode")
    # Model
    parser.add_argument("--log_path", default="logs", type=str)
    parser.add_argument("--policy_freq_low", default=2, type=int)
    parser.add_argument("--policy_freq_high", default=2, type=int)
    # Replay Buffer
    parser.add_argument("--buffer_size", default=200000, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--buffer_freq", default=10, type=int)
    parser.add_argument("--train_freq", default=10, type=int)
    parser.add_argument("--reward_scaling", default=0.1, type=float)

    # Added
    parser.add_argument("--use_correction", action="store_true")
    parser.add_argument("--use_reg_mse", action="store_true")
    parser.add_argument("--use_backward_loss", action="store_true")
    parser.add_argument(
        "--reg_mse_weight", default=0.0, type=float
    )  # weight for mse in high_con
    parser.add_argument(
        "--backward_weight", default=0.01, type=float
    )  # weight for low_con backward to high_con
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()
    setup_seed(args.seed)

    if args.env == "PointMaze":
        args.subgoal_dim = 3
        args.num_episode = 20000
        args.model_save_freq = 1000
        args.print_freq = 100

    experiment_name = f"{args.env}-" + ("td3" if args.td3 else "hiro")
    if args.use_correction:
        experiment_name += "-corr"
    if args.use_reg_mse:
        experiment_name += f"-reg_{args.reg_mse_weight}"
    if args.use_backward_loss:
        experiment_name += f"-bkw_{args.backward_weight}"
    experiment_name += f"/{args.seed}"
    print("Experiment name: " + experiment_name)
    model_path = os.path.join(args.log_path, experiment_name, "models")

    # Environment and its attributes
    env = EnvWithGoal(create_maze_env(args.env), args.env)
    goal_dim = 2
    state_dim = env.state_dim
    action_dim = env.action_dim
    scale = env.action_space.high * np.ones(action_dim)

    # Spawn an agent
    if args.td3:
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            scale=scale,
            model_save_freq=args.model_save_freq,
            model_path=model_path,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            start_training_steps=args.start_training_steps,
        )
    else:
        agent = HiroAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            subgoal_dim=args.subgoal_dim,
            scale_low=scale,
            start_training_steps=args.start_training_steps,
            model_path=model_path,
            model_save_freq=args.model_save_freq,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            buffer_freq=args.buffer_freq,
            train_freq=args.train_freq,
            reward_scaling=args.reward_scaling,
            policy_freq_high=args.policy_freq_high,
            policy_freq_low=args.policy_freq_low,
            use_reg_mse=args.use_reg_mse,
            use_backward_loss=args.use_backward_loss,
            reg_mse_weight=args.reg_mse_weight,
            backward_weight=args.backward_weight,
            use_correction=args.use_correction,
        )

    # Run training or evaluation
    if args.train:
        if args.continue_training:
            agent.load(-1)
        # Start training
        trainer = Trainer(args, env, agent, experiment_name)
        trainer.train()
    if args.eval:
        run_evaluation(args, env, agent)
