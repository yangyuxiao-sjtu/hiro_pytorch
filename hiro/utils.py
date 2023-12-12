import csv
import os

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def var(tensor):
    if torch.cuda.is_available():
        return tensor.to(device)
    else:
        return tensor


def get_tensor(z):
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


class Logger:
    def __init__(
        self, log_path, use_wandb: bool = False, csv_log_file: str = "progress.csv"
    ):
        self.tb_writer = SummaryWriter(os.path.join(log_path, "tblogs"))
        self.csv_fd = None
        self.csv_log_path = os.path.join(log_path, csv_log_file)
        self.use_wandb = use_wandb

    def print(self, name, value, episode=-1, step=-1):
        string = "{} is {}".format(name, value)
        if episode > 0:
            print("Episode:{}, {}".format(episode, string))
        if step > 0:
            print("Step:{}, {}".format(step, string))

    def write(self, name, value, index):
        self.tb_writer.add_scalar(name, value, index)
        if self.use_wandb:
            if name.startswith("train/"):
                metrics = {name: value, "train/step": int(index)}
            elif name.startswith("eval/"):
                metrics = {name: value, "eval/step": int(index)}
            else:
                raise ValueError(
                    f"Metric name must start with train/ or eval/, but got {name}"
                )
            wandb.log(metrics)

    def write_csv(self, data_dict):
        if self.csv_fd is None:
            first_time = True
            self.csv_fd = open(self.csv_log_path, "w")
        else:
            first_time = False
        writer = csv.DictWriter(self.csv_fd, fieldnames=list(data_dict.keys()))
        if first_time:
            writer.writeheader()
        writer.writerow(data_dict)
        self.csv_fd.flush()


def _is_update(episode, freq, ignore=0, rem=0):
    if episode != ignore and episode % freq == rem:
        return True
    return False


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((buffer_size, state_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.n_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.not_done = np.zeros((buffer_size, 1))

        self.device = device

    def append(self, state, action, n_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )


def record_experience_to_csv(args, experiment_name, csv_name="experiments.csv"):
    # append DATE_TIME to dict
    d = vars(args)
    d["date"] = experiment_name

    if os.path.exists(csv_name):
        # Save Dictionary to a csv
        with open(csv_name, "a") as f:
            w = csv.DictWriter(f, list(d.keys()))
            w.writerow(d)
    else:
        # Save Dictionary to a csv
        with open(csv_name, "w") as f:
            w = csv.DictWriter(f, list(d.keys()))
            w.writeheader()
            w.writerow(d)


def listdirs(directory):
    return [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]
