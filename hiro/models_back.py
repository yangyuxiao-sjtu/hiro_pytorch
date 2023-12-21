##################################################
# @copyright Kandai Watanabe
# @email kandai.wata@gmail.com
# @institute University of Colorado Boulder
#
# NN Models for HIRO
# (Data-Efficient Hierarchical Reinforcement Learning)
# Parameters can be find in the original paper
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hiro.hiro_utils import HighReplayBuffer, LowReplayBuffer, ReplayBuffer, Subgoal
from hiro.utils import _is_update, get_tensor,cuda_device_index

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(f"cuda:{cuda_device_index}" if torch.cuda.is_available() else "cpu")

class TD3Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, scale=None):
        super().__init__()
        if scale is None:
            scale = torch.ones(state_dim)
        else:
            scale = get_tensor(scale)
        self.scale = nn.Parameter(scale.clone().detach().float(), requires_grad=False)

        self.l1 = nn.Linear(state_dim + goal_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, state, goal):
        a = F.relu(self.l1(torch.cat([state, goal], 1)))
        a = F.relu(self.l2(a))
        return self.scale * torch.tanh(self.l3(a))


class TD3Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(TD3Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 300)
        self.l5 = nn.Linear(300, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q


class TD3Controller(object):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005,
    ):
        self.name = "td3"
        self.scale = scale
        self.model_path = model_path

        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau

        self.actor = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(device)
        self.actor_target = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(
            device
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic1_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)

        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=critic_lr
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=critic_lr
        )
        self._initialize_target_networks()

        self._initialized = False
        self.total_it = 0

    def _initialize_target_networks(self):
        self._update_target_network(self.critic1_target, self.critic1, 1.0)
        self._update_target_network(self.critic2_target, self.critic2, 1.0)
        self._update_target_network(self.actor_target, self.actor, 1.0)
        self._initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(
                tau * origin_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, episode):
        # create episode directory. (e.g. model/2000)
        model_path = os.path.join(self.model_path, str(episode))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # save file (e.g. model/2000/high_actor.h)
        torch.save(
            self.actor.state_dict(), os.path.join(model_path, self.name + "_actor.h5")
        )
        torch.save(
            self.critic1.state_dict(),
            os.path.join(model_path, self.name + "_critic1.h5"),
        )
        torch.save(
            self.critic2.state_dict(),
            os.path.join(model_path, self.name + "_critic2.h5"),
        )

    def load(self, episode):
        # episode is -1, then read most updated
        if episode < 0:
            episode_list = list(map(int, os.listdir(self.model_path)))
            if len(episode_list) == 0:
                return
            episode = max(episode_list)

        model_path = os.path.join(self.model_path, str(episode))
        print(f"\033[92m Loading model checkpoint {model_path} \033[0m")
        self.actor.load_state_dict(
            torch.load(os.path.join(model_path, self.name + "_actor.h5"))
        )
        self.critic1.load_state_dict(
            torch.load(os.path.join(model_path, self.name + "_critic1.h5"))
        )
        self.critic2.load_state_dict(
            torch.load(os.path.join(model_path, self.name + "_critic2.h5"))
        )

    def _train(
        self,
        states,
        goals,
        actions,
        rewards,
        n_states,
        n_goals,
        not_done,
        **kwargs,
    ):
        self.total_it += 1
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            n_actions = self.actor_target(n_states, n_goals) + noise
            n_actions = torch.min(n_actions, self.actor.scale)
            n_actions = torch.max(n_actions, -self.actor.scale)

            target_Q1 = self.critic1_target(n_states, n_goals, n_actions)
            target_Q2 = self.critic2_target(n_states, n_goals, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q_detached = (rewards + not_done * self.gamma * target_Q).detach()

        current_Q1 = self.critic1(states, goals, actions)
        current_Q2 = self.critic2(states, goals, actions)

        critic1_loss = F.smooth_l1_loss(current_Q1, target_Q_detached)
        critic2_loss = F.smooth_l1_loss(current_Q2, target_Q_detached)
        critic_loss = critic1_loss + critic2_loss

        td_error = (target_Q_detached - current_Q1).mean().cpu().data.numpy()

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            a = self.actor(states, goals)
            Q1 = self.critic1(states, goals, a)

            actor_loss = -Q1.mean()  # multiply by neg becuz gradient ascent
            if self.name == "high" and self.use_reg_mse:
                Q1_detached = Q1.detach()

                if kwargs["mode"] == "clamp":
                    q_clip_eps = kwargs["eps"]
                    Q1_normed = torch.clamp(
                        Q1_detached,
                        min=Q1_detached.mean() * (1 - q_clip_eps),
                        max=Q1_detached.mean() * (1 + q_clip_eps),
                    )

                elif kwargs["mode"] == "min_max_norm":
                    Q1_normed = (Q1_detached - Q1_detached.min()) / (
                        Q1_detached.max() - Q1_detached.min()
                    )
                    selected_states = states[:, : a.shape[1]]
                    selected_n_states = n_states[:, : a.shape[1]]
                if self.use_prob_trick:
                    low_con=kwargs['policy']
                    batch_size=kwargs['batch_size']
                    state_arr=kwargs['state_arr']
                    action_arr=kwargs['action_arr']
                    action_dim=action_arr[0][0].shape
                    observe_dim = state_arr[0][0].shape
                    subgoal_dim =a.shape[1]
                    sg = a.detach()
                    # print('subgoal dim',sg.shape)
                    # print('ac dim',state_arr.shape)
                    sg =(sg + state_arr[:, 0, :subgoal_dim])[:, None]-state_arr[:, :, :subgoal_dim]
                    sg = sg.reshape((-1,subgoal_dim)) 
                    #print(sg.shape)
                    action_arr=action_arr.reshape((-1,)+action_dim)
                    state_arr = state_arr.reshape((-1,)+observe_dim)
                    action_true = low_con.policy(state_arr,sg)
                    action_arr= action_arr.cpu().numpy()
 
                    difference=action_true-action_arr
                    difference = np.where(difference != -np.inf, difference, 0)
                    difference=np.linalg.norm(difference,axis =-1)
                    difference=difference.reshape((batch_size,-1))
                    difference=(difference-difference.min())/(difference.max()-difference.min())# another norm to prevent exp(diff)->0
                    log_p=-0.5*np.sum(difference,axis =-1)
                    prob = np.exp(log_p)
                    prob=get_tensor(prob)
                   
                    actor_loss_mse = (
                        prob
                        * Q1_normed
                        * torch.linalg.norm(selected_states + a - selected_n_states, dim=1)
                    )
                else :
                   
                    actor_loss_mse = (
                        Q1_normed
                        * torch.linalg.norm(selected_states + a - selected_n_states, dim=1)
                    )

                actor_loss = actor_loss + self.reg_mse_weight * actor_loss_mse.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.name == "low" and self.use_backward_loss:
                high_con, fg = kwargs["high_con"], kwargs["fg"]

                new_sg = high_con.policy(states, fg, to_numpy=False)
                # new_sg = new_sg.detach()
                a = self.actor(states, new_sg)
                new_Q1 = self.critic1(states, new_sg.detach(), a)  # changed
                actor_loss_new = -new_Q1.mean() * self.backward_weight
                high_con.actor_optimizer.zero_grad()
                actor_loss_new.backward()
                high_con.actor_optimizer.step()
                # for name, weight in high_con.actor.named_parameters():
                #     if weight.requires_grad:
                #         print("weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())

            self._update_target_network(self.critic1_target, self.critic1, self.tau)
            self._update_target_network(self.critic2_target, self.critic2, self.tau)
            self._update_target_network(self.actor_target, self.actor, self.tau)

            return {
                "actor_loss_" + self.name: actor_loss,
                "critic_loss_" + self.name: critic_loss,
            }, {"td_error_" + self.name: td_error}

        return {"critic_loss_" + self.name: critic_loss}, {
            "td_error_" + self.name: td_error
        }

    def train(self, replay_buffer):
        states, goals, actions, n_states, rewards, not_done = replay_buffer.sample()
        return self._train(states, goals, actions, rewards, n_states, goals, not_done)

    def policy(self, state, goal, to_numpy=True):
        if torch.is_tensor(state) is False:
            state = get_tensor(state)
        if torch.is_tensor(goal) is False:
            goal = get_tensor(goal)
        action = self.actor(state, goal)

        if to_numpy:
            return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def policy_with_noise(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(state, goal)

        action = action + self._sample_exploration_noise(action)
        action = torch.min(action, self.actor.scale)
        action = torch.max(action, -self.actor.scale)

        if to_numpy:
            return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def _sample_exploration_noise(self, actions):
        mean = torch.zeros(actions.size()).to(device)
        var = torch.ones(actions.size()).to(device)
        # expl_noise = self.expl_noise - (self.expl_noise/1200) * (self.total_it//10000)
        return torch.normal(mean, self.expl_noise * var)


class HigherController(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005,
        use_reg_mse=False,
        reg_mse_weight=1.0,
        use_prob_trick=False,
    ):
        super().__init__(
            state_dim,
            goal_dim,
            action_dim,
            scale,
            model_path,
            actor_lr,
            critic_lr,
            expl_noise,
            policy_noise,
            noise_clip,
            gamma,
            policy_freq,
            tau,
        )
        self.name = "high"
        self.use_reg_mse = use_reg_mse
        self.use_prob_trick=use_prob_trick
        self.reg_mse_weight = reg_mse_weight
        self.action_dim = action_dim

    def off_policy_corrections(
        self, low_con, batch_size, sgoals, states, actions, candidate_goals=8
    ):
        first_s = [s[0] for s in states]  # First x
        last_s = [s[-1] for s in states]  # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (np.array(last_s) - np.array(first_s))[
            :, np.newaxis, : self.action_dim
        ]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(
            loc=diff_goal,
            scale=0.5 * self.scale[None, None, :],
            size=(batch_size, candidate_goals, original_goal.shape[-1]),
        )
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:, c]
            candidate = (subgoal + states[:, 0, : self.action_dim])[:, None] - states[
                :, :, : self.action_dim
            ]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = policy_actions - true_actions
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape(
            (ncands, batch_size, seq_len) + action_dim
        ).transpose(1, 0, 2, 3)

        logprob = -0.5 * np.sum(np.linalg.norm(difference, axis=-1) ** 2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer: HighReplayBuffer, low_con: TD3Controller, use_correction: bool = True, **kwargs):
        if not self._initialized:
            self._initialize_target_networks()

        (
            states,
            goals,
            actions,
            n_states,
            rewards,
            not_done,
            states_arr,
            actions_arr,
        ) = replay_buffer.sample()
        if use_correction is True:
            actions = self.off_policy_corrections(
                low_con,
                replay_buffer.batch_size,
                actions.cpu().data.numpy(),
                states_arr.cpu().data.numpy(),
                actions_arr.cpu().data.numpy(),
            )

            actions = get_tensor(actions)

        # prob_trick={'mode':'min_max_norm',
        #       'policy':low_con,
        #       'state_arr':states_arr,
        #       'action_arr':actions_arr,
        #       'batch_size':replay_buffer.batch_size
        #       }
        if self.use_reg_mse:
            return self._train(
            states,
            goals,
            actions,
            rewards,
            n_states,
            goals,
            not_done,
            mode ='min_max_norm',
            policy=low_con,
            state_arr=states_arr,
            action_arr=actions_arr,
            batch_size= replay_buffer.batch_size,
        )

        return self._train(
            states,
            goals,
            actions,
            rewards,
            n_states,
            goals,
            not_done,
            **kwargs,
        )


class LowerController(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        scale,
        model_path,
        actor_lr=0.0001,
        critic_lr=0.001,
        expl_noise=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        gamma=0.99,
        policy_freq=2,
        tau=0.005,
        use_backward_loss=False,
        backward_weight=1.0,
    ):
        super().__init__(
            state_dim,
            goal_dim,
            action_dim,
            scale,
            model_path,
            actor_lr,
            critic_lr,
            expl_noise,
            policy_noise,
            noise_clip,
            gamma,
            policy_freq,
            tau,
        )
        self.use_backward_loss = use_backward_loss
        self.backward_weight = backward_weight
        self.name = "low"

    def train(self, replay_buffer, **kwargs):
        if not self._initialized:
            self._initialize_target_networks()

        (
            states,
            sgoals,
            actions,
            n_states,
            n_sgoals,
            rewards,
            not_done,
        ) = replay_buffer.sample()

        return self._train(
            states,
            sgoals,
            actions,
            rewards,
            n_states,
            n_sgoals,
            not_done,
            **kwargs,
        )


class Agent:
    def __init__(self):
        pass

    def set_final_goal(self, fg):
        self.fg = fg

    def step(self, s, env, step, global_step=0, explore=False):
        raise NotImplementedError

    def append(self, step, s, a, n_s, r, d):
        raise NotImplementedError

    def train(self, global_step):
        raise NotImplementedError

    def end_step(self):
        raise NotImplementedError

    def end_episode(self, episode, logger=None):
        raise NotImplementedError

    def evaluate_policy(
        self,
        env,
        eval_episodes=10,
        render=False,
        save_video=False,
        sleep=-1,
        log_loc=False,
    ):
        if save_video:
            from OpenGL import GL  # noqa

            # env = gym.wrappers.Monitor(
            #     env, directory='video',
            #     write_upon_reset=True, force=True, resume=True, mode='evaluation')
            render = False

        success = 0
        rewards = []
        ls = [[] for i in range(eval_episodes)]
        dis = [[] for i in range(eval_episodes)]
        env.evaluate = True
        for e in range(eval_episodes):
            obs = env.reset()
            fg = obs["desired_goal"]
            s = obs["observation"]
            done = False

            reward_episode_sum = 0
            step = 0

            self.set_final_goal(fg)

            while not done:
                if render:
                    env.render()
                if sleep > 0:
                    time.sleep(sleep)
                if log_loc is True:
                    ls[e].append(s[:2])
                    dis[e].append(self.sg[:2])
                a, r, n_s, done = self.step(s, env, step)
                reward_episode_sum += r

                s = n_s
                step += 1
                self.end_step()
            else:
                error_limit = 5
                error = np.sqrt(np.sum(np.square(fg - s[:2])))
                # print(
                #     "Goal, Curr: (%02.2f, %02.2f, %02.2f, %02.2f)     Error:%.2f  Error_limit:%.2f"
                #     % (fg[0], fg[1], s[0], s[1], error, error_limit)
                # )
                rewards.append(reward_episode_sum)
                success += 1 if error <= error_limit else 0
                self.end_episode(e)

        env.evaluate = False
        if log_loc is False:
            return np.array(rewards), success / eval_episodes
        return np.array(rewards), success / eval_episodes, ls, dis


class TD3Agent(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        scale,
        model_path,
        model_save_freq,
        buffer_size,
        batch_size,
        start_training_steps,
    ):
        self.con = TD3Controller(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            scale=scale,
            model_path=model_path,
        )

        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
        self.model_save_freq = model_save_freq
        self.start_training_steps = start_training_steps

    def step(self, s, env, step, global_step=0, explore=False):
        if explore:
            if global_step < self.start_training_steps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s)
        else:
            a = self._choose_action(s)

        obs, r, done, _ = env.step(a)
        n_s = obs["observation"]

        return a, r, n_s, done

    def append(self, step, s, a, n_s, r, d):
        self.replay_buffer.append(s, self.fg, a, n_s, r, d)

    def train(self, global_step):
        return self.con.train(self.replay_buffer)

    def _choose_action(self, s):
        return self.con.policy(s, self.fg)

    def _choose_action_with_noise(self, s):
        return self.con.policy_with_noise(s, self.fg)

    def end_step(self):
        pass

    def end_episode(self, episode, logger=None):
        if logger:
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

    def save(self, episode):
        self.con.save(episode)

    def load(self, episode):
        self.con.load(episode)


class HiroAgent(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        subgoal_dim,
        scale_low,
        start_training_steps,
        model_save_freq,
        model_path,
        buffer_size,
        batch_size,
        buffer_freq,
        train_freq,
        reward_scaling,
        policy_freq_high,
        policy_freq_low,
        use_reg_mse,
        use_backward_loss,
        reg_mse_weight,
        backward_weight,
        use_correction,
        use_prob_trick,
    ):
        self.subgoal = Subgoal(subgoal_dim)
        scale_high = self.subgoal.action_space.high * np.ones(subgoal_dim)

        self.model_save_freq = model_save_freq
        self.use_correction = use_correction
        
        self.high_con = HigherController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high,
            use_reg_mse=use_reg_mse,
            reg_mse_weight=reg_mse_weight,
            use_prob_trick=use_prob_trick
        )

        self.low_con = LowerController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            scale=scale_low,
            model_path=model_path,
            policy_freq=policy_freq_low,
            use_backward_loss=use_backward_loss,
            backward_weight=backward_weight,
        )

        self.replay_buffer_low = LowReplayBuffer(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        self.replay_buffer_high = HighReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq,
        )

        self.buffer_freq = buffer_freq
        self.train_freq = train_freq
        self.reward_scaling = reward_scaling
        self.episode_subreward = 0
        self.sr = 0

        self.buf = [None, None, None, 0, None, None, [], []]
        self.fg = np.array([0, 0])
        self.sg = self.subgoal.action_space.sample()
        self.batch_fg = np.array([0, 0])
        self.start_training_steps = start_training_steps

    def set_final_goal(self, fg):
        self.fg = fg
        bfg = torch.FloatTensor(self.fg)
        bfg = bfg.unsqueeze(0)
        bfg = bfg.repeat(self.replay_buffer_low.batch_size, 1).to(device)  # changed
        self.batch_fg = bfg

    def step(self, s, env, step, global_step=0, explore=False):
        # Lower Level Controller
        if explore:
            # Take random action for start_training_steps
            if global_step < self.start_training_steps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s, self.sg)
        else:
            a = self._choose_action(s, self.sg)

        # Take action
        obs, r, done, _ = env.step(a)
        n_s = obs["observation"]

        # Higher Level Controller
        # Take random action for start_training steps
        if explore:
            if global_step < self.start_training_steps:
                n_sg = self.subgoal.action_space.sample()
            else:
                n_sg = self._choose_subgoal_with_noise(step, s, self.sg, n_s)
        else:
            n_sg = self._choose_subgoal(step, s, self.sg, n_s)

        self.n_sg = n_sg
        # print('s:',n_s[:n_sg.shape[0]])
        return a, r, n_s, done

    def append(self, step, s, a, n_s, r, d):
        self.sr = self.low_reward(s, self.sg, n_s)

        # Low Replay Buffer
        self.replay_buffer_low.append(s, self.sg, a, n_s, self.n_sg, self.sr, float(d))

        # High Replay Buffer
        if _is_update(step, self.buffer_freq, rem=1):
            if len(self.buf[6]) == self.buffer_freq:
                self.buf[4] = s
                self.buf[5] = float(d)
                self.replay_buffer_high.append(
                    state=self.buf[0],
                    goal=self.buf[1],
                    action=self.buf[2],
                    n_state=self.buf[4],
                    reward=self.buf[3],
                    done=self.buf[5],
                    state_arr=np.array(self.buf[6]),
                    action_arr=np.array(self.buf[7]),
                )

            self.buf = [s, self.fg, self.sg, 0, None, None, [], []]

        self.buf[3] += self.reward_scaling * r
        self.buf[6].append(s)
        self.buf[7].append(a)

    def train(self, global_step):
        losses = {}
        td_errors = {}

        if global_step >= self.start_training_steps:
            loss, td_error = self.low_con.train(
                self.replay_buffer_low,
                high_con=self.high_con,
                fg=self.batch_fg,
            )
            losses.update(loss)
            td_errors.update(td_error)
            # whether use correction or not
            if global_step % self.train_freq == 0:
                loss, td_error = self.high_con.train(
                    self.replay_buffer_high, self.low_con, use_correction=self.use_correction
                )
                losses.update(loss)
                td_errors.update(td_error)

        return losses, td_errors

    def _choose_action_with_noise(self, s, sg):
        return self.low_con.policy_with_noise(s, sg)

    def _choose_subgoal_with_noise(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0:  # Should be zero
            sg = self.high_con.policy_with_noise(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def _choose_action(self, s, sg):
        return self.low_con.policy(s, sg)

    def _choose_subgoal(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0:
            sg = self.high_con.policy(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def subgoal_transition(self, s, sg, n_s):
        return s[: sg.shape[0]] + sg - n_s[: sg.shape[0]]

    def low_reward(self, s, sg, n_s):
        abs_s = s[: sg.shape[0]] + sg
        return -np.sqrt(np.sum((abs_s - n_s[: sg.shape[0]]) ** 2))

    def end_step(self):
        self.episode_subreward += self.sr
        self.sg = self.n_sg

    def end_episode(self, episode, logger=None):
        if logger:
            # log
            logger.write("eval/Intrinsic Reward", self.episode_subreward, episode)

            # Save Model
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

        self.episode_subreward = 0
        self.sr = 0
        self.buf = [None, None, None, 0, None, None, [], []]

    def save(self, episode: int):
        self.low_con.save(episode)
        self.high_con.save(episode)

    def load(self, episode: int):
        self.low_con.load(episode)
        self.high_con.load(episode)
