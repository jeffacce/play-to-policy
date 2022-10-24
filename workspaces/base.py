import logging
from collections import deque
from pathlib import Path
from omegaconf import OmegaConf
import einops
import gym
from gym.wrappers import RecordVideo
import hydra
import numpy as np
import torch
from models.action_ae.generators.base import GeneratorDataParallel
from models.latent_generators.latent_generator import LatentGeneratorDataParallel
import utils
import wandb


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        utils.set_seed_everywhere(cfg.seed)

        self.env = gym.make(cfg.env.name)
        if self.env.class_name() == "TimeLimit":
            # wrap with the correct num steps
            self.env = self.env.env
            self.env = gym.wrappers.TimeLimit(self.env, cfg.num_eval_steps)
        if cfg.record_video:
            self.env = RecordVideo(
                self.env,
                video_folder=self.work_dir,
                episode_trigger=lambda x: x % 1 == 0,
            )

        # Create the model
        self.action_ae = None
        self.obs_encoding_net = None
        self.state_prior = None
        if not self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()

        if self.cfg.goal_conditional:
            self.goal_fn = hydra.utils.instantiate(self.cfg.env.goal_fn, cfg)
        self.wandb_run = wandb.init(
            dir=self.work_dir,
            project=cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logging.info("wandb run url: %s", self.wandb_run.get_url())
        self.epoch = 0
        self.load_snapshot()

        # Set up rolling window contexts.
        self.window_size = cfg.window_size
        self.obs_context = deque(maxlen=self.window_size)
        self.goal_context = deque(maxlen=self.window_size)

        if self.cfg.flatten_obs:
            self.env = gym.wrappers.FlattenObservation(self.env)

        if self.cfg.plot_interactions:
            self._setup_plots()

        if self.cfg.start_from_seen:
            self._setup_starting_state()

        self._setup_action_sampler()

    def _init_action_ae(self):
        if self.action_ae is None:  # possibly already initialized from snapshot
            self.action_ae = hydra.utils.instantiate(
                self.cfg.action_ae, _recursive_=False
            ).to(self.device)
            if self.cfg.data_parallel:
                self.action_ae = GeneratorDataParallel(self.action_ae)

    def _init_obs_encoding_net(self):
        if self.obs_encoding_net is None:  # possibly already initialized from snapshot
            self.obs_encoding_net = hydra.utils.instantiate(self.cfg.encoder)
            self.obs_encoding_net = self.obs_encoding_net.to(self.device)
            if self.cfg.data_parallel:
                self.obs_encoding_net = torch.nn.DataParallel(self.obs_encoding_net)

    def _init_state_prior(self):
        if self.state_prior is None:  # possibly already initialized from snapshot
            self.state_prior = hydra.utils.instantiate(
                self.cfg.state_prior,
                latent_dim=self.action_ae.latent_dim,
                vocab_size=self.action_ae.num_latents,
            ).to(self.device)
            if self.cfg.data_parallel:
                self.state_prior = LatentGeneratorDataParallel(self.state_prior)
            self.state_prior_optimizer = self.state_prior.get_optimizer(
                learning_rate=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=tuple(self.cfg.betas),
            )

    def _setup_plots(self):
        raise NotImplementedError

    def _setup_starting_state(self):
        raise NotImplementedError

    def _setup_action_sampler(self):
        def sampler(actions):
            idx = np.random.randint(len(actions))
            return actions[idx]

        self.sampler = sampler

    def _start_from_known(self):
        raise NotImplementedError

    def run_single_episode(self, goal_idx=None):
        action_history = []
        obs = self.env.reset()
        last_obs = obs
        if self.cfg.start_from_seen:
            obs = self._start_from_known()
        if self.cfg.goal_conditional:
            goal = self.goal_fn(obs, goal_idx, 0)
        else:
            goal = None
        action = self._get_action(obs, goal=goal, sample=True)
        done = False
        total_reward = 0
        action_history.append(action)
        for i in range(self.cfg.num_eval_steps + 1):
            if self.cfg.plot_interactions:
                self._plot_obs_and_actions(obs, action, done)
            if done:
                result = self._report_result_upon_completion(goal_idx)
                break
            if self.cfg.enable_render:
                self.env.render(mode="human")
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if obs is None:
                obs = last_obs  # use cached observation in case of `None` observation
            else:
                last_obs = obs  # cache valid observation
            if self.cfg.goal_conditional == "onehot":
                goal = self.goal_fn(obs, goal_idx, i)
            action = self._get_action(obs, goal=goal, sample=True)
            action_history.append(action)
        logging.info(f"Total reward: {total_reward}")
        logging.info(f"Final info: {info}")
        logging.info(f"Result: {result}")
        return total_reward, action_history, info, result

    def _report_result_upon_completion(self, goal_idx=None):
        pass

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        print(obs, chosen_action, done)
        raise NotImplementedError

    def _get_action(self, obs, goal=None, sample=True):
        with utils.eval_mode(
            self.action_ae, self.obs_encoding_net, self.state_prior, no_grad=True
        ):
            obs = torch.from_numpy(obs).float().to(self.cfg.device).unsqueeze(0)
            enc_obs = self.obs_encoding_net(obs).squeeze(0)
            enc_obs = einops.repeat(
                enc_obs, "obs -> batch obs", batch=self.cfg.action_batch_size
            )
            # Now, add to history. This automatically handles the case where
            # the history is full.
            self.obs_context.append(enc_obs)
            if self.cfg.goal_conditional:
                if self.cfg.goal_conditional == "onehot":
                    # goal: Tensor[G]
                    goal = torch.Tensor(goal).float().to(self.cfg.device)
                    goal = einops.repeat(
                        goal, "goal -> batch goal", batch=self.cfg.action_batch_size
                    )
                    self.goal_context.append(goal)
                    goal_seq = torch.stack(tuple(self.goal_context), dim=0)
                elif self.cfg.goal_conditional == "future":
                    # goal: Tensor[T, O], a slice of future observations
                    goal = torch.Tensor(goal).float().to(self.cfg.device).unsqueeze(0)
                    enc_goal = self.obs_encoding_net(goal).squeeze(0)
                    enc_goal = einops.repeat(
                        enc_goal,
                        "seq goal -> seq batch goal",
                        batch=self.cfg.action_batch_size,
                    )
                    # We are not using goal_context for this case, since
                    # the future slice is already a sequence.
                    goal_seq = enc_goal
            else:
                goal_seq = None
            if self.cfg.use_state_prior:
                enc_obs_seq = torch.stack(tuple(self.obs_context), dim=0)  # type: ignore
                action_latents = self.state_prior.generate_latents(
                    enc_obs_seq,
                    torch.ones_like(enc_obs_seq).mean(dim=-1),
                    goal=goal_seq,
                )
            else:
                action_latents = self.action_ae.sample_latents(
                    num_latents=self.cfg.action_batch_size
                )
            actions = self.action_ae.decode_actions(
                latent_action_batch=action_latents,
                input_rep_batch=enc_obs,
            )
            actions = actions.cpu().numpy()
            # Take the last action; this assumes that SPiRL (CVAE) already selected the first action
            actions = actions[:, -1, :]
            # Now action shape is (batch_size, action_dim)
            if sample:
                actions = self.sampler(actions)
            return actions

    def run(self):
        rewards = []
        infos = []
        results = []
        if self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()
        for i in range(self.cfg.num_eval_goals):
            for j in range(self.cfg.num_eval_eps):
                reward, actions, info, result = self.run_single_episode(goal_idx=i)
                rewards.append(reward)
                infos.append(info)
                results.append(result)
                torch.save(actions, Path.cwd() / f"goal_{i}_actions_{j}.pth")
        self.env.close()
        logging.info(rewards)
        return rewards, infos, results

    @property
    def snapshot(self):
        return Path(self.cfg.load_dir or self.work_dir) / "snapshot.pt"

    def load_snapshot(self):
        keys_to_load = ["action_ae", "obs_encoding_net", "state_prior"]
        with self.snapshot.open("rb") as f:
            payload = torch.load(f, map_location=self.device)
        loaded_keys = []
        for k, v in payload.items():
            if k in keys_to_load:
                loaded_keys.append(k)
                self.__dict__[k] = v.to(self.cfg.device)

        if len(loaded_keys) != len(keys_to_load):
            raise ValueError(
                "Snapshot does not contain the following keys: "
                f"{set(keys_to_load) - set(loaded_keys)}"
            )
