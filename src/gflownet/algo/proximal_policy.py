import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from gflownet.config import Config
from gflownet.envs.graph_building_env import (
    GraphBuildingEnv,
    GraphBuildingEnvContext,
    generate_forward_trajectory,
    GraphActionCategorical,
)
from gflownet.utils.misc import get_worker_device

from .graph_sampling import GraphSampler
from .trajectory_balance import TrajectoryBalance


class PPO(TrajectoryBalance):
    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ):
        """Advantage Actor-Critic implementation, see
          Asynchronous Methods for Deep Reinforcement Learning,
          Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim
          Harley, David Silver, Koray Kavukcuoglu
          Proceedings of The 33rd International Conference on Machine Learning, 2016

        Hyperparameters used:
        illegal_action_logreward: float, log(R) given to the model for non-sane end states or illegal actions

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        cfg: Config
            The experiment configuration

        """
        self.ctx = ctx
        self.env = env
        self.global_cfg = cfg
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.illegal_action_logreward = cfg.algo.illegal_action_logreward
        self.entropy_coef = cfg.algo.a2c.entropy
        self.gamma = cfg.algo.a2c.gamma
        self.invalid_penalty = cfg.algo.a2c.penalty
        assert self.gamma == 1
        self.bootstrap_own_reward = False
        # Experimental flags
        self.sample_temp = 1
        self.do_q_prime_correction = False
        self.graph_sampler = GraphSampler(ctx, env, self.max_len, self.max_nodes, self.sample_temp)

    def set_is_eval(self, is_eval: bool):
        self.is_eval = is_eval

    def create_training_data_from_own_samples(
        self, model: nn.Module, n: int, cond_info: Tensor, random_action_prob: float
    ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: nn.Module
           The model being sampled
        graphs: List[Graph]
            List of N Graph endpoints
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(model, n, cond_info, random_action_prob)
        return data

    def create_training_data_from_graphs(self, graphs):
        """Generate trajectories from known endpoints

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        return [{"traj": generate_forward_trajectory(i)} for i in graphs]

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj["traj"]]
        actions = [
            self.ctx.GraphAction_to_ActionIndex(g, a)
            for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        return batch

    def get_inner_loop_constants(self, model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        dev = batch.x.device
        cond_info = batch.cond_info
        # Collect trajectories
        num_trajs = int(batch.traj_lens.shape[0])
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # Get current model policy
        # Need to make sure p_b is not parametrized to get tuple of len = 2
        policy, _ = model(batch, cond_info[batch_idx])
        # Run the policy on the batch
        logprob = policy.log_prob(batch.actions)
        # Get rewards
        rewards = torch.exp(batch.log_rewards)
        return model, cond_info, num_trajs, batch_idx, logprob, rewards, batch, dev

    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        return None

    def compute_batch_losses_ppo(self, model: nn.Module, cond_info, num_trajs, batch_idx, logprob, rewards, batch, dev):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        # Randomly select 16 indices from batch_idx
        # random_idx = torch.randint(0, batch_idx[-1] + 1, (16,), device=dev)
        # # Create a binary mask for the selected indices
        # masked_idx = torch.zeros_like(batch_idx, dtype=torch.bool)
        # for idx in random_idx:
        #     masked_idx |= batch_idx == idx
        # Masking the minibatch actions
        # batch.actions = batch.actions[masked_idx]
        # batch.x = batch.x[masked_idx]
        ### Mask node features
        num_graphs_to_sample = 16
        sampled_indices = torch.randperm(64)[:num_graphs_to_sample]
        node_mask = torch.isin(batch.batch, sampled_indices)
        # Filter the node features (x)
        batch.x = batch.x[node_mask]
        batch.batch = batch.batch[node_mask]
        ##################
        new_ptr = [0]
        for graph_idx in sampled_indices:
            num_nodes_in_graph = (batch.batch == graph_idx).sum().item()
            new_ptr.append(new_ptr[-1] + num_nodes_in_graph)
        batch.ptr = torch.tensor(new_ptr, dtype=torch.long)

        # Create a mask for the edges that belong to the sampled graphs
        # edge_index_batch tells us which graph each edge belongs to
        edge_mask = torch.isin(batch.edge_index_batch, sampled_indices)

        # Filter the edge_index and edge_attr
        batch.edge_index = batch.edge_index[:, edge_mask]
        batch.edge_attr = batch.edge_attr[edge_mask]

        # Update the edge_index_batch and edge_index_ptr
        batch.edge_index_batch = batch.edge_index_batch[edge_mask]
        new_edge_ptr = [0]
        for graph_idx in sampled_indices:
            num_edges_in_graph = (batch.edge_index_batch == graph_idx).sum().item()
            new_edge_ptr.append(new_edge_ptr[-1] + num_edges_in_graph)
        batch.edge_index_ptr = torch.tensor(new_edge_ptr, dtype=torch.long)
        ####
        # Filter the actions
        # actions are associated with graphs, so we filter based on sampled_indices
        action_mask = torch.isin(torch.arange(batch.actions.size(0)), sampled_indices)
        batch.actions = batch.actions[action_mask]
        inner_policy, inner_state_preds = model(batch, cond_info[masked_idx])
        inner_logprob = inner_policy.log_prob(batch.actions)
        #     # Get Value function
        inner_V = inner_state_preds[:, 0]
        # Group inner_V values by batch index
        # Calculate the ratio function. We have log(policy) so we are using the exp
        ratio = torch.exp(logprob - inner_logprob)
        # ratio = logprob - inner_logprob
        # Iterate over traj to get values for each state in each traj seperately
        # This is important to compute the advantage at different time steps in the trajectory
        advantages = []
        # This is same everywhere
        rewards_per_state = []
        for bi in range(num_trajs):
            for t in range(batch.traj_lens[bi]):
                r = rewards[bi]
                vs_t = inner_V[t]
                vs_t1 = inner_V[t + 1] if t + 1 < batch.traj_lens[bi] else 0
                advantage = r + vs_t1 - vs_t
                advantages.append(advantage)
                rewards_per_state.append(r)
        # Ensure correct dims
        assert len(advantages) == len(inner_V)
        advantages = torch.tensor(advantages, device=dev)
        surr1 = ratio * advantages
        # TODO: Make epsilon hyperparam?
        epsilon = 0.2
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
        action_loss = -torch.min(surr1, surr2).mean()
        G = rewards[batch_idx] + (1 - batch.is_valid[batch_idx]) * self.invalid_penalty
        value_loss = 0.5 * (G - inner_V).pow(2).mean()
        # TODO: Make this hparam
        entropy_coeff = 0.2
        # For some reason entropy loss gives nan values, atm are not taken into account
        entropy = -inner_policy.entropy()
        entropy_loss = entropy_coeff * entropy[~torch.isnan(entropy)].mean()
        # breakpoint()
        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.

        # V_loss = A.pow(2).mean()
        # # pol_objective = (log_probs * A.detach()).mean() + self.entropy_coef * policy.entropy().mean()
        # pol_loss = -pol_objective

        # loss = V_loss + pol_loss
        invalid_mask = 1 - batch.is_valid
        info = {
            "V_loss": value_loss.item(),
            # "A": advantages.item(),
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "loss": total_loss.item(),
        }
        if not torch.isfinite(total_loss).all():
            raise ValueError("loss is not finite")
        return total_loss, info
