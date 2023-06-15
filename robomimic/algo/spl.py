"""
Implementation of skill and policy learning.
"""

from collections import OrderedDict
import textwrap
from copy import deepcopy
import torch
import torch.nn as nn

import robomimic.models.base_nets as BaseNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo.skill_learning import SkillLearning

from robomimic.algo import register_algo_factory_func, Algo, PolicyAlgo

@register_algo_factory_func("spl")
def algo_config_to_class(algo_config):
    return SPL, {}

class SPL(Algo):
    def __init__(
            self,
            algo_config,
            obs_config,
            global_config,
            obs_key_shapes,
            ac_dim,
            device
    ):
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device

        self._check_config()

        ckpt_path = self.algo_config.policy.skill_params.model_ckpt_path
        if ckpt_path is not None:
            print("Reading skill checkpoint from {}".format(ckpt_path))

            from robomimic.utils.file_utils import policy_from_checkpoint
            ckpt = policy_from_checkpoint(
                ckpt_path=ckpt_path,
                device=self.device,
            )[0].policy

            if type(ckpt) == SkillLearning:
                self.skill = ckpt
            else:
                self.skill = ckpt.skill

            skill_config = self.skill.algo_config

            # set frame_stack for the skill config
            skill_global_config = self.skill.global_config
            skill_global_config_lock_state = skill_config._get_lock_state()
            skill_global_config.unlock()
            skill_global_config.train.frame_stack = self.global_config.train.frame_stack
            skill_global_config._set_lock_state(skill_global_config_lock_state)

            # update skill config
            config_lock_state = self.algo_config._get_lock_state()
            self.algo_config.unlock()
            self.algo_config.skill = skill_config
            self.algo_config._set_lock_state(config_lock_state)

            # update observation config
            config_lock_state = self.obs_config._get_lock_state()
            self.obs_config.unlock()
            self.obs_config.update(self.skill.obs_config)

            self.obs_config._set_lock_state(config_lock_state)

            self.skill.optim_params = deepcopy(skill_config.optim_params)
            self.skill._create_optimizers()
            self.skill.nets.train()
        else:
            self.skill = SkillLearning(
                algo_config=algo_config.skill,
                obs_config=obs_config,
                global_config=global_config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=ac_dim,
                device=device
            )

        self.policy = BC_Skill_RNN(
            skill_model=self.skill,
            algo_config=algo_config.policy,
            obs_config=obs_config,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device
        )

    def _check_config(self):
        assert self.training_mode in ["pt", "ft"]

        ckpt_path = self.algo_config.policy.skill_params.model_ckpt_path
        prior_ds = self.global_config.train.data_prior
        target_ds = self.global_config.train.data_target
        train_skill = self.algo_config.train_skill
        train_policy = self.algo_config.train_policy
        if self.training_mode == "pt":
            assert ckpt_path is None
            assert (prior_ds is not None) and (target_ds is None)
            assert train_skill
        elif self.training_mode == "ft":
            assert ckpt_path is not None
            assert (target_ds is not None)
            assert train_policy

    def process_batch_for_training(self, batch):
        input_batch = dict()

        input_batch["skill"] = {
            k: self.skill.process_batch_for_training(batch[k]) for k in {"prior", "target"} if k in batch
        }

        batch_keys_for_policy = []
        if self.training_mode == "ft":
            batch_keys_for_policy = ["target", "retrieval"]
        elif self.training_mode == "pt":
            if self.algo_config.train_policy:
                batch_keys_for_policy = ["prior"]
        else:
            raise ValueError

        if len(batch_keys_for_policy) > 0:
            input_batch["policy"] = {
                k: self.policy.process_batch_for_training(
                    batch[k],
                    detach_skills=True,
                    dataset_id=1 if k == "target" else 0
                ) for k in ["target", "retrieval", "prior"] if k in batch_keys_for_policy and k in batch
            }

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        info = dict(skill=dict(), policy=None)

        train_skill = (not validate) and (self.algo_config.train_skill)
        train_policy = (not validate) and (self.algo_config.train_policy)

        is_first_skill_backprpop = True
        for k in ["prior", "target"]:
            if k not in batch["skill"]:
                continue
            info["skill"][k] = self.skill.train_on_batch(
                batch["skill"][k], epoch,
                validate=(not train_skill),
                dont_step=True,
                dont_zero_grad=(not is_first_skill_backprpop),
                loss_weight=self.algo_config.skill_ds_weights[k],
            )
            is_first_skill_backprpop = False

        if "policy" in batch:
            info["policy"] = dict()

            is_first_policy_backprpop = True
            for k in ["target", "retrieval", "prior"]:
                if k not in batch["policy"]:
                    continue
                info["policy"][k] = self.policy.train_on_batch(
                    batch["policy"][k], epoch,
                    validate=(not train_policy),
                    dont_step=True,
                    dont_zero_grad=(not is_first_policy_backprpop),
                    loss_weight=self.algo_config.policy_ds_weights[k],
                )
                is_first_policy_backprpop = False
        else:
            assert train_policy is False

        if train_policy:
            self.policy.optimizers["policy"].step()

        if train_skill:
            self.skill.optimizers["skill"].step()
            self.skill.optimizers["inverse_dyn"].step()
            self.skill.optimizers["temp_dist_pred"].step()

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        loss = 0.

        log = dict()

        for k in ["prior", "target"]:
            if k not in info["skill"]:
                continue

            skill_log = self.skill.log_info(info["skill"][k])
            skill_log = dict(("Skill/{}/".format(k) + log_k, log_v) for log_k, log_v in skill_log.items())
            log.update(skill_log)
            loss += skill_log["Skill/{}/Loss".format(k)]

        if info["policy"] is not None:
            for k in ["prior", "target", "retrieval"]:
                if k not in info["policy"]:
                    continue

                policy_log = self.policy.log_info(info["policy"][k])
                policy_log = dict(("Policy/{}/".format(k) + log_k, log_v) for log_k, log_v in policy_log.items())
                log.update(policy_log)
                loss += policy_log["Policy/{}/Loss".format(k)]

        log["Loss"] = loss
        return log

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """
        self.skill.on_epoch_end(epoch)
        self.policy.on_epoch_end(epoch)

    def reset(self):
        self.policy.reset()
        self.skill.reset()

    def get_action(self, obs_dict, goal_dict=None):
        return self.policy.get_action(obs_dict, goal_dict)

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self.skill.set_eval()
        self.policy.set_eval()

    def set_train(self):
        """
        Prepare networks for training.
        """
        self.skill.set_train()
        self.policy.set_train()

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return dict(
            skill=self.skill.serialize(),
            policy=self.policy.serialize(),
        )

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.
        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.skill.deserialize(model_dict["skill"])
        self.policy.deserialize(model_dict["policy"])

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        """
        msg = str(self.__class__.__name__)
        return msg + "Skill:\n" + textwrap.indent(self.skill.__repr__(), '  ') + \
               "\n\nPolicy:\n" + textwrap.indent(self.policy.__repr__(), '  ')

    @property
    def training_mode(self):
        return self.algo_config.training_mode


class BC_Skill_RNN(PolicyAlgo):
    def __init__(self, skill_model=None, *args, **kwargs):
        self._skill_model = skill_model
        super(BC_Skill_RNN, self).__init__(*args, **kwargs)
        self._check_config()

    def _check_config(self):
        assert self.fs == self.bc_h

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        from robomimic.utils.file_utils import policy_from_checkpoint

        if self.algo_config.dataset_conditioned:
            self.obs_shapes["dataset_id"] = [10]

        if self._skill_model is None:
            raise NotImplementedError

        self.nets = nn.ModuleDict()

        if self._use_gmm:
            policy_class = PolicyNets.RNNGMMActorNetwork
            additional_policy_kwargs = dict(
                use_tanh=False,
                disable_transforms=True,
                num_modes=self.algo_config.gmm.num_modes,
                min_std=self.algo_config.gmm.min_std,
                std_activation=self.algo_config.gmm.std_activation,
                low_noise_eval=self.algo_config.gmm.low_noise_eval,
            )
        else:
            policy_class = PolicyNets.RNNActorNetwork
            additional_policy_kwargs = dict()

        self.nets["policy"] = policy_class(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self._skill_model.latent_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            use_tanh=False,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
            **additional_policy_kwargs,
        )

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch, detach_skills=True, dataset_id=None):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["goal_obs_bc"] = None

        if dataset_id is not None:
            first_obs_key = list(batch["obs"].keys())[0]
            B = batch["obs"][first_obs_key].shape[0]
            T = batch["obs"][first_obs_key].shape[1]
            batch["obs"]["dataset_id"] = torch.tensor([[[dataset_id]]]).repeat(B, T, 10)

        input_batch["obs_bc"] = {k: batch["obs"][k][:, self.fs-1:self.fs-1+self.bc_h, :] for k in batch["obs"]}

        skills_list = []
        for start_base in range(0, self.bc_h):
            start_ob = self.fs - 1 + start_base
            input_batch["obs_skill"] = {k: batch["obs"][k][:, start_ob:start_ob+self.skill_h, :] for k in batch["obs"]}
            start_ac = start_base
            input_batch["actions_skill"] = batch["actions"][:, start_ac:start_ac+self.skill_h, :]

            skill_batch = dict(
                obs=input_batch["obs_skill"],
                actions=input_batch["actions_skill"],
            )
            skill_batch = TensorUtils.to_device(TensorUtils.to_float(skill_batch), self.device)
            
            skill_encoding = self._skill_model.encode(
                skill_batch, mode="encoder"
            )
            skills = skill_encoding['mean']

            if detach_skills:
                skills = skills.detach()
            
            skills_list.append(skills)

        input_batch["skills"] = torch.stack(skills_list, dim=1)

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(
            self, batch, epoch,
            validate=False,
            dont_step=False,
            dont_zero_grad=False,
            loss_weight=1.0
    ):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC_Skill_RNN, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)
            info["losses_not_detached"] = losses

            if not validate:
                step_info = self._train_step(
                    losses,
                    loss_weight=loss_weight,
                    dont_step=dont_step,
                    dont_zero_grad=dont_zero_grad,
                )
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()

        if self._use_gmm:
            dists = self.nets["policy"].forward_train(
                obs_dict=batch["obs_bc"],
                goal_dict=batch["goal_obs_bc"],
            )

            # # make sure that this is a batch of multivariate action distributions, so that
            # # the log probability computation will be correct
            # assert len(dists.batch_shape) == 1
            log_probs = dists.log_prob(batch["skills"])

            stds = dists.component_distribution.base_dist.scale
            # assert np.all(stds >= 0)
            predictions['policy_std_mean'] = torch.mean(stds)

            predictions['log_probs'] = log_probs
            predictions["skills"] = dists.sample()
        else:
            skills = self.nets["policy"](obs_dict=batch["obs_bc"], goal_dict=batch["goal_obs_bc"])
            predictions["skills"] = skills

        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()

        if self._use_gmm:
            nll_loss = -predictions["log_probs"].mean()
            losses["log_probs"] = -nll_loss
            assert self.algo_config.loss.l2_decoding_weight == 0.0
            skill_losses = [
                nll_loss,
            ]
            assert self.algo_config.loss.l2_decoding_weight == 0.0
            skill_loss = sum(skill_losses)
            losses["skill_loss"] = skill_loss
        else:
            s_target = batch["skills"]
            skills = predictions["skills"]
            losses["l2_loss"] = nn.MSELoss()(skills, s_target)
            losses["l1_loss"] = nn.SmoothL1Loss()(skills, s_target)
            skill_losses = [
                self.algo_config.loss.l2_weight * losses["l2_loss"],
                self.algo_config.loss.l1_weight * losses["l1_loss"],
            ]
            assert self.algo_config.loss.l2_decoding_weight == 0.0
            skill_loss = sum(skill_losses)
            losses["skill_loss"] = skill_loss

        return losses

    def _train_step(self, losses, dont_step=False, dont_zero_grad=False, loss_weight=1.0):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()

        if not dont_zero_grad:
            self.optimizers["policy"].zero_grad()

        if loss_weight != 0.0:
            (losses["skill_loss"] * loss_weight).backward(retain_graph=False)
        info["policy_grad_norms"] = TorchUtils.get_grad_norms(self.nets["policy"])

        if not dont_step:
            self.optimizers["policy"].step()

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC_Skill_RNN, self).log_info(info)
        log["Loss"] = info["losses"]["skill_loss"].item()
        if "log_probs" in info["losses"]:
            log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]

        # gmm-specific logging
        if "policy_std_mean" in info["predictions"]:
            log["Policy_Std"] = info["predictions"]["policy_std_mean"].item()

        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        obs_dict_latest = {k: self._get_tensor_at_timestep(obs_dict[k], -1) for k in obs_dict.keys()}

        # update the skill
        if self._ep_step_count % self.skill_h == 0:
            if self._ep_step_count == 0:
                self._current_skill, _ = self.nets["policy"].forward_step(
                    obs_dict=obs_dict_latest, goal_dict=None, rnn_state=None,
                )
            else:
                bc_policy_output = self.nets["policy"](
                    obs_dict=obs_dict, goal_dict=None,
                )
                self._current_skill = self._get_tensor_at_timestep(bc_policy_output, -1)

            self._skill_model.reset()

        action = self._skill_model.decode_step(dict(
            z=self._current_skill, obs=obs_dict_latest,
        ))

        self._ep_step_count += 1

        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._ep_step_count = 0
        self._current_skill = None

    @property
    def _use_gmm(self):
        return self.algo_config.gmm.enabled

    @property
    def skill_h(self):
        return self._skill_model.skill_len

    @property
    def fs(self):
        return self.global_config.train.frame_stack

    @property
    def bc_h(self):
        return self.algo_config.rnn.horizon

    def _get_tensor_at_timestep(self, tensor, t):
        if tensor.ndim < 3:
            return tensor

        return tensor[:,t]

    def _encode_skill_batch(self, skill_batch):
        skill_encoding = self._skill_model.encode(skill_batch)
        skills = skill_encoding['mean'].detach()

        return skills