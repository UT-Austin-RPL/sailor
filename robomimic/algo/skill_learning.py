from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

import robomimic.models.skill_nets as SkillNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, Algo


@register_algo_factory_func("skill_learning")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    return SkillLearning, {}


class SkillLearning(Algo):
    """
    Skill learning with a VAE.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["skill"] = SkillNets.VAESkill(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            skill_len=self.skill_len,
            rnn_config=self.algo_config.vae.rnn,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )

        self.nets["inverse_dyn"] = SkillNets.InverseDynNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.latent_dim,
            mlp_layer_dims=self.algo_config.inverse_dyn.layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets["temp_dist_pred"] = SkillNets.TempDistPredictionNetwork(
            ac_dim=self.latent_dim,
            mlp_layer_dims=self.algo_config.temp_dist_pred.layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
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
        start_ob = self.fs - 1
        input_batch["obs"] = {k: batch["obs"][k][:, start_ob:start_ob+self.skill_len, :] for k in batch["obs"]}
        input_batch["goal_obs"] = None
        input_batch["actions"] = batch["actions"][:, :self.skill_len, :]
        input_batch["t"] = batch["t"]

        # process batch for slowness and temp_dist prediction losses
        for pf in ["slowness", "temp_dist"]:
            input_batch["obs_{}".format(pf)] = {
                k: batch["obs_{}".format(pf)][k][:, start_ob:start_ob+self.skill_len, :] for k in batch["obs_{}".format(pf)]
            }
            input_batch["goal_obs_{}".format(pf)] = None
            input_batch["actions_{}".format(pf)] = batch["actions_{}".format(pf)][:, :self.skill_len, :]
            input_batch["t_{}".format(pf)] = batch["t_{}".format(pf)]

        # process batch for dynamics losses
        input_batch["obs_dyn"] = {k: batch["obs"][k][:, start_ob, :] for k in batch["obs"]}
        input_batch["next_obs_dyn"] = {k: batch["obs"][k][:, start_ob + self.skill_len - 1, :] for k in batch["obs"]}

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(
            self, batch, epoch,
            validate=False,
            dont_step=False,
            dont_zero_grad=False,
            loss_weight=1.0
    ):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["skill"].set_gumbel_temperature(temperature)

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(SkillLearning, self).train_on_batch(batch, epoch, validate=validate)
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
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["skill"].forward_train(**vae_inputs)

        # compute slowness loss
        vae_inputs_slowness = dict(
            actions=batch["actions_slowness"],
            obs_dict=batch["obs_slowness"],
            goal_dict=batch["goal_obs_slowness"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )
        vae_outputs_slowness = self.nets["skill"].forward_train(**vae_inputs_slowness)
        slowness_loss = F.mse_loss(
            vae_outputs["encoder_params"]["mean"],
            vae_outputs_slowness["encoder_params"]["mean"]
        )

        # compute elements for temporal distance prediction loss
        vae_inputs_temp_dist = dict(
            actions=batch["actions_temp_dist"],
            obs_dict=batch["obs_temp_dist"],
            goal_dict=batch["goal_obs_temp_dist"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )
        vae_outputs_temp_dist = self.nets["skill"].forward_train(**vae_inputs_temp_dist)
        z_1 = vae_outputs["encoder_params"]["mean"]
        z_2 = vae_outputs_temp_dist["encoder_params"]["mean"]
        temp_dist_pred = self.nets["temp_dist_pred"].forward(
            z_1=z_1,
            z_2=z_2
        )
        temp_dist_loss = F.mse_loss(
            temp_dist_pred,
            batch["t"] - batch["t_temp_dist"]
        )

        # calculate inverse dynamics loss
        inverse_dyn_pred = self.nets["inverse_dyn"].forward(
            obs_dict=batch["obs_dyn"],
            next_obs_dict=batch["next_obs_dyn"]
        )
        z_invdyn = vae_outputs["encoder_params"]["mean"]
        inverse_dyn_loss = F.mse_loss(
            inverse_dyn_pred,
            z_invdyn
        )

        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],

            ARE_encoder=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],

            kl_loss=vae_outputs["kl_loss"],
            slowness_loss=slowness_loss,
            inverse_dyn_loss=inverse_dyn_loss,
            temp_dist_loss=temp_dist_loss,
        )

        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
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

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["ARE_encoder"]
        slowness_loss = predictions["slowness_loss"]
        inverse_dyn_loss = predictions["inverse_dyn_loss"]
        temp_dist_loss = predictions["temp_dist_loss"]
        skill_loss = recons_loss \
                     + self.algo_config.vae.kl_weight * kl_loss \
                     + self.algo_config.slowness.weight * slowness_loss \
                     + self.algo_config.inverse_dyn.weight * inverse_dyn_loss \
                     + self.algo_config.temp_dist_pred.weight * temp_dist_loss
        return OrderedDict(
            skill_loss=skill_loss,
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            slowness_loss=slowness_loss,
            inverse_dyn_loss=inverse_dyn_loss,
            temp_dist_loss=temp_dist_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = Algo.log_info(self, info)
        
        # losses
        log["Loss"] = info["losses"]["skill_loss"].item()
        log["ARE_encoder"] = info["predictions"]["ARE_encoder"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Slowness_Loss"] = info["losses"]["slowness_loss"].item()
        log["InvDyn_Loss"] = info["losses"]["inverse_dyn_loss"].item()
        log["TempDistPred_Loss"] = info["losses"]["temp_dist_loss"].item()
        
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["skill"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        if "skill_grad_norms" in info:
            log["Skill_Grad_Norms"] = info["skill_grad_norms"]
        return log

    def _train_step(self, losses, dont_step=False, dont_zero_grad=False, loss_weight=1.0):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        info = OrderedDict()

        if not dont_zero_grad:
            self.optimizers["skill"].zero_grad()
            self.optimizers["inverse_dyn"].zero_grad()
            self.optimizers["temp_dist_pred"].zero_grad()

        (losses["skill_loss"] * loss_weight).backward(retain_graph=False)

        if not dont_step:
            self.optimizers["skill"].step()
            self.optimizers["inverse_dyn"].step()
            self.optimizers["temp_dist_pred"].step()

        return info

    def encode(self, batch, mode="encoder"):
        assert mode in ["encoder", "invdyn", "invdyn_prior"]

        if mode == "encoder":
            return self.nets["skill"].encode(actions=batch["actions"], obs_dict=batch["obs"])
        elif mode == "invdyn":
            first_obs = batch.get("first_obs", TensorUtils.index_at_time(batch["obs"], ind=0))
            last_obs = batch.get("last_obs", TensorUtils.index_at_time(batch["obs"], ind=-1))
            output = self.nets["inverse_dyn"].forward(
                obs_dict=first_obs,
                next_obs_dict=last_obs,
            )
            # TODO: logvar
            return {
                'mean': output
            }
        elif mode == "invdyn_prior":
            assert self.algo_config.vae.prior.is_conditioned is True
            assert self.algo_config.vae.prior.use_gmm is False
            first_obs = batch.get("first_obs", TensorUtils.index_at_time(batch["obs"], ind=0))
            last_obs = batch.get("last_obs", TensorUtils.index_at_time(batch["obs"], ind=-1))
            output = self.nets["skill"]._vae.nets["prior"].forward(
                batch_size=len(batch["actions"]),
                obs_dict=first_obs,
                goal_dict=last_obs,
            )
            return {
                'mean': output['means'][:,0,:],
            }
        else:
            raise ValueError

    def decode(self, batch):
        return self.nets["skill"].decode(z=batch["z"], obs_dict=batch["obs"], t=self.skill_len)

    def decode_step(self, batch):
        z = batch["z"]
        obs_dict = batch["obs"]
        output, self._rnn_state = self.nets["skill"].decode_step(
            z=z, obs_dict=obs_dict, rnn_state=self._rnn_state,
        )
        action = output['action']
        return action

    def reset(self):
        self._rnn_state = None
        self._action_plan = deque()

    @property
    def latent_dim(self):
        return self.nets["skill"]._vae.latent_dim

    @property
    def skill_len(self):
        return self.algo_config.skill_params.skill_len

    @property
    def fs(self):
        return self.global_config.train.frame_stack
