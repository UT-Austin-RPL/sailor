"""
Contains torch Modules for policy networks. These networks take an
observation dictionary as input (and possibly additional conditioning,
such as subgoal or goal dictionaries) and produce action predictions,
samples, or distributions as outputs. Note that actions
are assumed to lie in [-1, 1], and most networks will have a final
tanh activation to help ensure this range.
"""
from collections import OrderedDict
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.models.base_nets as BaseNets

from robomimic.models.base_nets import Module
from robomimic.models.obs_nets import MIMO_MLP
from robomimic.models.vae_nets import VAE_RNN


class VAESkill(Module):
    """
    A VAE that models a distribution of actions conditioned on observations.
    The VAE prior and decoder are used at test-time as the policy.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        skill_len,
        rnn_config,
        encoder_layer_dims,
        decoder_layer_dims,
        latent_dim,
        device,
        decoder_is_conditioned=False, #True,
        decoder_reconstruction_sum_across_elements=False,
        latent_clip=None,
        prior_learn=False,
        prior_is_conditioned=False,
        prior_layer_dims=(),
        prior_use_gmm=False,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=False,
        prior_use_categorical=False,
        prior_categorical_dim=10,
        prior_categorical_gumbel_softmax_hard=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes. 

            visual_feature_dimension (int): feature dimension to encode images into.

            visual_core_class (str): specifies Visual Backbone network for encoding images.

            visual_core_kwargs (dict): arguments to pass to @visual_core_class. 

            obs_randomizer_class (str): specifies a Randomizer class for the input modality

            obs_randomizer_kwargs (dict): kwargs for the observation randomizer

            use_spatial_softmax (bool): if True, introduce a spatial softmax layer at
                the end of the visual backbone network, resulting in a sharp bottleneck
                representation for visual inputs.

            spatial_softmax_kwargs (dict): arguments to pass to spatial softmax layer.

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.
        """
        super(VAESkill, self).__init__()

        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim

        self._skill_len = skill_len

        additional_vae_kwargs = dict()
        additional_vae_kwargs.update(
            BaseNets.rnn_args_from_config(rnn_config)
        )
        action_shapes = OrderedDict(action=(self.ac_dim,))

        # ensure VAE decoder will squash actions into [-1, 1]
        output_squash = ['action']
        output_scales = OrderedDict(action=1.)

        self._vae = VAE_RNN(
            input_shapes=action_shapes,
            output_shapes=action_shapes,
            encoder_layer_dims=encoder_layer_dims,
            decoder_layer_dims=decoder_layer_dims,
            latent_dim=latent_dim,
            device=device,
            condition_shapes=self.obs_shapes,
            decoder_is_conditioned=decoder_is_conditioned,
            decoder_reconstruction_sum_across_elements=decoder_reconstruction_sum_across_elements,
            latent_clip=latent_clip,
            output_squash=output_squash,
            output_scales=output_scales,
            prior_learn=prior_learn,
            prior_is_conditioned=prior_is_conditioned,
            prior_condition_first_last=True,
            prior_layer_dims=prior_layer_dims,
            prior_use_gmm=prior_use_gmm,
            prior_gmm_num_modes=prior_gmm_num_modes,
            prior_gmm_learn_weights=prior_gmm_learn_weights,
            prior_use_categorical=prior_use_categorical,
            prior_categorical_dim=prior_categorical_dim,
            prior_categorical_gumbel_softmax_hard=prior_categorical_gumbel_softmax_hard,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
            **additional_vae_kwargs
        )

    def encode(self, actions, obs_dict=None, goal_dict=None):
        """
        Args:
            actions (torch.Tensor): a batch of actions

            obs_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to the observation modalities 
                used for conditioning in either the decoder or the prior (or both).

            goal_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities.

        Returns:
            posterior params (dict): dictionary with the following keys:

                mean (torch.Tensor): posterior encoder means

                logvar (torch.Tensor): posterior encoder logvars
        """
        inputs = OrderedDict(action=actions)

        return self._vae.encode(inputs=inputs, conditions=obs_dict, goals=goal_dict)

    def decode(self, obs_dict=None, goal_dict=None, z=None, n=None, t=None):
        """
        Thin wrapper around @VaeNets.VAE implementation.

        Args:
            obs_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. Only needs to be provided if @decoder_is_conditioned
                or @z is None (since the prior will require it to generate z).

            goal_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities.

            z (torch.Tensor): if provided, these latents are used to generate
                reconstructions from the VAE, and the prior is not sampled.

            n (int): this argument is used to specify the number of samples to 
                generate from the prior. Only required if @z is None - i.e.
                sampling takes place

        Returns:
            recons (dict): dictionary of reconstructed inputs (this will be a dictionary
                with a single "action" key)
        """
        additional_args = dict(t=self._skill_len)

        return self._vae.decode(conditions=obs_dict, goals=goal_dict, z=z, n=n, **additional_args)

    def decode_step(self, obs_dict=None, goal_dict=None, z=None, n=None, rnn_state=None):
        return self._vae.decode_step(conditions=obs_dict, goals=goal_dict, z=z, n=n, rnn_state=rnn_state)

    def sample_prior(self, obs_dict=None, goal_dict=None, n=None):
        """
        Thin wrapper around @VaeNets.VAE implementation.

        Args:
            n (int): this argument is used to specify the number
                of samples to generate from the prior.

            obs_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. Only needs to be provided if @prior_is_conditioned.

            goal_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities.

        Returns:
            z (torch.Tensor): latents sampled from the prior
        """
        return self._vae.sample_prior(n=n, conditions=obs_dict, goals=goal_dict)

    def set_gumbel_temperature(self, temperature):
        """
        Used by external algorithms to schedule Gumbel-Softmax temperature,
        which is used during reparametrization at train-time. Should only be
        used if @prior_use_categorical is True.
        """
        self._vae.set_gumbel_temperature(temperature)

    def get_gumbel_temperature(self):
        """
        Return current Gumbel-Softmax temperature. Should only be used if
        @prior_use_categorical is True.
        """
        return self._vae.get_gumbel_temperature()

    def output_shape(self, input_shape=None):
        """
        This implementation is required by the Module superclass, but is unused since we 
        never chain this module to other ones.
        """
        return [self.ac_dim]

    def forward_train(self, actions, obs_dict, goal_dict=None, freeze_encoder=False):
        """
        A full pass through the VAE network used during training to construct KL
        and reconstruction losses. See @VAE class for more info.

        Args:
            actions (torch.Tensor): a batch of actions

            obs_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to the observation modalities 
                used for conditioning in either the decoder or the prior (or both).

            goal_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities.

        Returns:
            vae_outputs (dict): a dictionary that contains the following outputs.

                encoder_params (dict): parameters for the posterior distribution
                    from the encoder forward pass

                encoder_z (torch.Tensor): latents sampled from the encoder posterior

                decoder_outputs (dict): action reconstructions from the decoder

                kl_loss (torch.Tensor): KL loss over the batch of data

                reconstruction_loss (torch.Tensor): reconstruction loss over the batch of data
        """
        vae_inputs = OrderedDict(action=actions)
        vae_outputs = OrderedDict(action=actions)

        return self._vae.forward(
            inputs=vae_inputs,
            outputs=vae_outputs,
            conditions=obs_dict, 
            goals=goal_dict,
            freeze_encoder=freeze_encoder)

    def forward(self, obs_dict, goal_dict=None, z=None):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations
            z (torch.Tensor): if not None, use the provided batch of latents instead
                of sampling from the prior

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        n = None
        if z is None:
            # prior will be sampled - so we must provide number of samples explicitly
            mod = list(obs_dict.keys())[0]
            n = obs_dict[mod].shape[0]
        return self.decode(obs_dict=obs_dict, goal_dict=goal_dict, z=z, n=n)["action"]


class InverseDynNetwork(MIMO_MLP):
    def __init__(
            self,
            obs_shapes,
            ac_dim,
            mlp_layer_dims,
            goal_shapes=None,
            encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            visual_feature_dimension (int): feature dimension to encode images into.

            visual_core_class (str): specifies Visual Backbone network for encoding images.

            visual_core_kwargs (dict): arguments to pass to @visual_core_class.

            obs_randomizer_class (str): specifies observation randomizer class

            obs_randomizer_kwargs (dict): kwargs for observation randomizer (e.g., CropRandomizer)

            use_spatial_softmax (bool): if True, introduce a spatial softmax layer at
                the end of the visual backbone network, resulting in a sharp bottleneck
                representation for visual inputs.

            spatial_softmax_kwargs (dict): arguments to pass to spatial softmax layer.

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.
        """
        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        observation_group_shapes["next_obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(InverseDynNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_MLP, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        return OrderedDict(action=(self.ac_dim,))

    def output_shape(self, input_shape=None):
        return [self.ac_dim]

    def forward(self, obs_dict, next_obs_dict):
        actions = super(InverseDynNetwork, self).forward(obs=obs_dict, next_obs=next_obs_dict)["action"]
        return actions

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)


class TempDistPredictionNetwork(MIMO_MLP):
    def __init__(
            self,
            ac_dim,
            mlp_layer_dims,
            encoder_kwargs=None,
            num_discrete_bins=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            visual_feature_dimension (int): feature dimension to encode images into.

            visual_core_class (str): specifies Visual Backbone network for encoding images.

            visual_core_kwargs (dict): arguments to pass to @visual_core_class.

            obs_randomizer_class (str): specifies observation randomizer class

            obs_randomizer_kwargs (dict): kwargs for observation randomizer (e.g., CropRandomizer)

            use_spatial_softmax (bool): if True, introduce a spatial softmax layer at
                the end of the visual backbone network, resulting in a sharp bottleneck
                representation for visual inputs.

            spatial_softmax_kwargs (dict): arguments to pass to spatial softmax layer.

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.
        """
        self.ac_dim = ac_dim

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["z"] = OrderedDict()
        observation_group_shapes["z"]["z_1"] = self.ac_dim
        observation_group_shapes["z"]["z_2"] = self.ac_dim

        self._is_goal_conditioned = False
        self.goal_shapes = OrderedDict()

        self._num_discrete_bins = num_discrete_bins

        output_shapes = self._get_output_shapes()
        super(TempDistPredictionNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_MLP, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        if self._num_discrete_bins is not None:
            return OrderedDict(temp_dist=(self._num_discrete_bins,))
        else:
            return OrderedDict(temp_dist=(1,))

    def output_shape(self, input_shape=None):
        if self._num_discrete_bins is not None:
            return [(self._num_discrete_bins,)]
        else:
            return [(1,)]

    def forward(self, z_1, z_2):
        z = dict(
            z_1=z_1,
            z_2=z_2,
        )
        actions = super(TempDistPredictionNetwork, self).forward(z=z)["temp_dist"]
        return actions

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)

