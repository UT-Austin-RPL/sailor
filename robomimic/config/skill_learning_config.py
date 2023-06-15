"""
Config for Skill Learning algorithm.
"""

from robomimic.config.base_config import BaseConfig


class SkillLearningConfig(BaseConfig):
    ALGO_NAME = "skill_learning"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        self.algo.skill_params.skill_len = 10

        self.algo.slowness.window_len = 5
        self.algo.slowness.weight = 1e-3

        self.algo.temp_dist_pred.window_len = 50
        self.algo.temp_dist_pred.weight = 1e-6
        self.algo.temp_dist_pred.layer_dims = (128, 128)

        self.algo.inverse_dyn.weight = 1e-1
        self.algo.inverse_dyn.layer_dims = (1024, 1024)

        # optimization parameters
        self.algo.optim_params.skill.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.skill.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.skill.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.skill.regularization.L2 = 0.00          # L2 regularization strength

        self.algo.optim_params.inverse_dyn.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.inverse_dyn.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.inverse_dyn.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.inverse_dyn.regularization.L2 = 0.00          # L2 regularization strength

        self.algo.optim_params.temp_dist_pred.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.temp_dist_pred.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.temp_dist_pred.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.temp_dist_pred.regularization.L2 = 0.00          # L2 regularization strength

        # loss weights
        self.algo.loss.l2_weight = 1.0      # L2 loss weight
        self.algo.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        # stochastic VAE policy settings
        self.algo.vae.latent_dim = 64                   # VAE latent dimnsion - set to twice the dimensionality of action space
        self.algo.vae.latent_clip = None                # clip latent space when decoding (set to None to disable)
        self.algo.vae.kl_weight = 1e-5                  # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO

        # RNN settings
        self.algo.vae.rnn.hidden_dim = 400      # hidden dimension size    
        self.algo.vae.rnn.rnn_type = "LSTM"     # rnn type - one of "LSTM" or "GRU"
        self.algo.vae.rnn.num_layers = 2        # number of RNN layers that are stacked
        self.algo.vae.rnn.kwargs.bidirectional = False            # rnn kwargs

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = False                        # whether decoder should condition on observation
        self.algo.vae.decoder.reconstruction_sum_across_elements = False    # sum instead of mean for reconstruction loss

        # VAE prior settings
        self.algo.vae.prior.learn = False                                   # learn Gaussian / GMM prior instead of N(0, 1)
        self.algo.vae.prior.is_conditioned = False                          # whether to condition prior on observations
        self.algo.vae.prior.use_gmm = False                                 # whether to use GMM prior
        self.algo.vae.prior.gmm_num_modes = 10                              # number of GMM modes
        self.algo.vae.prior.gmm_learn_weights = False                       # whether to learn GMM weights 
        self.algo.vae.prior.use_categorical = False                         # whether to use categorical prior
        self.algo.vae.prior.categorical_dim = 10                            # the number of categorical classes for each latent dimension
        self.algo.vae.prior.categorical_gumbel_softmax_hard = False         # use hard selection in forward pass
        self.algo.vae.prior.categorical_init_temp = 1.0                     # initial gumbel-softmax temp
        self.algo.vae.prior.categorical_temp_anneal_step = 0.001            # linear temp annealing rate
        self.algo.vae.prior.categorical_min_temp = 0.3                      # lowest gumbel-softmax temp

        self.algo.vae.encoder_layer_dims = (300, 400)                       # encoder MLP layer dimensions
        self.algo.vae.decoder_layer_dims = (300, 400)                       # decoder MLP layer dimensions
        self.algo.vae.prior_layer_dims = (300, 400)                         # prior MLP layer dimensions (if learning conditioned prior)

        self.algo.vae.use_rnn = False