"""
Config for skill and policy learning (SPL) algorithm.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.skill_learning_config import SkillLearningConfig


class SPLConfig(BaseConfig):
    ALGO_NAME = "spl"

    def train_config(self):
        super(SPLConfig, self).train_config()
        self.train.data_prior = None
        self.train.data_target = None
        self.train.target_hdf5_filter_key = None

    def algo_config(self):
        self.algo.training_mode = "pt"
        self.algo.train_skill = True
        self.algo.train_policy = True

        # ================== Skill Config ==================
        self.algo.skill = SkillLearningConfig().algo  # config for goal learning
        self.algo.skill.unlock()
        self.algo.skill_ds_weights.prior = 1.0
        self.algo.skill_ds_weights.target = 1.0

        # ================== Policy Config ===================
        self.algo.policy.skill_params.model_ckpt_path = None
        self.algo.policy.dataset_conditioned = True # whether to condition the policy on the dataset id (batch must contain dataset id)

        # optimization parameters
        self.algo.policy.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.policy.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.policy.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.policy.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # loss weights
        self.algo.policy.loss.l2_decoding_weight = 0.0
        self.algo.policy.loss.l2_weight = 1.0      # L2 loss weight
        self.algo.policy.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.policy.loss.cos_weight = 0.0     # cosine loss weight

        # MLP network architecture (layers after observation encoder and RNN, if present)
        self.algo.policy.actor_layer_dims = (1024, 1024)

        # stochastic GMM policy settings
        self.algo.policy.gmm.enabled = False                   # whether to train a GMM policy
        self.algo.policy.gmm.num_modes = 5                     # number of GMM modes
        self.algo.policy.gmm.min_std = 0.0001                  # minimum std output from network
        self.algo.policy.gmm.std_activation = "softplus"       # activation to use for std output from policy net
        self.algo.policy.gmm.low_noise_eval = True             # low-std at test-time 

        # RNN policy settings
        self.algo.policy.rnn.enabled = False       # whether to train RNN policy
        self.algo.policy.rnn.horizon = 10          # unroll length for RNN - should usually match train.seq_length
        self.algo.policy.rnn.hidden_dim = 400      # hidden dimension size    
        self.algo.policy.rnn.rnn_type = "LSTM"     # rnn type - one of "LSTM" or "GRU"
        self.algo.policy.rnn.num_layers = 2        # number of RNN layers that are stacked
        self.algo.policy.rnn.open_loop = False     # if True, action predictions are only based on a single observation (not sequence)
        self.algo.policy.rnn.kwargs.bidirectional = False            # rnn kwargs
        
        self.algo.policy_ds_weights.prior = 1.0
        self.algo.policy_ds_weights.target = 1.0
        self.algo.policy_ds_weights.retrieval = 1.0

        # ================== Retrieval Config ===================
        self.algo.retrieval.enabled = False
        self.algo.retrieval.data = None
        self.algo.retrieval.filter_key = None
        self.algo.retrieval.type = None
        self.algo.retrieval.model_ckpt_path = None
        self.algo.retrieval.bs = None

        self.algo.retrieval.num_ret_cands = 250000
        self.algo.retrieval.num_target_samples = 2500
        self.algo.retrieval.selection_frac = 0.10