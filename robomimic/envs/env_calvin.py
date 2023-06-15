"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy
import random

from calvin_env.envs.play_table_env import get_env

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB

import calvin_env
import os
import hydra
import omegaconf


class EnvCalvin(EB.EnvBase):
    """Wrapper class for calvin environments (https://github.com/mees/calvin)"""
    def __init__(
            self,
            env_name,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
            postprocess_visual_obs=True,
            **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        self._init_kwargs = deepcopy(kwargs)
        self._env_name = env_name
        self._current_obs = None
        self._current_reward = None
        self._current_done = None
        self._done = None

        calvin_env_base_path = os.path.abspath(os.path.join(os.path.dirname(calvin_env.__file__), os.pardir))
        self.env = get_env(
            os.path.join(calvin_env_base_path, '../dataset/task_D_D/training'),
            show_gui=render,
            cam_sizes=self._init_kwargs.get("cam_sizes", None),
            scene=self._init_kwargs.get("scene", None),
        )
        self.tasks = hydra.utils.instantiate(
            omegaconf.OmegaConf.load(os.path.join(calvin_env_base_path, 'conf/tasks/new_playtable_tasks.yaml'))
        )
        self._start_info = self.env.get_info()

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        action = deepcopy(action)
        if action[-1] < 0:
            action[-1] = -1
        else:
            action[-1] = 1

        obs, reward, done, info = self.env.step(action)
        reward, done = self._get_reward_and_done(info)

        self._current_obs = obs
        self._current_reward = reward
        self._current_done = done
        return self.get_observation(obs), reward, self.is_done(), info

    def _get_reward_and_done(self, info):
        task_success = False
        task_success_dict = self.tasks.get_task_info(self._start_info, info)
        if self._env_name in task_success_dict:
            task_success = True

        reward = float(task_success)
        done = bool(task_success)

        return reward, done

    def reset(self, state=None):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        if state is not None:
            self._current_obs = self.env.reset(robot_obs=state[0:-24], scene_obs=state[-24:])
        else:
            state = self.get_reset_state()
            if state is not None:
                self._current_obs = self.reset(state=state)
                for _ in range(10):
                    self.step(np.zeros(7))
            else:
                self._current_obs = self.env.reset()
            self._start_info = self.env.get_info()
        self._current_reward = None
        self._current_done = None
        return self.get_observation(self._current_obs)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains:
                - states (np.ndarray): initial state of the mujoco environment

        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        return self.reset(state["states"])

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        if mode =="human":
            return self.env.render(mode=mode, height=height, width=width, **kwargs)
        if mode == "rgb_array":
            return self.env.render(mode="rgb_array", height=height, width=width)
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            ob (np.array): current flat observation vector to wrap and provide as a dictionary.
                If not provided, uses self._current_obs.
        """
        if obs is None:
            assert self._current_obs is not None
            obs = self._current_obs

        ret = {}

        # proprio observations
        for k in [
            'robot_obs', 'scene_obs',
            'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
            'robot0_eef_euler', 'robot0_joint_qpos', 'robot0_prev_gripper_action',
            'block_red', 'block_blue', 'block_pink', 'non_blocks',
            'eef_to_block_red_pos', 'eef_to_block_blue_pos', 'eef_to_block_pink_pos',
        ]:
            ret[k] = obs[k]

        # image observations
        for k in ['rgb_static', 'rgb_gripper']:
            if k not in ObsUtils.OBS_KEYS_TO_MODALITIES:
                continue

            im = obs['rgb_obs'][k]
            if self.postprocess_visual_obs:
                ret[k] = ObsUtils.process_obs(obs=im, obs_key=k)
            else:
                ret[k] = im

        scene = self._init_kwargs.get("scene", None)
        env_id = np.zeros(4)
        if scene == 'calvin_scene_A':
            env_id[0] = 1.0
        elif scene == 'calvin_scene_B':
            env_id[1] = 1.0
        elif scene == 'calvin_scene_C':
            env_id[2] = 1.0
        elif scene == 'calvin_scene_D' or scene is None:
            env_id[3] = 1.0
        else:
            raise ValueError("EnvCalvin: invalid CALVIN scene")
        ret['env_id'] = env_id

        ret['dataset_id'] = np.ones((10,))

        return ret

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        return dict(states=np.concatenate([self._current_obs['robot_obs'], self._current_obs['scene_obs']]))

    def get_reward(self):
        """
        Get current reward.
        """
        if self._current_reward is None:
            info = self.env.get_info()
            reward, _ = self._get_reward_and_done(info)
            return reward

        return self._current_reward

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        assert self._current_done is not None
        return self._current_done

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        if self._current_reward is None:
            info = self.env.get_info()
            reward, _ = self._get_reward_and_done(info)
            success = bool(reward)
        else:
            success = bool(self._current_reward)
        return {"task": success}

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        # return self.env.action_space.shape[0]
        return 7

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.CALVIN_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(cls, env_name, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. For gym environments, input arguments (other than @env_name)
        are ignored, since environments are mostly pre-configured.

        Args:
            env_name (str): name of gym environment to create

        Returns:
            env (EnvGym instance)
        """

        # make sure to initialize obs utils so it knows which modalities are image modalities.
        # For currently supported gym tasks, there are no image observations.
        obs_modality_specs = {
            "obs": {
                "low_dim": [
                    'robot_obs', 'scene_obs',
                    'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                    'robot0_eef_euler', 'robot0_joint_qpos', 'robot0_prev_gripper_action',
                ],
                "rgb": ["rgb_static"] if "rgb_static" in camera_names else [],
                "rgb2": ["rgb_gripper"] if "rgb_gripper" in camera_names else [],
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        cam_sizes = dict(
            static=dict(
                height=camera_height,
                width=camera_width,
            ),
            gripper=dict(
                height=camera_height,
                width=camera_width,
            ),
        )
        if "cam_sizes" in kwargs:
            del kwargs["cam_sizes"]

        return cls(env_name=env_name, postprocess_visual_obs=False, cam_sizes=cam_sizes, **kwargs)

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return ()

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)

    def get_reset_state(self):
        def get_env_state_for_initial_condition(initial_condition):
            robot_obs = np.array(
                [
                    0.02586889,
                    -0.2313129,
                    0.5712808,
                    3.09045411,
                    -0.02908596,
                    1.50013585,
                    0.07999963,
                    -1.21779124,
                    1.03987629,
                    2.11978254,
                    -2.34205014,
                    -0.87015899,
                    1.64119093,
                    0.55344928,
                    1.0,
                ]
            )
            block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
            block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
            block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])

            table_left = np.array([-0.25, -0.10, 4.59990009e-01])

            # block_table = [
            #     # np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
            #     np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
            # ]
            table_low = np.array([0.0, -0.12, 4.59990009e-01])
            table_high = np.array([0.30, -0.02, 4.59990009e-01])
            # we want to have a "deterministic" random seed for each initial condition
            # np.random.shuffle(block_table)

            scene_obs = np.zeros(24)
            if initial_condition["slider"] == "left":
                scene_obs[0] = 0.28
            elif initial_condition["slider"] == "right":
                scene_obs[0] = 0.0

            if initial_condition["drawer"] == "open":
                scene_obs[1] = 0.22
            elif initial_condition["drawer"] == "open_partial":
                scene_obs[1] = np.random.uniform(0.10, 0.22)

            if initial_condition["lightbulb"] == 1:
                scene_obs[3] = 0.088
            scene_obs[4] = initial_condition["lightbulb"]
            scene_obs[5] = initial_condition["led"]
            scene_obs[2] = (0.02 * initial_condition["led"]) # hack: press down on button to simulate led on

            for (k, (s,e)) in zip(["red_block", "blue_block", "pink_block"], [(6,9), (12,15), (18,21)]):
                if k not in initial_condition:
                    continue
                if initial_condition[k] == "slider_right":
                    scene_obs[s:e] = block_slider_right
                elif initial_condition[k] == "slider_left":
                    scene_obs[s:e] = block_slider_left
                elif initial_condition[k] == "table":
                    scene_obs[s:e] = np.random.uniform(table_low, table_high)
                elif initial_condition[k] == "table_left":
                    scene_obs[s:e] = table_left
                elif initial_condition[k] == "table_left_rand":
                    scene_obs[s:e] = np.random.uniform(
                        [-0.30, -0.11, 4.59990009e-01],
                        [-0.25, -0.06, 4.59990009e-01]
                    )
                elif initial_condition[k] == "table_center_rand":
                    scene_obs[s:e] = np.random.uniform(
                        [0.0, -0.11, 4.59990009e-01],
                        [0.05, -0.06, 4.59990009e-01]
                    )
                elif initial_condition[k] == "table_right_rand":
                    scene_obs[s:e] = np.random.uniform(
                        [0.20, -0.11, 4.59990009e-01],
                        [0.25, -0.06, 4.59990009e-01]
                    )
                elif initial_condition[k] == "drawer_rand":
                    scene_obs[s:e] = np.random.uniform(
                        [0.125, -0.10, 0.363],
                        [0.175, -0.05, 0.363]
                    )
                else:
                    raise ValueError

                scene_obs[e+2] = np.random.uniform(*block_rot_z_range)

            return np.concatenate([robot_obs, scene_obs])

        block_choices = random.choice([
            {
                'blue': "slider_right",
                'red': "slider_left",
            },
            {
                'blue': "slider_left",
                'red': "slider_right",
            },
            {
                'blue': "table_left",
                'red': "slider_left",
            },
            {
                'blue': "slider_left",
                'red': "table_left",
            },
            {
                'blue': "slider_right",
                'red': "table_left",
            },
            {
                'blue': "table_left",
                'red': "slider_right",
            },
        ])

        if self._env_name == 'cleanup':
            pos_choices = ['left', 'center', 'right']
            random.shuffle(pos_choices)
            return get_env_state_for_initial_condition(dict(
                slider="left",
                drawer="closed",
                lightbulb=1,
                led=1,
                red_block="table_{}_rand".format(pos_choices[0]),
                pink_block="table_{}_rand".format(pos_choices[1]),
                blue_block="table_{}_rand".format(pos_choices[2]),
            ))
        elif self._env_name == 'cleanup_pink_easy':
            return get_env_state_for_initial_condition(dict(
                slider="left",
                drawer="closed",
                lightbulb=1,
                led=1,
                red_block=block_choices["red"],
                pink_block="table",
                blue_block=block_choices["blue"],
            ))
        elif self._env_name == 'setup':
            pos_choices = ['slider_left', 'slider_right']
            random.shuffle(pos_choices)
            return get_env_state_for_initial_condition(dict(
                slider=["right", "left"][random.choice(range(2))],
                drawer="closed",
                lightbulb=0,
                led=0,
                red_block=pos_choices[0],
                blue_block=pos_choices[1],
                pink_block="drawer_rand",
            ))

        return None
