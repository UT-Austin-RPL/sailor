from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    assert args.mod == "im", "currently this config script only support image experiments"
    generator = get_generator(
        algo_name="sailorft",
        config_file=os.path.join(base_path, "robomimic/exps/templates/sailor.json"),
        args=args,
    )

    # env specific settings
    if args.env == 'calvin':
        generator.add_param(
            key="train.data_prior",
            name="dsprior",
            group=1,
            values=[
                os.path.join(base_path, 'datasets/calvin/play/play_ABCD_im84.hdf5'),
            ],
            value_names=[
                "play",
            ],
            hidename=True,
        )
        generator.add_param(
            key="train.hdf5_filter_key",
            name="priords-fkey",
            group=1,
            values=["env_ABCD"],
            hidename=True,
        )

        generator.add_param(
            key="train.data_target",
            name="dstest",
            group=-5,
            values=[
                os.path.join(base_path, path) for path in [
                    'datasets/calvin/cleanup/05_10_im84.hdf5',
                    'datasets/calvin/setup/05_10_im84.hdf5',
                ]
            ],
            value_names=[
                "cleanup",
                "setup",
            ],
            hidename=False,
        )
        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-1,
            values=[
                1000,
            ],
        )
        generator.add_param(
            key="train.target_hdf5_filter_key",
            name="ndemos",
            group=2,
            values=["30_demos"],
            value_names=["30"],
            hidename=False,
        )

        generator.add_param(
            key="algo.policy.skill_params.model_ckpt_path",
            name="skill",
            group=51,
            values=[os.path.join(base_path, "expdata/calvin/im/sailor_pt", path) for path in [
                "add-path-here",
            ]],
            value_names=[
                "ckpt",
            ],
            hidename=True,
        )

    elif args.env == 'kitchen':
        generator.add_param(
            key="train.data_target",
            name="dstest",
            group=-5,
            values=[
                os.path.join(base_path, path) for path in [
                    "datasets/franka_kitchen/kitchen_test.hdf5",
                ]
            ],
            value_names=[
                "kitchen_test",
            ],
            hidename=True,
        )

        generator.add_param(
            key="train.data_prior",
            name="dsprior",
            group=51,
            values=[
                os.path.join(base_path, "datasets/franka_kitchen/kitchen_no_test.hdf5"),
                os.path.join(base_path, "datasets/franka_kitchen/kitchen_no_microwave.hdf5"),
            ],
            value_names=[
                "kitchen_no_test",
                "kitchen_no_microwave",
            ],
            hidename=True,
        )
        generator.add_param(
            key="algo.policy.skill_params.model_ckpt_path",
            name="skill",
            group=51,
            values=[os.path.join(base_path, 'expdata/kitchen/im/sailor_pt', path) for path in [
                "add-path-here",
                "add-path-here",
            ]],
            value_names=[
                "kitchen-no-test-seed1",
                "kitchen-no-microwave-seed1",
            ],
            hidename=False,
        )
    else:
        raise ValueError

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "../expdata/{env}/{mod}/sailor_ft".format(
                env=args.env,
                mod=args.mod,
            )
        ],
    )

    generator.add_param(
        key="algo.policy.rnn.hidden_dim",
        name="policyrnnhiddendim",
        group=-1,
        values=[
            1000,
        ],
        hidename=True,
    )
    generator.add_param(
        key="train.batch_size",
        name="bs",
        group=-1,
        values=[
            16,
        ],
        hidename=True,
    )

    generator.add_param(
        key="algo.training_mode",
        name="",
        group=-1,
        values=["ft"]
    )
    generator.add_param(
        key="algo.train_policy",
        name="",
        group=-1,
        values=[True],
    )

    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[200],
    )
    generator.add_param(
        key="experiment.save.epochs",
        name="",
        group=-1,
        values=[[]],
    )
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=-1,
        values=[
            10
        ],
    )
    
    generator.add_param(
        key="experiment.save.enabled",
        name="",
        group=-1,
        values=[False],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=-1,
        values=[10],
    )

    if args.debug:
        generator.add_param(
            key="algo.retrieval.num_ret_cands",
            name="",
            group=-1,
            values=[256],
            value_names=[""],
        )
        generator.add_param(
            key="algo.retrieval.num_target_samples",
            name="",
            group=-1,
            values=[32],
            value_names=[""],
        )

    generator.add_param(
        key="train.seed",
        name="seed",
        group=-10,
        values=[
            1,
            # 2,
            # 3,
        ],
    )

    return generator


if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)

