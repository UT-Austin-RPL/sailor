from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    assert args.mod == "im", "currently this config script only support image experiments"
    generator = get_generator(
        algo_name="sailorpt",
        config_file=os.path.join(base_path, "robomimic/exps/templates/sailor.json"),
        args=args,
    )

    # env specific skill settings
    if args.env == 'calvin':
        generator.add_param(
            key="train.data_prior",
            name="dsprior",
            group=1,
            values=[
                os.path.join(base_path, "datasets/calvin/play/play_ABCD_im84.hdf5"),
            ],
            value_names=[
                "play",
            ],
            hidename=True,
        )
        generator.add_param(
            key="train.hdf5_filter_key",
            name="priords-fkey",
            group=56,
            values=[
                "env_ABCD",
            ],
            hidename=True,
        )
    elif args.env == 'kitchen':
        generator.add_param(
            key="train.data_prior",
            name="dsprior",
            group=-4,
            values=[
                os.path.join(base_path, "datasets/franka_kitchen/kitchen_no_microwave.hdf5"),
                os.path.join(base_path, "datasets/franka_kitchen/kitchen_no_test.hdf5"),
            ],
            value_names=[
                "kitchen_no_microwave",
                "kitchen_no_test",
            ],
        )
    else:
        raise ValueError

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "../expdata/{env}/{mod}/sailor_pt".format(
                env=args.env,
                mod=args.mod,
            )
        ],
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
        key="algo.skill.vae.rnn.hidden_dim",
        name="skillrnnhiddim",
        group=9,
        values=[
            1000,
        ],
        hidename=True,
    )


    generator.add_param(
        key="algo.training_mode",
        name="",
        group=-1,
        values=["pt"]
    )
    generator.add_param(
        key="algo.train_skill",
        name="",
        group=-1,
        values=[True],
    )
    generator.add_param(
        key="algo.train_policy",
        name="trainpolicy",
        group=-1,
        values=[
            False,
        ],
        hidename=True,
    )
    generator.add_param(
        key="algo.retrieval.enabled",
        name="",
        group=-1,
        values=[
            False,
        ],
        hidename=True,
    )
    generator.add_param(
        key="experiment.rollout.enabled",
        name="",
        group=-1,
        values=[False],
        value_names=[""],
    )

    if args.env == "kitchen":
        generator.add_param(
            key="experiment.save.epochs",
            name="",
            group=-1,
            values=[
                [200, 300, 600],
            ],
        )
    elif args.env == "calvin":
        generator.add_param(
            key="experiment.save.epochs",
            name="",
            group=-1,
            values=[
                [200, 400, 600],
            ],
        )
    else:
        raise NotImplementedError
        
    generator.add_param(
        key="train.seed",
        name="seed",
        group=-10,
        values=[
            1,
        ],
    )

    return generator


if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
