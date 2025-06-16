import os
import argparse


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Train neural network models for radiation emulation"
    )
    parser.add_argument("--model", type=str, default="mlp", help="Name of the model")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of train epochs"
    )
    parser.add_argument(
        "--loss", default="mean_squared_error", help="Train loss function"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "mean_squared_error",
            "mean_absolute_error",
            "mean_squared_logarithmic_error",
        ],
        help="Metrics to be monitored",
    )
    parser.add_argument(
        "--num_cells", type=int, default=81920, help="Number of icon grid cells"
    )
    # MLP parameters
    parser.add_argument(
        "--mlp_units",
        type=int,
        nargs="+",
        default=[128, 256],
        help="Number of hidden units in MLP layers",
    )
    parser.add_argument(
        "--mlp_head_units",
        type=int,
        nargs="+",
        default=[1024, 1024, 1024],
        help="Number of the hidden units in MLP head layers",
    )
    parser.add_argument(
        "--n_ensambels",
        type=int,
        default=20,
        help="Number of ensambles",
    )
    parser.add_argument(
        "--mlp_maxpool_kernel_size",
        type=int,
        default=7,
        help="MaxPoll kernel_size for MlpIgMaxPool class",
    )
    parser.add_argument(
        "--cutoff_height", type=int, default=70, help="Cutoff height for ToA MLP"
    )
    parser.add_argument("--xclipp", type=float, default=1e-9, help="Clipp small values")
    # CNN parameters
    parser.add_argument(
        "--cnn_maxpool_kernel_sizes",
        type=int,
        nargs="+",
        default=[2, 2, 2, 2, 2],
        help="Unet kernel sizes",
    )
    parser.add_argument(
        "--cnn_units",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Unet units",
    )
    # LSTM parameters
    parser.add_argument(
        "--lstm_units",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Number of LSTM units",
    )
    parser.add_argument(
        "--lstm_droprate", type=float, default=0.2, help="LSTM dropout rate"
    )
    # ViT parameters
    parser.add_argument("--vit_dim", type=int, default=256, help="Vit dim")
    parser.add_argument("--vit_patch_size", type=int, default=7, help="Vit patch size")
    parser.add_argument("--vit_depth", type=int, default=4, help="Vit depth")
    parser.add_argument("--vit_heads", type=int, default=6, help="Vit heads")
    parser.add_argument("--vit_dropout", type=float, default=0.0, help="Vit dropout")
    parser.add_argument(
        "--vit_emb_dropout", type=float, default=0.0, help="Vit emb dropout"
    )
    parser.add_argument("--vit_head_dim", type=int, default=64, help="Vit head dim")
    parser.add_argument("--vit_mlp_dim", type=int, default=256, help="Vit mlp dim")

    parser.add_argument(
        "--adversarial_training",
        type=str,
        default=None,
        help="Define adversarial training method",
    )
    parser.add_argument("--pgd_alpha", type=float, default=0.01, help="PGD alpha")
    parser.add_argument("--pgd_eps", type=float, default=0.1, help="PGD epsilon")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("-d", "--data_dir", required=True, help="Path to the dataset")
    parser.add_argument("--cache_dir", default=None, help="Path to cache dataset files")
    parser.add_argument(
        "-s", "--save_dir", required=True, help="Path to save the result"
    )
    parser.add_argument(
        "--rf_model_path", default=None, type=str, help="Random forest model path"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers to load data",
    )
    parser.add_argument(
        "--log_interval", type=int, default=500, help="Logging interval"
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=1.0,
        help="Fraction of data is used for training",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices={"online", "offline", "disabled"},
        help="Operating mode for W&B",
    )
    parser.add_argument(
        "--height_in", type=int, default=70, help="number of height layers"
    )
    parser.add_argument(
        "--height_out", type=int, default=71, help="Number of output layers"
    )
    parser.add_argument(
        "--channel_3d", type=int, default=6, help="Number of 3D channels"
    )
    parser.add_argument(
        "--channel_2d", type=int, default=6, help="Number of 2D channels"
    )
    parser.add_argument(
        "--channel_out", type=int, default=4, help="Number of output channels"
    )
    parser.add_argument(
        "--spatial_subsample",
        type=float,
        default=1.0,
        help="Percent of cells to be sampled from each sample",
    )
    parser.add_argument(
        "--temporal_subsample",
        type=float,
        default=1.0,
        help="Sampling ratio accross time.",
    )
    parser.add_argument(
        "--subsample_ratio",
        type=float,
        default=1.0,
        help="Ratio of dataset to trian, validate or test models models on",
    )
    parser.add_argument(
        "--smoothing_kernel", type=str, default=None, help="smooting kernel"
    )
    parser.add_argument(
        "--feats_2d",
        nargs="+",
        default=[
            "pres_sfc_ecrad_in",
            "cosmu0_ecrad_in",
            "qv_s_ecrad_in",
            "albvisdir_ecrad_in",
            "albnirdir_ecrad_in",
            "tsfctrad_ecrad_in",
        ],
        help="2D input features",
    )
    parser.add_argument(
        "--feats_3d",
        nargs="+",
        default=[
            "clc_ecrad_in",
            "temp_ecrad_in",
            "pres_ecrad_in",
            "qc_ecrad_in",
            "qi_ecrad_in",
            "qv_ecrad_in",
        ],
        help="3D input features",
    )
    parser.add_argument(
        "--feats_out",
        nargs="+",
        default=[
            "lwflx_up_ecrad_out",
            "lwflx_dn_ecrad_out",
            "swflx_up_ecrad_out",
            "swflx_dn_ecrad_out",
        ],
        help="Output features",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="pf",
        choices=["pf", "pfph", "hybrid"],
        help="Normalization type",
    )
    parser.add_argument(
        "--exponential_decay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use exponential decay",
    )
    parser.add_argument(
        "--beta_height", type=int, default=30, help="Hieght for exponetial decay"
    )
    parser.add_argument(
        "--beta_height_lw", type=int, default=30, help="Hieght for exponetial decay"
    )
    parser.add_argument(
        "--beta_height_sw", type=int, default=30, help="Hieght for exponetial decay"
    )
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print summary",
    )
    parser.add_argument(
        "--run_eagerly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Specify whether or not to run tensorflow in eager mode",
    )
    parser.add_argument(
        "--scale_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Specify whether or not to scale output [default False]",
    )
    parser.add_argument(
        "--normalize_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Specify whether or not to normalize output [default False]",
    )
    parser.add_argument(
        "--smoothing", default=None, help="Define the output smoothing method"
    )
    parser.add_argument(
        "--smoothing_sigma",
        default=5,
        type=int,
        help="Sigma value for gaussian smoothing",
    )
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Specify if training takes place [--train: yes --no-train: no]",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Specify if test takes place [--test: yes --no-test: no]",
    )
    parser.add_argument(
        "--serialize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the seralize model",
    )
    parser.add_argument(
        "--test_serialized",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Should I test serialized model?",
    )

    args = parser.parse_args()
    return args
