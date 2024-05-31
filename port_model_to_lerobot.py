from pathlib import Path

from omegaconf import OmegaConf
import torch

from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.utils.utils import init_hydra_config

PATH_TO_ORIGINAL_WEIGHTS = "/home/alexander/Projects/act/outputs/sim_transfer_cube_human_vae/policy_last.ckpt"
PATH_TO_CONFIG = (
    "outputs/train/act_aloha_sim_transfer_cube_human/.hydra/config.yaml"
)
PATH_TO_SAVE_NEW_WEIGHTS = "/tmp/act"


cfg = init_hydra_config(PATH_TO_CONFIG)

policy = make_policy(hydra_cfg=cfg, dataset_stats=make_dataset(cfg).stats)

state_dict = torch.load(PATH_TO_ORIGINAL_WEIGHTS)

# Remove keys based on what they start with.

start_removals = [
    # There is a bug that means the pretrained model doesn't even use the final decoder layers.
    *[f"model.transformer.decoder.layers.{i}" for i in range(1, 7)],
    "model.is_pad_head.",
]

for to_remove in start_removals:
    for k in list(state_dict.keys()):
        if k.startswith(to_remove):
            del state_dict[k]


# Replace keys based on what they start with.

start_replacements = [
    ("model.query_embed.weight", "model.pos_embed.weight"),
    ("model.pos_table", "model.vae_encoder_pos_enc"),
    ("model.pos_embed.weight", "model.decoder_pos_embed.weight"),
    ("model.encoder.", "model.vae_encoder."),
    ("model.encoder_action_proj.", "model.vae_encoder_action_input_proj."),
    ("model.encoder_joint_proj.", "model.vae_encoder_robot_state_input_proj."),
    ("model.latent_proj.", "model.vae_encoder_latent_output_proj."),
    ("model.latent_proj.", "model.vae_encoder_latent_output_proj."),
    ("model.input_proj.", "model.encoder_img_feat_input_proj."),
    ("model.input_proj_robot_state", "model.encoder_robot_state_input_proj"),
    ("model.latent_out_proj.", "model.encoder_latent_input_proj."),
    ("model.transformer.encoder.", "model.encoder."),
    ("model.transformer.decoder.", "model.decoder."),
    ("model.backbones.0.0.body.", "model.backbone."),
    ("model.additional_pos_embed.weight", "model.encoder_robot_and_latent_pos_embed.weight"),
    ("model.cls_embed.weight", "model.vae_encoder_cls_embed.weight"),
]

for to_replace, replace_with in start_replacements:
    for k in list(state_dict.keys()):
        if k.startswith(to_replace):
            k_ = replace_with + k.removeprefix(to_replace)
            state_dict[k_] = state_dict[k]
            del state_dict[k]


state_dict["normalize_inputs.buffer_observation_images_top.mean"] = torch.tensor(
    [[[0.4850]], [[0.4560]], [[0.4060]]]
)
state_dict["normalize_inputs.buffer_observation_images_top.std"] = torch.tensor(
    [[[0.2290]], [[0.2240]], [[0.2250]]]
)
state_dict["normalize_inputs.buffer_observation_state.mean"] = torch.tensor(
    [
        -0.0074,
        -0.6319,
        1.0357,
        -0.0503,
        -0.4620,
        -0.0747,
        0.4747,
        -0.0362,
        -0.3320,
        0.9039,
        -0.2206,
        -0.3101,
        -0.2348,
        0.6842,
    ]
)
state_dict["normalize_inputs.buffer_observation_state.std"] = torch.tensor(
    [
        0.0122,
        0.2975,
        0.1673,
        0.0473,
        0.1486,
        0.0879,
        0.3175,
        0.1050,
        0.2793,
        0.1809,
        0.2660,
        0.3047,
        0.5299,
        0.2550,
    ]
)
state_dict["unnormalize_outputs.buffer_action.mean"] = torch.tensor(
    [
        -0.0076,
        -0.6282,
        1.0313,
        -0.0466,
        -0.4721,
        -0.0745,
        0.3739,
        -0.0372,
        -0.3261,
        0.8997,
        -0.2137,
        -0.3184,
        -0.2336,
        0.5519,
    ]
)
state_dict["normalize_targets.buffer_action.mean"] = state_dict["unnormalize_outputs.buffer_action.mean"]
state_dict["unnormalize_outputs.buffer_action.std"] = torch.tensor(
    [
        0.0125,
        0.2957,
        0.1670,
        0.0458,
        0.1483,
        0.0876,
        0.3067,
        0.1060,
        0.2757,
        0.1806,
        0.2630,
        0.3071,
        0.5305,
        0.3838,
    ]
)
state_dict["normalize_targets.buffer_action.std"] = state_dict["unnormalize_outputs.buffer_action.std"]

missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)

if len(missing_keys) != 0:
    print("MISSING KEYS")
    print(missing_keys)
if len(unexpected_keys) != 0:
    print("UNEXPECTED KEYS")
    print(unexpected_keys)

if len(missing_keys) != 0 or len(unexpected_keys) != 0:
    print("Failed due to mismatch in state dicts.")
    exit()

policy.save_pretrained(PATH_TO_SAVE_NEW_WEIGHTS)
OmegaConf.save(cfg, Path(PATH_TO_SAVE_NEW_WEIGHTS) / "config.yaml")
