from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG # must import first

import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt

from training.utils import *


# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1')
args = parser.parse_args()
task = args.task

# configs
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)

# device
device = os.environ.get('DEVICE', 'cuda')


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def train_bc(train_dataloader, val_dataloader, policy_config):
    ################## TODO remove this part
    # torch.backends.cudnn.deterministic=True
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cuda.matmul.allow_tf32 = True
    ##################


    # load policy
    policy = make_policy(policy_config['policy_class'], policy_config)
    print(f"training on {device}")
    policy.to(device)

    # load optimizer
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    # create checkpoint dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in range(train_cfg['num_epochs']):
        # print(f'\nEpoch {epoch}')
        # # validation
        # with torch.inference_mode():
        #     policy.eval()
        #     epoch_dicts = []
        #     for batch_idx, data in enumerate(val_dataloader):
        #         forward_dict = forward_pass(data, policy)
        #         epoch_dicts.append(forward_dict)
        #     epoch_summary = compute_dict_mean(epoch_dicts)
        #     validation_history.append(epoch_summary)

        #     epoch_val_loss = epoch_summary['loss']
        #     if epoch_val_loss < min_val_loss:
        #         min_val_loss = epoch_val_loss
        #         best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        # print(f'Val loss:   {epoch_val_loss:.5f}')
        # summary_string = ''
        # for k, v in epoch_summary.items():
        #     summary_string += f'{k}: {v.item():.3f} '
        # print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            # # normalize data
            # image_data, qpos_data, action_data, is_pad = data
            # action_data = action_data[:, :policy_config['num_queries'], :]
            # is_pad = is_pad[:, :policy_config['num_queries']]

            # ##########################
            # export_convert_model_and_batch(policy, (image_data, qpos_data, action_data, is_pad))

            # policy.eval()  # No dropout for debug

            # # print some stats
            # def model_stats(model):
            #     na = list(n for n, a in model.named_parameters() if not 'normalize_' in n)
            #     me = list(a.mean().item() for n, a in model.named_parameters() if not 'normalize_' in n)
            #     print(na[me.index(min(me))], min(me))
            #     print(sum(me))
            #     mi = list(a.min().item() for n, a in model.named_parameters() if not 'normalize_' in n)
            #     print(na[mi.index(min(mi))], min(mi))
            #     print(sum(mi))
            #     ma = list(a.max().item() for n, a in model.named_parameters() if not 'normalize_' in n)
            #     print(na[ma.index(max(ma))], max(ma))
            #     print(sum(ma))

            # model_stats(policy)

            # def batch_stats(data):
            #     print(min(d.min() for d in data))
            #     print(max(d.max() for d in data))
            # batch_stats(data)

            # ##############################

            # action_data = (action_data - train_dataloader.dataset.norm_stats["action_mean"]) / train_dataloader.dataset.norm_stats["action_std"]
            # qpos_data = (qpos_data - train_dataloader.dataset.norm_stats["qpos_mean"]) / train_dataloader.dataset.norm_stats["qpos_std"]
            # data = (image_data, qpos_data, action_data, is_pad)

            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 200 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])

    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    

def export_convert_model_and_batch(policy, data):
    import torch
    from safetensors.torch import load_file, save_file, _remove_duplicate_names
    from pprint import pprint
    import pickle
    import numpy as np

    # batch
    with open('/home/thomwolf/Documents/Github/ACT/batch_save.pt', 'wb') as f:
        torch.save(data, f)
    original_batch_file = "/home/thomwolf/Documents/Github/ACT/batch_save.pt"
    data = torch.load(original_batch_file)
    conv = {}
    conv['observation.images.front'] = data[0][:, 0]
    conv['observation.images.top'] = data[0][:, 1]
    conv['observation.state'] = data[1]
    conv['action'] = data[2]
    conv['episode_index'] = np.zeros(data[0].shape[0])
    conv['frame_index'] = np.zeros(data[0].shape[0])
    conv['timestamp'] = np.zeros(data[0].shape[0])
    conv['next.done'] = np.zeros(data[0].shape[0])
    conv['index'] = np.arange(data[0].shape[0])
    conv['action_is_pad'] = data[3]
    torch.save(conv, "/home/thomwolf/Documents/Github/ACT/batch_save_converted.pt")

    # model
    original_ckpt_path = "/home/thomwolf/Documents/Github/ACT/checkpoints/blue_red_sort_raw/policy_initial_state.ckpt"
    stats_path = original_ckpt_path.replace('policy_initial_state.ckpt', f'dataset_stats.pkl')
    converted_ckpt_path = "/home/thomwolf/Documents/Github/ACT/checkpoints/blue_red_sort_raw/initial_state/model.safetensors"

    comparison_main_path = "/home/thomwolf/Documents/Github/lerobot/examples/real_robot_example/outputs/train/blue_red_debug_no_masking/checkpoints/last/pretrained_model/"
    comparison_safetensor_path = comparison_main_path + "model.safetensors"
    comparison_config_json_path = comparison_main_path + "config.json"
    comparison_config_yaml_path = comparison_main_path + "config.yaml"
    torch.save(policy.state_dict(), original_ckpt_path)

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    image_stats_mean = torch.tensor([0.485, 0.456, 0.406])
    image_stats_std = torch.tensor([0.229, 0.224, 0.225])

    a = torch.load(original_ckpt_path)
    b = load_file(comparison_safetensor_path)

    to_remove_startswith = ['model.transformer.decoder.layers.1.',
                'model.transformer.decoder.layers.2.',
                'model.transformer.decoder.layers.3.',
                'model.transformer.decoder.layers.4.',
                'model.transformer.decoder.layers.5.',
                'model.transformer.decoder.layers.6.',
                'model.is_pad_head']

    # to_remove_in = ['num_batches_tracked',]
    to_remove_in = ['num_batches_tracked',]

    conv = {}

    keys = list(a.keys())
    for k in keys:
        if any(k.startswith(tr) for tr in to_remove_startswith):
            a.pop(k)
            continue
        if any(tr in k for tr in to_remove_in):
            a.pop(k)
            continue
        if k.startswith('model.transformer.encoder.layers.'):
            conv[k.replace('transformer.', '')] = a.pop(k)
        if k.startswith('model.transformer.decoder.layers.0.'):
            conv[k.replace('transformer.', '')] = a.pop(k)
        if k.startswith('model.transformer.decoder.norm.'):
            conv[k.replace('transformer.', '')] = a.pop(k)
        if k.startswith('model.encoder.layers.'):
            conv[k.replace('encoder.', 'vae_encoder.')] = a.pop(k)
        if k.startswith('model.action_head.'):
            conv[k] = a.pop(k)
        if k.startswith('model.pos_table'):
            conv[k.replace('pos_table', 'vae_encoder_pos_enc')] = a.pop(k)
        if k.startswith('model.query_embed.'):
            conv[k.replace('query_embed', 'decoder_pos_embed')] = a.pop(k)
        if k.startswith('model.input_proj.'):
            conv[k.replace('input_proj.', 'encoder_img_feat_input_proj.')] = a.pop(k)
        if k.startswith('model.input_proj_robot_state.'):
            conv[k.replace('input_proj_robot_state.', 'encoder_robot_state_input_proj.')] = a.pop(k)
        if k.startswith('model.backbones.0.0.body.'):
            conv[k.replace('backbones.0.0.body', 'backbone')] = a.pop(k)
        if k.startswith('model.cls_embed.'):
            conv[k.replace('cls_embed', 'vae_encoder_cls_embed')] = a.pop(k)
        if k.startswith('model.encoder_action_proj.'):
            conv[k.replace('encoder_action_proj', 'vae_encoder_action_input_proj')] = a.pop(k)
        if k.startswith('model.encoder_joint_proj.'):
            conv[k.replace('encoder_joint_proj', 'vae_encoder_robot_state_input_proj')] = a.pop(k)
        if k.startswith('model.latent_proj.'):
            conv[k.replace('latent_proj', 'vae_encoder_latent_output_proj')] = a.pop(k)
        if k.startswith('model.latent_out_proj.'):
            conv[k.replace('latent_out_proj', 'encoder_latent_input_proj')] = a.pop(k)
        if k.startswith('model.additional_pos_embed.'):
            conv[k.replace('additional_pos_embed', 'encoder_robot_and_latent_pos_embed')] = a.pop(k)

    conv['normalize_inputs.buffer_observation_images_front.mean'] = image_stats_mean[:, None, None]
    conv['normalize_inputs.buffer_observation_images_front.std'] = image_stats_std[:, None, None]
    conv['normalize_inputs.buffer_observation_images_top.mean'] = image_stats_mean[:, None, None].clone()
    conv['normalize_inputs.buffer_observation_images_top.std'] = image_stats_std[:, None, None].clone()

    conv['normalize_inputs.buffer_observation_state.mean'] = torch.tensor(stats['qpos_mean'])
    conv['normalize_inputs.buffer_observation_state.std'] = torch.tensor(stats['qpos_std'])
    conv['normalize_targets.buffer_action.mean'] = torch.tensor(stats['action_mean'])
    conv['normalize_targets.buffer_action.std'] = torch.tensor(stats['action_std'])

    conv['unnormalize_outputs.buffer_action.mean'] = torch.tensor(stats['action_mean'])
    conv['unnormalize_outputs.buffer_action.std'] = torch.tensor(stats['action_std'])

    not_converted = set(b.keys())
    for k, v in conv.items():
        try:
            b[k].shape == v.squeeze().shape
        except Exception as e:
            print(k, v)
            print(b[k].shape)
            print(e)
        b[k] = v
        not_converted.remove(k)

    metadata = None
    to_removes = _remove_duplicate_names(b)
    print(to_removes)

    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if metadata is None:
                metadata = {}

            if to_remove not in metadata:
                # Do not override user data
                metadata[to_remove] = kept_name
            del b[to_remove]
    save_file(b, converted_ckpt_path)

    # Now also copy the config files
    import shutil
    shutil.copy(comparison_config_json_path, converted_ckpt_path.replace('model.safetensors', 'config.json'))
    shutil.copy(comparison_config_yaml_path, converted_ckpt_path.replace('model.safetensors', 'config.yaml'))


if __name__ == '__main__':
    # set seed
    set_seed(train_cfg['seed'])
    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)
   # number of training episodes
    data_dir = os.path.join(task_cfg['dataset_dir'], task)
    num_episodes = len(os.listdir(data_dir))

    # load data
    train_dataloader, val_dataloader, stats, _ = load_data(data_dir, num_episodes, task_cfg['camera_names'],
                                                            train_cfg['batch_size_train'], train_cfg['batch_size_val'])
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # train
    train_bc(train_dataloader, val_dataloader, policy_config)