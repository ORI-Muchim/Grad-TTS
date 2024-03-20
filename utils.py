import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    if f_list:
        x = f_list[-1]
        return x
    return None


def load_checkpoint(logdir, model, optimizer=None):
    checkpoint_path = latest_checkpoint_path(logdir)
    if checkpoint_path:
        print(f"Loading checkpoint '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        if optimizer and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch = checkpoint.get('epoch', 1)
        iteration = checkpoint.get('iteration', 0)
        print(f"Checkpoint loaded: Epoch {epoch}, Iteration {iteration}")
    else:
        epoch, iteration = 0, 0
        print("No checkpoint found, starting from scratch.")
    return model, optimizer, epoch, iteration


def save_checkpoint(model, optimizer, epoch, iteration, log_dir):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iteration
    }

    print(f"Saving checkpoint: Epoch {epoch}, Iteration {iteration}")
    torch.save(checkpoint, os.path.join(log_dir, f"grad_{epoch}.pt"))


def remove_optimizer_latest_checkpoint(dir_path, output_path, regex="G_*.pth"):
    model_path = latest_checkpoint_path(dir_path, regex)
    if model_path is None:
        print("No latest checkpoint found.")
        return

    checkpoint_dict = torch.load(model_path, map_location='cpu')

    if "optimizer" in checkpoint_dict:
        checkpoint_dict_new = {k: v for k, v in checkpoint_dict.items() if k != "optimizer"}

        base_name = os.path.basename(model_path)
        new_base_name = os.path.splitext(base_name)[0] + "_rem_opti.pth"
        new_output_path = os.path.join(output_path, new_base_name)
        print("Optimizer state removed from the latest checkpoint.")

        torch.save(checkpoint_dict_new, new_output_path)
        print(f"Modified checkpoint saved to {new_output_path}")
    else:
        print("No optimizer state found in the latest checkpoint.")


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Grad-TTS')
    plt.tight_layout()
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return

