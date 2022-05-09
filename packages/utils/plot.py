import numpy as np
import matplotlib.pyplot as plt

def plot_grid(images, n_cols=3, figsize=(20,20), caption="Image", save_path=None):

    n_lines = int(np.ceil(len(images) / n_cols))
    fig, ax = plt.subplots(n_lines, n_cols, figsize=figsize)
    num_images = len(images)

    # plot word cloud
    for i in range(num_images):
        ax.flatten()[i].imshow(images[i], interpolation='bilinear')

    # clean up
    for i in range(n_cols*n_lines):
        ax.flatten()[i].axis('off')
        if i < num_images:
            ax.flatten()[i].set_title(f'{caption} #{i+1}')

    if save_path is None:
        save_path = f"{caption}s_grid.png"
    # save figure
    plt.savefig(save_path)