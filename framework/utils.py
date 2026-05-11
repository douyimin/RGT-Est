import matplotlib.pyplot as plt


def show_results(tensor_dict: dict, save_path: str) -> None:
    """Plot three orthogonal mid-slices for each tensor in ``tensor_dict``.

    Each tensor is expected to be shaped ``(N, C, T, H, W)``; the first
    sample of the first channel is used. The ``seis`` key is rendered
    in grayscale; everything else uses the ``jet`` colormap.
    """
    dict_len = len(tensor_dict)
    plt.figure(figsize=(30, 30))

    for idx, key in enumerate(tensor_dict):
        data = tensor_dict[key][0, 0]
        t, h, w = data.shape
        cmap = "gray" if key == "seis" else "jet"

        plt.subplot(3, dict_len, idx + 1)
        plt.title(key)
        plt.imshow(data[:, int(h / 2)], cmap=cmap)

        plt.subplot(3, dict_len, idx + dict_len + 1)
        plt.title(key)
        plt.imshow(data[:, :, int(w / 2)], cmap=cmap)

        plt.subplot(3, dict_len, idx + dict_len * 2 + 1)
        plt.title(key)
        plt.imshow(data[int(t / 2), :, :], cmap=cmap)

    plt.savefig(save_path)
    plt.close()
