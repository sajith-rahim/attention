import matplotlib.pyplot as plt


def plot(attn):
    r"""
    Utility to print attn
    :param attn:
    :return: None
    """

    total = attn.shape[1]
    Cols = 2

    # Compute rows required

    rows = total // Cols
    rows += total % Cols

    # Create a Position index

    position = range(1, total + 1)

    fig = plt.figure(1)
    for k in range(total):
        # add every single subplot to the figure with a for loop

        ax = fig.add_subplot(rows, Cols, position[k])
        ax.matshow(attn[0][k].detach().numpy())

    plt.show()
