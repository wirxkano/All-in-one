import matplotlib.pyplot as plt
from blindsr import *


def visualize_kernel(kernel):
    """
    Visualizes a 2D kernel using matplotlib.

    Parameters:
    kernel (numpy.ndarray): 2D array representing the kernel to visualize.
    """

    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array.")

    plt.imshow(kernel)
    plt.title("Anisotropic Gaussian Kernel")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    kernel_list = ["aniso"]
    kernel_prob = [1]
    kernel_size = 21
    sigma_x = [(0.1, 1), (1.5, 3.5), (0.75, 4)]
    sigma_y = [(0.1, 1), (1.5, 3.5), (0.3, 8)]
    theta = (np.deg2rad(89), np.deg2rad(91))
    betag = (0.5, 4)
    betap = (0.5, 4)

    blur_kernel = random_mixed_kernels(
        kernel_list,
        kernel_prob,
        kernel_size,
        sigma_x[2],
        sigma_y[2],
        theta,
        betag,
        betap,
    )

    visualize_kernel(blur_kernel)
