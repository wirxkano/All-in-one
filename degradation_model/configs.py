import math

BLUR_SETTINGS = dict(
    kernel_list=[
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],
    kernel_prob=[0.2, 0.3, 0.1, 0.3, 0.1, 0.1],
    kernel_size=range(7, 13, 2),
    sigma_x=[(0.1, 2), (1.5, 4.5), (3.5, 6)],
    sigma_y=[(0.05, 1), (2.5, 3.5), (2.5, 4)],
    theta=(0.0, math.pi),
    betag=(0.5, 4),
    betap=(0.5, 4),
)

SINC_SETTINGS = dict(kernel_size=range(7, 21, 2))

NOISE_SETTINGS = dict(
    noise_list=["gaussian", "poisson"],
    noise_prob=[0.6, 0.4],
    sigma=[(13, 17), (8, 13), (3, 5)],
    scale=[(2.5, 3), (2, 2.5), (0.25, 1.5)],
)

JPEG_COMPRESSION_SETTINGS = dict(quality_range=(85, 90))


SETTINGS = dict(
    blur_settings=BLUR_SETTINGS,
    sinc_settings=SINC_SETTINGS,
    noise_settings=NOISE_SETTINGS,
    jpeg_compression_settings=JPEG_COMPRESSION_SETTINGS,
)
