import numpy as np
from jwst.pipeline import Detector1Pipeline


# int_ims = np.ones((10, 400, 72)) * np.linspace(0, 10, 400)[np.newaxis, :, np.newaxis]
# ref_pixels = int_ims[:, :, 0:4]
# # print(ref_pixels)
#
# smoothing_length = 11
# sm_ref_pixels = np.copy(ref_pixels)
# n_rows = ref_pixels.shape[1]
# sm_radius = int((smoothing_length - 1) / 2)
# for idx_row in range(n_rows):
#     # Define window.
#     start_row = max(0, idx_row - sm_radius)
#     end_row = min(n_rows - 1, idx_row + sm_radius)
#
#     # Compute median in window.
#     sm_ref_pixels[:, idx_row, :] = np.median(
#         ref_pixels[:, start_row:end_row + 1, :], axis=1)
#
# aa = np.tile(np.median(ref_pixels, axis=1)[:, np.newaxis, :], (1, 400, int(72 / 4)))
# print(aa.shape)
#
# bb = np.tile(sm_ref_pixels, (1, 1, int(72 / 4)))
# print(bb.shape)
#
# odd_even_medians = np.copy(ref_pixels)
# odd_even_medians[:, 0::2, :] = np.median(odd_even_medians[:, 0::2, :], axis=1)[:, np.newaxis, :]
# odd_even_medians[:, 1::2, :] = np.median(odd_even_medians[:, 1::2, :], axis=1)[:, np.newaxis, :]
# cc = np.tile(odd_even_medians, (1, 1, int(72 / 4)))
# print(cc.shape)
