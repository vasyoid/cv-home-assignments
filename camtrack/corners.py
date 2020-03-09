#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    pixels_count = np.size(frame_sequence[0])
    block_size = 15
    max_corners = pixels_count // 800
    min_distance = block_size * 2
    feature_params = dict(maxCorners=max_corners,
                          qualityLevel=0.05,
                          minDistance=min_distance,
                          useHarrisDetector=False,
                          blockSize=block_size)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_image = frame_sequence[0]
    prev_image *= 255
    prev_image = prev_image.astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(image=prev_image, **feature_params)
    _set_frame_corners(block_size, 0, np.arange(len(corners)), corners, builder)

    indices = np.arange(corners.shape[0])
    max_index = indices.size

    for frame_ind, next_image in enumerate(frame_sequence[1:]):
        next_image *= 255
        next_image = next_image.astype(np.uint8)
        next_corners = cv2.calcOpticalFlowPyrLK(prev_image, next_image, corners, None, **lk_params)[0].squeeze()
        back_flow = cv2.calcOpticalFlowPyrLK(next_image, prev_image, next_corners, None, **lk_params)[0].squeeze()
        good = np.abs(corners.squeeze() - back_flow).max(-1) < 0.2
        indices = indices[good]
        corners = next_corners[good]

        if corners.shape[0] < max_corners:
            feature_params['maxCorners'] = max_corners - corners.shape[0]
            mask = np.full_like(next_image, 255)
            for cx, cy in corners.reshape(-1, 2):
                cv2.circle(mask, (cx, cy), block_size, 0, -1)
            new_corners = cv2.goodFeaturesToTrack(next_image, mask=mask, **feature_params)
            if new_corners is not None:
                indices = np.append(indices, np.arange(max_index, max_index + new_corners.shape[0]))
                max_index = indices[-1] + 1
                corners = np.append(corners, new_corners).reshape((-1, 1, 2))
        prev_image = next_image
        _set_frame_corners(block_size, frame_ind + 1, indices, corners, builder)


def _set_frame_corners(block_size: int, frame_ind: int, indices: np.ndarray, cv_corners: np.ndarray,
                       builder: _CornerStorageBuilder):
    corners = FrameCorners(
        indices,
        cv_corners,
        np.full(len(cv_corners), block_size)
    )
    builder.set_corners_at_frame(frame_ind, corners)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
