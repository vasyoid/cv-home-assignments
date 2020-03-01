#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    pose_to_view_mat3x4
)

triang_params = TriangulationParameters(max_reprojection_error=1,
                                        min_triangulation_angle_deg=1.5,
                                        min_depth=0.1)

# 1, 1.5, 0.1 for fox_head_short
# 10, 0.01, 0.1 for bike_translation_slow

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    mat_1 = pose_to_view_mat3x4(known_view_1[1])
    mat_2 = pose_to_view_mat3x4(known_view_2[1])

    points, ids = build_and_triangulate_correspondences(corner_storage, intrinsic_mat,
                                                        known_view_1[0], mat_1,
                                                        known_view_2[0], mat_2)
    if len(points) < 10:
        exit(0)

    point_cloud_builder = PointCloudBuilder(ids, points)

    total_frames = len(corner_storage)
    tracked_mats = np.full(total_frames, None)
    tracked_mats[known_view_1[0]] = mat_1
    tracked_mats[known_view_2[0]] = mat_2

    updated = True
    while updated:
        updated = False
        for cur_frame, corners in enumerate(corner_storage):
            if tracked_mats[cur_frame] is not None:
                continue

            _, comm1, comm2 = np.intersect1d(point_cloud_builder.ids.flatten(),
                                             corners.ids.flatten(),
                                             return_indices=True)
            try:
                _, rvec, tvec, inliers = cv2.solvePnPRansac(point_cloud_builder.points[comm1],
                                                            corners.points[comm2],
                                                            intrinsic_mat,
                                                            None)
                inliers = inliers.astype(int)
                inliers_len = len(inliers)
                print(f'Tracking corners. [{cur_frame} frames of {total_frames}] {inliers_len} inliers')
                if inliers_len > 0:
                    tracked_mats[cur_frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                    updated = True
                else:
                    continue
            except:
                print(f'Tracking corners. [{cur_frame} frames of {total_frames}] no inliers')
                continue

            for another_frame in range(total_frames):
                if another_frame == cur_frame or tracked_mats[another_frame] is None:
                    continue
                points, ids = build_and_triangulate_correspondences(corner_storage, intrinsic_mat,
                                                                    another_frame, tracked_mats[another_frame],
                                                                    cur_frame, tracked_mats[cur_frame])
                if len(ids) != 0:
                    point_cloud_builder.add_points(ids, points)

    cur_mat = None
    for mat in tracked_mats:
        if mat is not None:
            cur_mat = mat
            break

    if cur_mat is None:
        exit(0)

    tracked_mats[0] = cur_mat
    for i in range(1, len(tracked_mats)):
        if tracked_mats[i] is None:
            tracked_mats[i] = cur_mat
        else:
            cur_mat = tracked_mats[i]

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        tracked_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, tracked_mats))
    return poses, point_cloud


def build_and_triangulate_correspondences(corner_storage, intrinsic_mat, idx_1, mat_1, idx_2, mat_2):
    correspondences = build_correspondences(corner_storage[idx_1], corner_storage[idx_2])
    if len(correspondences.ids) == 0:
        return [], []
    points, ids, _ = triangulate_correspondences(correspondences,
                                                 mat_1, mat_2,
                                                 intrinsic_mat,
                                                 triang_params)
    return points, ids


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
