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


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None,
                          reproj_err: float = 1,
                          min_angle: float = 1,
                          min_depth: float = 0.1) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    triang_params = TriangulationParameters(max_reprojection_error=reproj_err,
                                            min_triangulation_angle_deg=min_angle,
                                            min_depth=min_depth)

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    mat_1 = pose_to_view_mat3x4(known_view_1[1])
    mat_2 = pose_to_view_mat3x4(known_view_2[1])

    points, ids = build_and_triangulate_correspondences(intrinsic_mat,
                                                        corner_storage[known_view_1[0]], mat_1,
                                                        corner_storage[known_view_2[0]], mat_2,
                                                        triang_params)
    if len(points) < 10:
        exit(0)

    total_frames = len(corner_storage)
    tracked_mats = np.full(total_frames, None)
    tracked_mats[known_view_1[0]] = mat_1
    tracked_mats[known_view_2[0]] = mat_2

    cloud_points = [None] * (corner_storage.max_corner_id() + 1)
    for pt, i in zip(points, ids):
        cloud_points[i] = pt

    updated = True
    while updated:
        updated = False
        for cur_frame, corners in enumerate(corner_storage):
            if tracked_mats[cur_frame] is not None:
                continue
            updated = updated or try_update(corner_storage, corners, cur_frame, intrinsic_mat, cloud_points, total_frames,
                                            tracked_mats, triang_params)

    point_cloud_builder = PointCloudBuilder(
        ids=np.array([i for i, point in enumerate(cloud_points) if point is not None]),
        points=np.array([point for point in cloud_points if point is not None])
    )

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
    cloud_points = list(map(view_mat3x4_to_pose, tracked_mats))
    return cloud_points, point_cloud


def try_update(corner_storage, corners, cur_frame, intrinsic_mat, cloud_points, total_frames, tracked_mats, triang_params):
    ids = corners.ids.flatten()
    mask = np.array([cloud_points[id] is not None for id in ids])
    try:
        if corners.points[mask].shape[0] < 10:
            raise
        _, rvec, tvec, inliers = cv2.solvePnPRansac(np.array([cloud_points[ind] for ind in ids[mask]]),
                                                    corners.points[mask], intrinsic_mat, None)
        inliers = inliers.flatten()
        print(f'Tracking corners. [{cur_frame} frames of {total_frames}] {len(inliers)} inliers')
        for id in ids:
            cloud_points[id] = None if id not in inliers else cloud_points[id]
        tracked_mats[cur_frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
    except:
        print(f'Tracking corners. [{cur_frame} frames of {total_frames}] no inliers')
        return False
    if tracked_mats[cur_frame] is not None:
        for another_frame in range(total_frames):
            if cur_frame == another_frame or tracked_mats[another_frame] is None:
                continue
            points, ids, = build_and_triangulate_correspondences(intrinsic_mat,
                                                                 corners,
                                                                 tracked_mats[cur_frame],
                                                                 corner_storage[another_frame],
                                                                 tracked_mats[another_frame],
                                                                 triang_params)
            for pt, id in zip(points, ids):
                if cloud_points[id] is None:
                    cloud_points[id] = pt
    return tracked_mats[cur_frame] is not None


def build_and_triangulate_correspondences(intrinsic_mat, corners_1, mat_1, corners_2, mat_2, triang_params):
    correspondences = build_correspondences(corners_1, corners_2)
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
