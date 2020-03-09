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
    pose_to_view_mat3x4,
    eye3x4,
    remove_correspondences_with_ids
)


def init_camera_positions(corner_storage, intrinsic_mat, triang_params):
    best_pose = None
    best_cloud_size = -1
    best_ind = -1
    total_frames = len(corner_storage)
    for i, frame in enumerate(corner_storage):
        print(f'\rInitializing camera positions. [{i + 2} frames of {total_frames}], ', end='')
        pose, cloud_size = pose_by_two_frames(corner_storage[0], frame, intrinsic_mat, triang_params)
        if cloud_size > best_cloud_size:
            best_pose = pose
            best_cloud_size = cloud_size
            best_ind = i
        print(f'{cloud_size} points   ', end='')
    print(f'\nframes 0 and {best_ind} have been chosen, {best_cloud_size} correspondences found')
    return (0, eye3x4()), (best_ind, pose_to_view_mat3x4(best_pose))


def pose_by_two_frames(frame1, frame2, intrinsic_mat, triang_params):
    correspondences = build_correspondences(frame1, frame2)
    if correspondences.ids.shape[0] < 7:
        return None, 0
    mat, mat_mask = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2, intrinsic_mat,
                                         method=cv2.RANSAC, prob=0.99, threshold=1)
    mat_mask = mat_mask.flatten()
    correspondences = remove_correspondences_with_ids(correspondences, np.argwhere(mat_mask == 0))
    r1, r2, t = cv2.decomposeEssentialMat(mat)
    best_pose = None
    best_points = -1
    for mat in (r1.T, r2.T):
        for vec in (t, -t):
            pose = Pose(mat, mat @ vec)
            points, _, _ = triangulate_correspondences(correspondences, eye3x4(), pose_to_view_mat3x4(pose),
                                                       intrinsic_mat, triang_params)
            if points.shape[0] > best_points:
                best_pose = pose
                best_points = points.shape[0]
    return best_pose, best_points


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None,
                          reproj_err: float = 1,
                          min_angle: float = 1,
                          min_depth: float = 0.1) \
        -> Tuple[List[Pose], PointCloud]:
    triang_params = TriangulationParameters(max_reprojection_error=reproj_err,
                                            min_triangulation_angle_deg=min_angle,
                                            min_depth=min_depth)

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = init_camera_positions(corner_storage, intrinsic_mat, triang_params)
        mat_1 = known_view_1[1]
        mat_2 = known_view_2[1]
    else:
        mat_1 = pose_to_view_mat3x4(known_view_1[1])
        mat_2 = pose_to_view_mat3x4(known_view_2[1])

    points, ids = build_and_triangulate_correspondences(intrinsic_mat,
                                                        corner_storage[known_view_1[0]], mat_1,
                                                        corner_storage[known_view_2[0]], mat_2,
                                                        triang_params)
    if len(points) < 7:
        print(f'too few correspondences: {points.shape[0]}')
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
            updated = updated or try_update(corner_storage, corners, cur_frame, intrinsic_mat, cloud_points,
                                            total_frames,
                                            tracked_mats, triang_params)

    print()

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


def try_update(corner_storage, corners, cur_frame, intrinsic_mat, cloud_points, total_frames, tracked_mats,
               triang_params):
    ids = corners.ids.flatten()
    mask = np.array([cloud_points[id] is not None for id in ids])
    try:
        if corners.points[mask].shape[0] < 10:
            raise
        _, rvec, tvec, inliers = cv2.solvePnPRansac(np.array([cloud_points[ind] for ind in ids[mask]]),
                                                    corners.points[mask], intrinsic_mat, None)
        inliers = inliers.flatten()
        print(f'\rTracking corners. [{cur_frame + 1} frames of {total_frames}] {len(inliers)} inliers', end='')
        for id in ids:
            cloud_points[id] = None if id not in inliers else cloud_points[id]
        tracked_mats[cur_frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
    except:
        print(f'\rTracking corners. [{cur_frame + 1} frames of {total_frames}] no inliers', end='')
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
