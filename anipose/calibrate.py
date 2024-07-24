#!/usr/bin/env python3

from tqdm import tqdm
import numpy as np
import cv2
import os
from glob import glob
from collections import defaultdict
import pickle
import re

from .common import \
    find_calibration_folder, make_process_fun, process_all, \
    get_cam_name, get_video_name, \
    get_calibration_board, split_full_path, get_video_params

from .triangulate import load_pose2d_fnames, load_offsets_dict

from aniposelib.cameras import CameraGroup

def get_pose2d_fnames(config, session_path):
    if config['filter']['enabled']:
        pipeline_pose = config['pipeline']['pose_2d_filter']
    else:
        pipeline_pose = config['pipeline']['pose_2d']
    fnames = glob(os.path.join(session_path, pipeline_pose, '*.h5'))
    return session_path, fnames


def load_2d_data(config, calibration_path):
    # start_path, _ = os.path.split(calibration_path)
    start_path = calibration_path

    nesting_path = len(split_full_path(config['path']))
    nesting_start = len(split_full_path(start_path))
    new_nesting = config['nesting'] - (nesting_start - nesting_path)

    new_config = dict(config)
    new_config['path'] = start_path
    new_config['nesting'] = new_nesting

    pose_fnames = process_all(new_config, get_pose2d_fnames)
    cam_videos = defaultdict(list)
    all_cam_names = set()

    for key, (session_path, fnames) in pose_fnames.items():
        for fname in fnames:
            # print(fname)
            vidname = get_video_name(config, fname)
            cname = get_cam_name(config, fname)
            k = (key, session_path, vidname)
            cam_videos[k].append(fname)
            all_cam_names.add(cname)

    all_cam_names = sorted(list(all_cam_names))
    vid_names = sorted(cam_videos.keys())

    all_points = []
    all_scores = []

    for name in tqdm(vid_names, desc='load points', ncols=80):
        (key, session_path, vidname) = name
        fnames = sorted(cam_videos[name])
        cam_names = [get_cam_name(config, f) for f in fnames]
        fname_dict = dict(zip(cam_names, fnames))
        video_folder = os.path.join(session_path, config['pipeline']['videos_raw'])
        offsets_dict = load_offsets_dict(config, cam_names, video_folder)
        out = load_pose2d_fnames(fname_dict, offsets_dict, cam_names)
        points_raw_dict = dict(zip(cam_names, out['points']))
        scores_dict = dict(zip(cam_names, out['scores']))

        _, n_frames, n_joints, _ = out['points'].shape
        points_raw = np.full((len(all_cam_names), n_frames, n_joints, 2), np.nan, 'float')
        scores = np.full((len(all_cam_names), n_frames, n_joints), np.nan, 'float')

        for cnum, cname in enumerate(all_cam_names):
            if cname in points_raw_dict:
                points_raw[cnum] = points_raw_dict[cname]
                scores[cnum] = scores_dict[cname]


        all_points.append(points_raw)
        all_scores.append(scores)

    all_points = np.hstack(all_points)
    all_scores = np.hstack(all_scores)

    return all_points, all_scores, all_cam_names

def process_points_for_calibration(all_points, all_scores):
    """Takes in an array all_points of shape CxFxJx2 and all_scores of shape CxFxJ, where
    C: number of cameras
    F: number of frames
    J: number of joints"""

    assert all_points.shape[3] == 2, \
        "points are not 2 dimensional, or shape of points is wrong: {}".format(all_points.shape)

    n_cams, n_frames, n_joints, _ = all_points.shape

    points = np.copy(all_points).reshape(n_cams, -1, 2)
    scores = all_scores.reshape(n_cams, -1)

    bad = np.isnan(points[:, :, 0])
    scores[bad] = 0

    thres = np.percentile(scores, 90)
    thres = max(min(thres, 0.95), 0.8)
    points[scores < thres] = np.nan

    num_good = np.sum(~np.isnan(points[:, :, 0]), axis=0)
    good = num_good >= 2
    points = points[:, good]

    max_size = int(100e3)

    if points.shape[1] > max_size:
        sample_ixs = np.random.choice(points.shape[1], size=max_size, replace=False)
        points = points[:, sample_ixs]

    return points

# 自分で作った関数。corner detectionの動画を出力する
def EvaluateCornerDetection(board, video_list, data_fold_path, **kwargs):
    print('---------------start corner detection---------------')
    axis_flag=False
    # corner detectionの結果(all_rows)とカメラの内部パラメータをロードする
    all_rows_data_path = os.path.join(data_fold_path, 'optimized_all_rows.pickle')
    matrix_data_path = os.path.join(data_fold_path, 'optimized_intrinsic_matrix.pickle')
    distortion_data_path = os.path.join(data_fold_path, 'optimized_distortion_vector.pickle')
    if os.path.exists(all_rows_data_path):
        axis_flag=True
        with open(distortion_data_path, 'rb') as f:
            distortion_list = pickle.load(f)
        with open(all_rows_data_path, 'rb') as f:
            all_rows = pickle.load(f)
        with open(matrix_data_path, 'rb') as f:
            intrinsic_matrix_list = pickle.load(f)
    else:
        all_rows_data_path = os.path.join(data_fold_path, 'detections.pickle')
        with open(all_rows_data_path, 'rb') as f:
            all_rows = pickle.load(f)

    output_fold_path = data_fold_path.replace('calibration', 'cornerDetection')
    if not os.path.exists(output_fold_path):
        os.makedirs(output_fold_path)

    # これ以下の処理はカメラごとにループさせる
    for video_idx in range(len(video_list)):
        videoPath = video_list[video_idx][0]
        # 出力する動画の名前を設定
        if axis_flag:
            output_video_name = videoPath.split('\\')[-1].split('.')[0] + '(corner_detection_with_axis).mp4'
        else:
            output_video_name = videoPath.split('\\')[-1].split('.')[0] + '(corner_detection_no_axis).mp4'

        outputPath = os.path.join(output_fold_path, output_video_name)
        if os.path.exists(outputPath):
            continue
        
        camera_name = re.search(r'-cam([A-Za-z])', output_video_name).group(1)

        # 該当するrowsとintrinsic_matrixを取得
        rows = all_rows[video_idx]
        detected_frame_idx_list = [row['framenum'][1] for row in rows]
        if axis_flag:
            intrinsic_matrix = intrinsic_matrix_list[camera_name]

        try:
            cap = cv2.VideoCapture(videoPath)
            # 動画のプロパティ取得
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 動画の出力設定'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画のフォーマット
            out = cv2.VideoWriter(outputPath, fourcc, fps, (frame_width, frame_height))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f'output_movie_path: {outputPath}')

            # 指定した動画から各フレームの画像を抽出
            for frame_idx in tqdm(range(frame_count), desc=f'camera-{camera_name} Processing', ncols=80):
                ret, frame = cap.read()
                if not ret:
                    break

                # マーカーが検出された場合
                if frame_idx in detected_frame_idx_list:
                    row_idx = detected_frame_idx_list.index(frame_idx)
                    row = rows[row_idx]
                    markerCorners = row['corners']
                    markerIds = row['ids']
                    # 検出されたコーナーを描画
                    cv2.aruco.drawDetectedCornersCharuco(frame, markerCorners, markerIds)
                    
                    # 軸の描画
                    if axis_flag:
                        rvec = row['rvec']
                        tvec = row['tvec']
                        if rvec is not None and tvec is not None:
                            distCoeffs = distortion_list[camera_name]
                            # ChArUcoボードの軸を描画
                            cv2.drawFrameAxes(frame, intrinsic_matrix, distCoeffs, rvec, tvec, board.square_length*1.5, 2)

                # フレームを動画に書き込み
                out.write(frame)
        finally:
            # 動画の解放
            cap.release()
            out.release()
            cv2.destroyAllWindows()

def process_session(config, session_path):
    pipeline_calibration_videos = config['pipeline']['calibration_videos']
    pipeline_calibration_results = config['pipeline']['calibration_results']
    video_ext = config['video_extension']
    display_calibration_flag = config['calibration']['display_calibration_result']

    print(session_path)
    
    # 'calibration'という名前のフォルダを見つけて、pathを返す(見つからなかったらなにも返さない)
    calibration_path = find_calibration_folder(config, session_path)

    if calibration_path is None:
        return

    videos = glob(os.path.join(calibration_path,
                               pipeline_calibration_videos,
                               '*.'+video_ext))
    videos = sorted(videos)


    cam_videos = defaultdict(list)
    cam_names = set()
    for vid in videos:
        # cam_regexパラメータの正規表現に従ってnameを抽出
        name = get_cam_name(config, vid)
        cam_videos[name].append(vid)
        cam_names.add(name)

    cam_names = sorted(cam_names)

    video_list = [sorted(cam_videos[cname]) for cname in cam_names]

    outname_base = 'calibration.toml'
    outdir = os.path.join(calibration_path, pipeline_calibration_results)
    outname = os.path.join(outdir, outname_base)

    print(outname)
    skip_calib = False
    init_stuff = True
    error = None

    # charucoボードのインスタンスを生成(aniposelibのboard.pyで定義されたクラス)
    board = get_calibration_board(config)

    # calibration.tomlがcalibrationフォルダの中にあるかどうか
    if os.path.exists(outname):
        cgroup = CameraGroup.load(outname)
        # コーナー検出を行う(ボードの実座標の軸もプロット)
        if display_calibration_flag:
            EvaluateCornerDetection(board, video_list, outdir)

        if (not config['calibration']['animal_calibration']) or \
           ('adjusted' in cgroup.metadata and cgroup.metadata['adjusted']):
            return
        else:
            skip_calib = True
            if 'error' in cgroup.metadata:
                error = cgroup.metadata['error']
            else:
                error = None
        init_stuff = False
    elif config['calibration']['calibration_init'] is not None:
        calib_path = os.path.join(config['path'], config['calibration']['calibration_init'])
        print('loading calibration from: {}'.format(calib_path))
        cgroup = CameraGroup.load(calib_path)
        init_stuff = False
        skip_calib = len(videos) == 0
    else:
        if len(videos) == 0:
            print('no videos or calibration file found, continuing...')
            return
        # aniposelib内のモジュールによってcgroupクラスを作成
        cgroup = CameraGroup.from_names(cam_names, config['calibration']['fisheye'])


    if not skip_calib:
        rows_fname = os.path.join(outdir, 'detections.pickle')
        # すでにボードのキャリブレーションが済んでいるかどうかの確認(pickleファイルの有無)
        if os.path.exists(rows_fname):
            with open(rows_fname, 'rb') as f:
                all_rows = pickle.load(f)
        else:
            # video_listの各videoの全フレームの全コーナーを検出して、その画像座標、実座標、その他の必要情報を辞書にして返す
            all_rows = cgroup.get_rows_videos(video_list, board)
            with open(rows_fname, 'wb') as f:
                pickle.dump(all_rows, f)

        if display_calibration_flag:
            EvaluateCornerDetection(board, video_list, outdir)

        cgroup.set_camera_sizes_videos(video_list)

        # cgroup.calibrate_rows(all_rows, board,
        #                       init_extrinsics=init_stuff,
        #                       init_intrinsics=init_stuff,
        #                       max_nfev=100, n_iters=2,
        #                       n_samp_iter=100, n_samp_full=300,
        #                       verbose=True)

        # calibrationの実行(バンドル調整によって記述される損失関数を最小化してカメラパラメータを最適化)
        # 最適化されたカメラパラメータは各カメラのオブジェクトのプロパティとして格納される
        # 全カメラの組み合わせでの全フレームにおける再投影誤差の中央値を返す
        pickle_save_path = os.path.join(calibration_path, pipeline_calibration_results)
        error = cgroup.calibrate_rows(all_rows, board,
                          init_intrinsics=init_stuff, init_extrinsics=init_stuff,
                          max_nfev=200, n_iters=6,
                          n_samp_iter=200, n_samp_full=1000,
                          verbose=True,
                          pickle_save_path=pickle_save_path)

    cgroup.metadata['adjusted'] = False
    if error is not None:
        cgroup.metadata['error'] = float(error)

    # outnameで指定したファイル名でtomlファイルを作成(各カメラの最終的なパラメータ情報等を記録)
    cgroup.dump(outname)

    if config['calibration']['animal_calibration']:
        all_points, all_scores, all_cam_names = load_2d_data(config, calibration_path)
        imgp = process_points_for_calibration(all_points, all_scores)
        # error = cgroup.bundle_adjust(imgp, threshold=10, ftol=1e-4, loss='huber')
        cgroup = cgroup.subset_cameras_names(all_cam_names)
        error = cgroup.bundle_adjust_iter(imgp, ftol=1e-4, n_iters=10,
                                          n_samp_iter=300, n_samp_full=1000,
                                          max_nfev=500,
                                          verbose=True)
        cgroup.metadata['adjusted'] = True
        cgroup.metadata['error'] = float(error)

    # outnameで指定したファイル名でtomlファイルを作成
    cgroup.dump(outname)

    # コーナー検出を行う(ボードの実座標の軸もプロット)
    # matrix,distortion,rotationのデータをpickleファイルに保存
    matrix_dict={}
    distortion_dict={}
    for cam in cgroup.cameras:
        cam_name = cam.name
        matrix_dict[cam_name]=cam.matrix
        distortion_dict[cam_name]=cam.dist

    # all_rowsのrvec,tvecの更新(matrixとdistortionを新しくしたので)
    for i, (row, cam) in enumerate(zip(all_rows, cgroup.cameras)):
        all_rows[i] = board.estimate_pose_rows(cam, row)
    
    # 保存
    matrix_file_name = os.path.join(pickle_save_path, 'optimized_intrinsic_matrix.pickle')
    distortion_file_name = os.path.join(pickle_save_path, 'optimized_distortion_vector.pickle')
    all_rows_file_name = os.path.join(pickle_save_path, 'optimized_all_rows.pickle')
    if not os.path.exists(all_rows_file_name):
        with open(all_rows_file_name, 'wb') as f:
            pickle.dump(all_rows, f)

    if not os.path.exists(matrix_file_name):
        with open(matrix_file_name, 'wb') as f:
            pickle.dump(matrix_dict, f)

    if not os.path.exists(distortion_file_name):
        with open(distortion_file_name, 'wb') as f:
            pickle.dump(distortion_dict, f)

    if display_calibration_flag:
        EvaluateCornerDetection(board, video_list, outdir)

calibrate_all = make_process_fun(process_session)
