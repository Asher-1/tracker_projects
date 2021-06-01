# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import os
import sys
import cv2
import glob
from tools.test import *

ROOT_PATH = '/home/yons/develop/AI/projects/SiamMask/experiments/siammask_sharp/'
model_file = ROOT_PATH + "SiamMask_DAVIS.pth"
config_file = ROOT_PATH + "config_davis.json"
video_path = "../data/video_3.flv"
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default=model_file, type=str, required=False,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default=config_file,
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default=video_path, help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode', default=True)
args = parser.parse_args()


def get_roi_img(img, rect):
    box = np.int0(cv2.boxPoints(rect))
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = np.copy(img[y1:y1 + hight, x1:x1 + width])
    return cropImg


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom

    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    cap = cv2.VideoCapture(args.base_path)
    ret, frame = cap.read()
    # 如果无法读取视频文件就退出
    if not ret:
        print('Failed to read video')
        sys.exit(1)

    # Select ROI
    cv2.namedWindow("camera2", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("camera1", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('camera1', frame, False, False)
        x, y, w, h = init_rect
    except:
        exit()

    toc = 0
    f = 0
    while cap.isOpened():
        ret, im = cap.read()
        if not ret:
            break

        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            try:
                rect = cv2.minAreaRect(state['ploygon'])
            except Exception as e:
                print(e)
                continue
            target_img = get_roi_img(im, rect)

            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('camera1', im)
            cv2.imshow('camera2', target_img)
            key = cv2.waitKey(1)
            # if key > 0:
            #     break
        f += 1
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
