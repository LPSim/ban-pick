"""
录制一个录像，依次展示每个角色和每个卡牌，然后用这个脚本来解析录像，得到每一帧的角色和卡牌的名字。
录制开始时必须已经展示第一个角色（即甘雨），鼠标放在右箭头上；开始录制后，每个角色详情页展示至少
2秒，然后点击右箭头展示下一个，直到最后一个角色详情页展示完毕，直接停止录制。使用WinAltR开始
和结束录制不会录到多余的东西。如果前后有多余的帧可能会有未知错误或者错误匹配。录制完成后，将
视频文件拷贝到同目录并命名为test.mp4，然后运行这个脚本，脚本会输出每一帧的角色和卡牌的名字。
"""
import json
from typing import Any
import cv2
import os
import numpy as np

from tqdm import tqdm


IMAGE_FOLDER = r'D:\Downloads\assetstudio1.36\GI_map\lpsim-images'
PATCH_JSON = r'./frontend/src/descData.json'


def read_video(vname):
    # Open the video file
    video = cv2.VideoCapture(vname)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate the frame skip value (if you want 4 fps and your video is 60 fps, you need to skip every 15 frames)
    frame_skip = int(fps / 4)

    frame_count = 0
    frames = []

    while True:
        print(f'{frame_count} / {total_frames} frame', end = '\r')
        ret, frame = video.read()

        # If the frame was not retrieved, then we have reached the end of the video
        if not ret:
            break

        # If the frame_count is divisible by frame_skip (i.e., if we are on a frame that we want to keep), save the frame
        if frame_count % frame_skip == 0:
            # cv2.imwrite('frame{}.png'.format(frame_count), frame)
            frames.append(frame)

        frame_count += 1
    print()

    # Release the video file
    video.release()
    return frames


def diff_pixel_ratio(img1, img2):
    """
    Calculate the ratio of different pixels between two images
    """
    imax = np.maximum(img1, img2)
    imin = np.minimum(img1, img2)
    diff = ((imax - imin) > 10).sum(axis = -1) != 0
    assert len(diff.shape) == 2
    return diff.mean()


def filter_video(frames, threshold = 0.02):
    """
    Only keep frames that are different enough from the previous frame and almost
    same as next frame.
    """
    diff = []
    for prev, next in tqdm(zip(frames, frames[1:]), total = len(frames) - 1):
        diff.append(diff_pixel_ratio(prev, next))
    # print(diff)
    res = [frames[0]]
    for diff1, current, diff2 in zip(diff, frames[1:], diff[1:]):
        if diff1 > threshold and diff2 < threshold:
            # print(diff1, diff2)
            res.append(current)
    return res


def get_parts(img):
    """
    scale img to 3840x2160, and get characters and 4 cards
    """
    base_resolution = (3840, 2160)
    character_idx = 863, 635, 1412, 1570
    card_idx: list[tuple[int, int, int, int]] = [(1525, 633, 1785, 1082)]
    delta = 300
    for i in range(3):
        last_card = card_idx[-1]
        card_idx.append(
            (last_card[0] + delta, last_card[1], last_card[2] + delta, last_card[3])
        )
    img_scaled = cv2.resize(img, base_resolution)
    return [
        img_scaled[
            character_idx[1] : character_idx[3], character_idx[0] : character_idx[2]
        ],
        *[img_scaled[y1:y2, x1:x2] for x1, y1, x2, y2 in card_idx],
    ]


def get_image_feature(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def compare_images(feat1, feat2):
    kp1, des1 = feat1
    kp2, des2 = feat2

    # 使用 FLANN 匹配器来匹配描述符
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # 仅保留好的匹配项
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # 计算相似度
    similarity = len(good_matches) / len(matches)

    return similarity


def warn_not_confident(img, sim, diff_threshold = 2, match_threshold = 0.2):
    """
    diff_threshold: first should be how many times better than second
    match_threshold: how similar should it be
    """
    if sim[0][1] < match_threshold:
        print(f'{sim[0][0]} match too low: {sim[0][1]:.6f}')
    if sim[0][1] < diff_threshold * sim[1][1]:
        print(f'{sim[0][0]} not too much better than {sim[1][0]}: {sim[0][1]:.6f} {sim[1][1]:.6f}')


def do_one_img(character_feats: dict[str, Any], card_feats: dict[str, Any], img):
    """
    get parts of img, and find their names, return a list of names. 
    """
    character, *cards = get_parts(img)
    current_character_feat = get_image_feature(character)
    current_cards_feat = [get_image_feature(card) for card in cards]
    # cv2.imshow("character", character)
    # cv2.waitKey(0)
    # for card in cards:
    #     cv2.imshow("card", card)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    character_sim = []
    for character_name, character_feat in (character_feats.items()):
        similarity = compare_images(current_character_feat, character_feat)
        character_sim.append([character_name, similarity])
    character_sim.sort(key=lambda x: x[1], reverse=True)
    cards_sim = []
    for current_card_feat in current_cards_feat:
        card_sim = []
        for card_name, card_feat in (card_feats.items()):
            similarity = compare_images(current_card_feat, card_feat)
            card_sim.append([card_name, similarity])
        card_sim.sort(key=lambda x: x[1], reverse=True)
        cards_sim.append(card_sim)

    # print('\n'.join([f'{x[0]} {x[1]}' for x in character_sim[:3]]))
    # for card_sim in cards_sim:
    #     print('\n'.join([f'{x[0]} {x[1]}' for x in card_sim[:3]]))

    warn_not_confident(character, character_sim)
    for card, card_sim in zip(cards, cards_sim):
        warn_not_confident(card, card_sim)

    return character_sim[0][0], [x[0][0] for x in cards_sim]


def get_all_img_feat():
    desc_data = json.load(open(PATCH_JSON, encoding = 'utf8'))
    character_res = {}
    card_res = {}
    for key in tqdm(desc_data):
        data = desc_data[key]
        if '_STATUS/' in key:
            continue
        if key.startswith('CHARACTER/'):
            res = character_res
        else:
            res = card_res
        if 'image_path' in data:
            img = cv2.imread(os.path.join(IMAGE_FOLDER, data['image_path']))
            img_feat = get_image_feature(img)
            chinese_name = data['names']['zh-CN']
            res[chinese_name] = img_feat
    # print(character_res.keys(), card_res.keys())
    return character_res, card_res


if __name__ == '__main__':
    vd = read_video('test.mp4')
    # for i, frame in enumerate(vd):
    #     cv2.imwrite(f'test_o/frame{i}.png', frame)
    print('frame number', len(vd))
    vd_filtered = filter_video(vd)
    print(f'frame number after filter {len(vd_filtered)}')
    # for i, frame in enumerate(vd_filtered):
    #     cv2.imwrite(f'test/frame{i}.png', frame)

    character_feats, card_feats = get_all_img_feat()
    # img = cv2.imread('test.png')
    res_txt = open('test.txt', 'w')
    res_json = []
    for idx, img in enumerate(vd_filtered):
        print(f'processing {idx} / {len(vd_filtered)} frame')
        res = do_one_img(character_feats, card_feats, img)
        print(res)
        res_txt.write(res[0] + ' ')
        res_txt.write(' '.join(res[1]) + '\n')
        res_json.append([res[0]] + res[1][2:])
    json.dump(res_json, open('test.json', 'w'))
    print('done!')
