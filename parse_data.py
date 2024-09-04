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
import imagehash
from PIL import Image

from tqdm import tqdm


IMAGE_FOLDER = (
    r'C:\Users\zyr17\Documents\Projects\LPSim\frontend\collector\splitter\4.5'
)
PATCH_JSON = r'./frontend/src/descData.json'
GUYU_PATCH_JSON = (
    r'C:\Users\zyr17\Documents\Projects\ban-pick\frontend\src\guyu_json'
)
BACKEND = 'cv2'


def read_video(vname):
    # Open the video file
    video = cv2.VideoCapture(vname)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate the frame skip value (if you want 4 fps and your video is 60 fps, 
    # you need to skip every 15 frames)
    frame_skip = int(fps / 4)

    frame_count = 0
    frames = []

    while True:
        print(f'{frame_count} / {total_frames} frame', end = '\r')
        ret, frame = video.read()

        # If the frame was not retrieved, then we have reached the end of the video
        if not ret:
            break

        # If the frame_count is divisible by frame_skip (i.e., if we are on a frame 
        # that we want to keep), save the frame
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


def get_image_feature_cv2(img):
    sift = cv2.SIFT_create()  # type: ignore
    kp, des = sift.detectAndCompute(img, None)
    return des


def get_image_feature_imagehash(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imagehash.dhash(img, hash_size = 128)


def get_flann_index_params():
    # 使用 FLANN 构建索引
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=5)
    return index_params, search_params


def build_flann_index(features):

    flann = cv2.FlannBasedMatcher(*get_flann_index_params())  # type: ignore
    flann.add([features])
    flann.train()
    return flann


def load_flann(cache_path):
    flann = cv2.FlannBasedMatcher(*get_flann_index_params())  # type: ignore
    flann.read(cache_path)
    return flann


def save_flann(flann: cv2.FlannBasedMatcher, cache_path):
    # TODO load result wrong
    flann.write(cache_path)


def cache_and_build_flann_index(img, cache_key = None, cache_folder = './cache/flann'):
    # TODO: currently save & load will become empty, disable it.
    # cache_path = f'{cache_folder}/{cache_key}'
    # if cache_key is not None and os.path.exists(cache_path):
    #     return load_flann(cache_path)
    feature = get_image_feature(img)
    flann = build_flann_index(feature)
    # save_flann(flann, cache_path)
    return flann


def find_best_match(names, flann, query_feature):
    matches = flann.knnMatch(query_feature, k=2)

    # 获取最匹配的特征的索引和相似度
    best_match_names = [m.trainIdx for m, n in matches]
    match_similarities = [m.distance for m, n in matches]
    print(best_match_names, len(names))
    best_match_names = [names[i] for i in best_match_names]

    return zip(best_match_names, match_similarities)


def compare_images_cv2(feat1, flann):
    des1 = feat1
    # kp2, des2 = feat2

    # 使用 FLANN 匹配器来匹配描述符
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=4)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore

    # print(des1.shape)
    matches = flann.knnMatch(des1, k=2)

    # 仅保留好的匹配项
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # 计算相似度
    similarity = len(good_matches) / len(matches)

    return similarity


def compare_images_imagehash(feat1, feat2):
    return 1 - (feat1 - feat2) / len(feat1.hash) ** 2


def get_helper():
    if BACKEND == 'cv2':
        return get_image_feature_cv2, compare_images_cv2, 2, 0.2
    elif BACKEND == 'imagehash':
        return get_image_feature_imagehash, compare_images_imagehash, 1.1, 0.6
    else:
        raise ValueError('unknown backend')


get_image_feature, compare_images, DIFF_THRESHOLD, MATCH_THRESHOLD = get_helper()


def warn_not_confident(
    sim, diff_threshold = DIFF_THRESHOLD, match_threshold = MATCH_THRESHOLD
):
    """
    diff_threshold: first should be how many times better than second
    match_threshold: how similar should it be
    """
    if sim[0][1] < match_threshold:
        print(f'{sim[0][0]} match too low: {sim[0][1]:.6f}')
    if sim[0][1] < diff_threshold * sim[1][1]:
        print(
            f'{sim[0][0]} not too much better than {sim[1][0]}: '
            f'{sim[0][1]:.6f} {sim[1][1]:.6f}'
        )


def do_one_img(
    character_feats: dict[str, Any], 
    card_feats: dict[str, Any], 
    current_character_feats: list[Any],
    current_card_feats: list[Any],
    verbose: bool = False,
):
    """
    get parts of img, and find their names, return a list of names. 
    """
    # if BACKEND == 'cv2':
        # character_feats = {k: build_flann_index(v[1]) for k, v in character_feats.items()}
        # card_names = list(card_feats.keys())
        # card_feats = [card_feats[name] for name in card_names]
        # card_feats = build_flann_index(card_feats)
    if verbose:
        current_character_feats = tqdm(current_character_feats)
        current_card_feats = tqdm(current_card_feats)
    characters_sim = []
    for current_character_feat in current_character_feats:
        # print(current_character_feat)
        character_sim = []
        for character_name, character_feat in (character_feats.items()):
            similarity = compare_images(current_character_feat, character_feat)
            character_sim.append([character_name, similarity])
        character_sim.sort(key=lambda x: x[1], reverse=True)
        characters_sim.append(character_sim)
    cards_sim = []
    for current_card_feat in current_card_feats:
        card_sim = []
        for card_name, card_feat in (card_feats.items()):
            similarity = compare_images(current_card_feat, card_feat)
            card_sim.append([card_name, similarity])
        card_sim.sort(key=lambda x: x[1], reverse=True)
        cards_sim.append(card_sim)

    # print('\n'.join([f'{x[0]} {x[1]}' for x in character_sim[:3]]))
    # for card_sim in cards_sim:
    #     print('\n'.join([f'{x[0]} {x[1]}' for x in card_sim[:3]]))

    for character_sim in characters_sim:
        warn_not_confident(character_sim)
    for card_sim in cards_sim:
        warn_not_confident(card_sim)

    return [x[0][0] for x in characters_sim], [x[0][0] for x in cards_sim]


def get_all_img_feat_lpsim(patch_json = PATCH_JSON, image_folder = IMAGE_FOLDER):
    desc_data = json.load(open(patch_json, encoding = 'utf8'))
    character_res = {}
    card_res = {}
    for key in tqdm(desc_data):
        data = desc_data[key]
        if '_STATUS/' in key or 'SUMMON/' in key:
            continue
        if key.startswith('CHARACTER/'):
            res = character_res
        else:
            res = card_res
        if 'image_path' in data:
            img = cv2.imread(os.path.join(image_folder, data['image_path']))
            chinese_name = data['names']['zh-CN']
            res[chinese_name] = cache_and_build_flann_index(img, chinese_name)
    # print(character_res.keys(), card_res.keys())
    return character_res, card_res


def get_all_img_feat_guyu(patch_json = GUYU_PATCH_JSON, image_folder = IMAGE_FOLDER):
    character_data = json.load(
        open(patch_json + '/guyu_characters.json', encoding = 'utf8'))
    card_data = json.load(
        open(patch_json + '/guyu_action_cards.json', encoding = 'utf8'))
    character_res = {}
    card_res = {}
    for data, res in [(character_data, character_res), (card_data, card_res)]:
        for one_data in tqdm(data):
            if 'shareId' not in one_data:
                continue
            share_id = one_data['shareId']
            img_path = one_data['cardFace'].replace('UI_Gcg_CardFace_', '') + ".png"
            if not os.path.exists(os.path.join(image_folder, 'cardface', img_path)):
                print(f'{img_path} not exists')
                continue
            img = cv2.imread(os.path.join(image_folder, 'cardface', img_path))
            chinese_name = one_data['name']
            res[chinese_name] = cache_and_build_flann_index(img, str(share_id))
    # print(character_res.keys(), card_res.keys())
    return character_res, card_res


get_all_img_feat = get_all_img_feat_guyu


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
        character, *cards = get_parts(img)
        current_character_feats = [get_image_feature(character)]
        current_card_feats = [get_image_feature(card) for card in cards]
        res = do_one_img(
            character_feats, card_feats, current_character_feats, current_card_feats)
        print(res)
        res_txt.write(res[0][0] + ' ')
        res_txt.write(' '.join(res[1]) + '\n')
        res_json.append(' '.join([res[0][0]] + res[1][2:]))
    json.dump(
        res_json, 
        open('test.json', 'w', encoding='utf8'), 
        ensure_ascii=False,
        indent=4,
    )
    print('done!')
