import cv2
import numpy as np

from parse_data import do_one_img, get_all_img_feat, get_image_feature


def crop_on_official_share_image(image):
    # image is PIL, crop cards from it.
    # # 定义RGB颜色
    # rgb_color = np.uint8([[[225, 212, 203]]])  # 例如，RGB颜色为 (220, 213, 205)
    # # 将RGB颜色转换为HSV颜色
    # hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    # print("HSV颜色值:", hsv_color)

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义背景颜色范围
    lower_color = np.array([10, 10, 210])
    upper_color = np.array([30, 30, 230])

    # 创建掩码
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # print(mask, mask.mean(), mask.std(), mask.shape)

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)

    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 获取包围矩形
    x, y, w, h = cv2.boundingRect(largest_contour)

    # print(x, y, w, h)

    # 裁剪图片
    cropped_image = image[y:y+h, x:x+w]
    cropped_hsv = hsv[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    # lower_color_cropped = np.array([3, 3, 210])
    # upper_color_cropped = np.array([30, 40, 230])
    lower_color_cropped = np.array([3, 3, 210])
    upper_color_cropped = np.array([230, 240, 230])
    cropped_mask = cv2.inRange(cropped_hsv, lower_color_cropped, upper_color_cropped)

    print(cropped_mask.min(), cropped_mask.max(), cropped_mask.mean(), cropped_mask.std())

    thresh = 255 - cropped_mask

    # 找到小矩形物体的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = thresh.shape

    iheight, iwidth = height * 0.11111, width * 0.117
    scale_multiplier = [0.9, 1.6]

    # 过滤出符合要求的矩形并统计数量
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            iheight * scale_multiplier[0] < h < iheight * scale_multiplier[1]
            and iwidth * scale_multiplier[0] < w < iwidth * scale_multiplier[1]
        ):
            rectangles.append((x, y, w, h))

    # 统计符合要求的矩形数量
    num_rectangles = len(rectangles)
    if num_rectangles != 33:
        print(f"符合要求的矩形数量与预期不符: {num_rectangles}")

    # print(f"符合要求的矩形数量: {num_rectangles}")

    # 按位置排序
    rectangles = sorted(rectangles, key=lambda r: (r[1] * 20 + r[0]))

    # cut the image
    output_images = []
    for i, (x, y, w, h) in enumerate(rectangles):
        output_images.append(cropped_image[y:y+h, x:x+w])

    # 绘制矩形
    for (x, y, w, h) in rectangles:
        cv2.rectangle(cropped_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 保存并显示结果
    # print(cropped_mask.shape)
    # cropped_image[cropped_mask.astype(bool)] = 255
    # cv2.imwrite('cropped_image_with_rectangles.jpg', cropped_image)
    # cv2.imshow('Cropped Image with Rectangles', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return output_images


if __name__ == '__main__':
    image_path = r'C:\Users\zyr17\Downloads\QQ20240904220500.jpg'
    sub_images = crop_on_official_share_image(cv2.imread(image_path))
    characters = sub_images[:3]
    cards = sub_images[3:]

    image_folder = (
        r'C:\Users\zyr17\Documents\Projects\LPSim\frontend\collector\splitter\4.5'
    )
    patch_json = r'./frontend/src/guyu_json'
    character_feats, card_feats = get_all_img_feat(patch_json, image_folder)
    current_character_feats = [get_image_feature(character) for character in characters]
    current_card_feats = [get_image_feature(card) for card in cards]
    res = do_one_img(
        character_feats, card_feats, current_character_feats, current_card_feats,
        True)
    print(res)
    print('done!')
