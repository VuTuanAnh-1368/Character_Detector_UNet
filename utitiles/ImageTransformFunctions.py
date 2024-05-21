import cv2
import numpy as np

IMG_SIZE = 400
BATCH_SIZE = 4

class ImageTransformFunctions:
    def __init__(self):
        pass

    @staticmethod
    def load_image(path):
        img = cv2.imread(path)
        img = img / 255.
        return img

    def coords_to_square(coords, shape):
        new = []
        w, h = shape[:2]
        for x, y in coords:
            if h > w:
                y = int(np.round(y * IMG_SIZE / h))
                x = x + (h - w) / 2
                x = int(np.round(x * IMG_SIZE / h))
            else:
                x = int(np.round(x * IMG_SIZE / w))
                y = y + (w - h) / 2
                y = int(np.round(y * IMG_SIZE / w))
            new.append([x, y])
        return np.array(new)

    def to_square(img, img_size=IMG_SIZE):
        three_d = len(img.shape) == 3
        if three_d:
            w, h, c = img.shape
        else:
            w, h = img.shape
            c = 1
        if w > h:
            h = int(h * img_size / w)
            w = img_size
        else:
            w = int(w * img_size / h)
            h = img_size
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_NEAREST).reshape([w, h, c])
        margin_w = (img_size - w) // 2
        margin_h = (img_size - h) // 2
        new_img = np.zeros((img_size, img_size, c))
        new_img[margin_w: margin_w + w, margin_h: margin_h + h, :] = img
        if not three_d:
            new_img = new_img.reshape([img_size, img_size])
        return new_img.astype('float32')

    def unsquare(img, width, height, coords=None):
        if coords is None:
            if width > height:
                w = IMG_SIZE
                h = int(height * IMG_SIZE / width)
            else:
                h = IMG_SIZE
                w = int(width * IMG_SIZE / height)
            margin_w = (IMG_SIZE - w) // 2
            margin_h = (IMG_SIZE - h) // 2
            img = img[margin_w: margin_w + w, margin_h: margin_h + h]
            img = cv2.resize(img, (height, width))
        else:
            [x1, y1], [x2, y2] = coords
            [sx1, sy1], [sx2, sy2] = ImageTransformFunctions.coords_to_square(coords, [width, height])
            img = cv2.resize(img[sx1: sx2, sy1: sy2], (y2 - y1, x2 - x1))
        return img

    def get_mask(img, labels):
        mask = np.zeros((img.shape[0], img.shape[1], 2), dtype='float32')
        if isinstance(labels, str):
            labels = np.array(labels.split(' ')).reshape(-1, 5)
            for char, x, y, w, h in labels:
                x, y, w, h = int(x), int(y), int(w), int(h)

                if x + w >= img.shape[1] or y + h >= img.shape[0]:
                    continue
                mask[y: y + h, x: x + w, 0] = 1
                radius = 2
                mask[y + h // 2 - radius: y + h // 2 + radius + 1, x + w // 2 - radius: x + w // 2 + radius + 1, 1] = 1
        return mask

    def preprocess(img, width, height):
        skip = 8
        if width > height:
            w = IMG_SIZE
            h = int(height * IMG_SIZE / width)
        else:
            h = IMG_SIZE
            w = int(width * IMG_SIZE / height)
        margin_w = (IMG_SIZE - w) // 2
        margin_h = (IMG_SIZE - h) // 2
        sl_x = slice(margin_w, margin_w + w)
        sl_y = slice(margin_h, margin_h + h)
        stat = img[margin_w:margin_w + w:skip, margin_h:margin_h + h:skip].reshape([-1, 3])
        img[sl_x, sl_y] = img[sl_x, sl_y] - np.median(stat, 0)
        img[sl_x, sl_y] = img[sl_x, sl_y] / np.std(stat, 0)