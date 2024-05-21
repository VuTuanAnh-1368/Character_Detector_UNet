from sklearn.cluster import KMeans
import cv2
import numpy as np

class PostProcessing:
    def __init__(self):
        pass

    from sklearn.cluster import KMeans

def get_centers(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cy = M['m10'] / M['m00']
            cx = M['m01'] / M['m00']
        else:
            cy, cx = cnt[0][0]
        cx = int(np.round(cx))
        cy = int(np.round(cy))
        centers.append([cx, cy])
    centers = np.array(centers)
    return centers

def get_labels(centers, shape):
    if len(centers) == 0:
        return
    kmeans = KMeans(len(centers), init=centers)
    kmeans.fit(centers)
    coords = []
    mlt = 2
    for i in range(0, shape[0], mlt):
        coords.append([])
        for j in range(0, shape[1], mlt):
            coords[-1].append([i, j])
    coords = np.array(coords).reshape([-1, 2])
    preds = kmeans.predict(coords)
    preds = preds.reshape([shape[0] // mlt, shape[1] // mlt])
    labels = np.zeros(shape, dtype='int')
    for k in range(mlt):
        labels[k::mlt, k::mlt] = preds
    return labels

def get_voronoi(centers, mask):
    labels = get_labels(centers, mask.shape)
    colors = np.random.uniform(0, 1, size=[len(centers), 3])
    voronoi = colors[labels]
    voronoi *= mask[:, :, None]
    return voronoi

def get_rectangles(centers, mask):
    mask_sq = to_square(mask)
    centers_sq = coords_to_square(centers, mask.shape)
    labels_sq = get_labels(centers_sq, mask_sq.shape)
    rects = [None for _ in centers]
    valid_centers = []
    for i, (xc, yc) in enumerate(centers):
        msk = (labels_sq == i).astype('float') * mask_sq / mask_sq.max()
        # crop msk
        max_size = 400
        x1 = max(0, int(np.round(xc - max_size // 2)))
        y1 = max(0, int(np.round(yc - max_size // 2)))
        x2 = min(mask.shape[0], int(np.round(xc + max_size // 2)))
        y2 = min(mask.shape[1], int(np.round(yc + max_size // 2)))
        msk = unsquare(msk, mask.shape[0], mask.shape[1], coords=[[x1,y1], [x2, y2]])
        msk = cv2.inRange(msk, 0.5, 10000)
        contours, _ = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            y, x, h, w = cv2.boundingRect(cnt)
            x += x1
            y += y1
            if xc >= x and xc <= x + w and yc >= y and yc <= y + h:
                rects[i] = [x, y, w, h]
                if cv2.contourArea(cnt) <= h * w * 0.66:
                    rad_x = min(xc - x, x + w - xc)
                    rad_y = min(yc - y, y + h - yc)
                    rects[i] = [int(np.round(xc - rad_x)), y, int(np.round(2 * rad_x)), h]
                break
        if rects[i] is not None:
            valid_centers.append([xc, yc])
    return np.array([r for r in rects if r is not None]), np.array(valid_centers)

def draw_rectangles(img, rects, centers, fill_rect=[1, 0, 0], fill_cent=[1, 0, 0], fill_all=False):
    new = np.array(img)
    for x, y, w, h in rects:
        for shift in range(4):
            try:
                if fill_all:
                    new[x: x + w, y: y + h] = fill_rect
                else:
                    new[x: x + w, y + shift] = fill_rect
                    new[x: x + w, y + h - shift] = fill_rect
                    new[x + shift, y: y + h] = fill_rect
                    new[x + w - shift, y: y + h] = fill_rect
            except:
                pass
    for x, y in centers:
        r = 2
        new[x - r: x + r, y - r: y + r] = fill_cent
    return new

def add_skipped(mask, boxes, centers):
    avg_w = np.mean([b[2] for b in boxes])
    avg_area = np.mean([b[2] * b[3] for b in boxes])
    new_centers, new_boxes = [], []
    mask_c = draw_rectangles(mask, boxes, [], 0, fill_all=True)
    contours, _ = cv2.findContours(mask_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        y, x, h, w = cv2.boundingRect(cnt)
        found = False
        for xc, yc in centers:
            if xc >= x and xc <= x + w and yc >= y and yc <= y + h:
                found = True
                break
        if not found and (w * h > avg_area * 0.66 or w > avg_w * 1.5):
            new_centers.append([x + w // 2, y + h // 2])
            new_boxes.append([x, y, w, h])
    if len(new_centers) > 0:
        boxes = np.concatenate([boxes, new_boxes], 0)
        centers = np.concatenate([centers, new_centers], 0)
    return boxes, centers