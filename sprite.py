import cv2
import numpy as np
import math

def cv2_putSprite_alpha(back, front4, pos, alpha=255):
    # 行列の範囲外に出ないよう表示域を制限する
    x, y = pos
    fh, fw = front4.shape[:2]
    bh, bw = back.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)                           # 左上座標
    x2, y2 = min(x+fw, bw), min(y+fh, bh)                   # 右下座標

    if not ((-fw < x < bw) and (-fh < y < bh)) :            # 完全に背景からはみ出ていたら
        return back                                         # 何もせず返す
    
    # 表示域の部分のみにトリミングする
    image = back.copy()                                     # 加工する背景
    front = front4[:, :, :3]                                # RGBA画像のRGB要素
    front = front[y1-y:y2-y, x1-x:x2-x]                     # を、必要ならばトリミングする
    roi = image[y1:y2, x1:x2]                               # 背景画像も同様にトリミングする
    mask = front4[:, :, 3]                                  # RGBA画像のA要素
    mask = cv2.merge((mask, mask, mask))                    # 3chにする
    mask = 1/255 * mask.astype(np.float64)                  # 0~1の小数値にする
    mask = mask[y1-y:y2-y, x1-x:x2-x]                       # マスクも同様にトリミングする

    # 合成
    compo = alpha * front + (1-alpha) * roi                 # 合成画像
    compo_with_mask = compo * mask                          # 合成画像　マスクで前景のみにする
    back_with_mask = roi * (1-mask)                         # 背景　マスクで前景を黒にする
    roi = compo_with_mask + back_with_mask                  # 前景と背景を合成
    image[y1:y2, x1:x2] = roi                               # 以上の加工画像を元の背景画像に組み込む
    return image


def cv2_putSprite(back, front4, pos, angle=0, home=(0,0), alpha=1):
    fh, fw = front4.shape[:2]
    bh, bw = back.shape[:2]
    x, y = pos
    xc, yc = home[0] - fw/2, home[1] - fh/2             # homeを左上基準から画像中央基準にする
    a = np.radians(angle)
    cos , sin = np.cos(a), np.sin(a)                    # この三角関数は何度も出るので変数にする
    w_rot = int(fw * abs(cos) + fh * abs(sin))
    h_rot = int(fw * abs(sin) + fh * abs(cos))
    M = cv2.getRotationMatrix2D((fw/2,fh/2), angle, 1)  # 画像中央で回転
    M[0][2] += w_rot/2 - fw/2
    M[1][2] += h_rot/2 - fh/2
    imgRot = cv2.warpAffine(front4, M, (w_rot,h_rot))   # 回転画像を含む外接四角形

    # 外接四角形の全体が背景画像外なら何もしない
    xc_rot = xc * cos + yc * sin                        # 画像中央で回転した際の移動量
    yc_rot = -xc * sin + yc * cos
    x0 = int(x - xc_rot - w_rot / 2)                    # 外接四角形の左上座標   
    y0 = int(y - yc_rot - h_rot / 2)
    if not ((-w_rot < x0 < bw) and (-h_rot < y0 < bh)) :
        return back

    # 外接四角形のうち、背景画像内のみを取得する
    x1, y1 = max(x0,  0), max(y0,  0)
    x2, y2 = min(x0 + w_rot, bw), min(y0 + h_rot, bh)
    imgRot = imgRot[y1-y0:y2-y0, x1-x0:x2-x0]

    # マスク手法で外接四角形と背景を合成する
    result = back.copy()
    front = imgRot[:, :, :3]
    mask = imgRot[:, :, 3]                                  # RGBA画像のA要素
    mask = cv2.merge((mask, mask, mask))                    # 3chにする
    mask = 1/255 * mask.astype(np.float64)                  # 0~1の小数値にする

    roi = result[y1:y2, x1:x2]
    compo = alpha * front + (1-alpha) * roi                 # 合成画像
    compo_with_mask = compo * mask                          # 合成画像　マスクで前景のみにする
    back_with_mask = roi * (1-mask)                         # 背景　マスクで前景を黒にする
    roi = (compo_with_mask + back_with_mask).astype(np.uint8)                  # 前景と背景を合成
    result[y1:y2, x1:x2] = roi
    return result

def main():
    front_RGBA = cv2.imread("ghost.png", -1)        # 前景
    back_origin = cv2.imread("background.png")      # 背景
    height, width = back_origin.shape[:2]

    for i in range(0, 5*360, 10):
        a = 2 * math.pi * i / 360
        x = int(width//2 + 20*math.sin(a))
        y = int(height//2 + 20*math.cos(a))
        alpha = 0.5 * (math.sin(2*a) + 1)               # アルファ値
        image = cv2_putSprite(back_origin, front_RGBA, (x,y), angle=i, home=(0,0), alpha=alpha)
        cv2.imshow("image", image)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
