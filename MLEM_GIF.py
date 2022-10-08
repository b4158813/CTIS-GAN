
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import rgb2gray_01, normalize, psnr, ssim, GIF_filter
from time import perf_counter

def get_psf(dir="./psf_simulated_ctis", save=False): # 需要修改一下，泛用性不强
    '''
        get psf data
    '''
    psf = np.zeros((512, 512, 31))
    for i in range(31):
        psf[..., i] = rgb2gray_01(cv2.imread(f"{dir}/psf_simulated_{i*10+400}.png"))
    if save:
        np.save("./psf.npy", psf)
    return psf


def get_MaxLocVal(psf, partsize=170):
    '''
        input:

            psf (H, W, lambda) -> val

        return:

            1. PsfMaxLocVal: (lambda, order position) -> (x, y, val)
    '''
    PsfMaxLocVal = np.array([[(0, 0, 0.) for i in range(9)] for j in range(psf.shape[-1])]) # psf最大值坐标与权值
    for k in range(psf.shape[-1]):
        sum_k = 0
        for i in range(3):
            for j in range(3):
                tpmax, maxi, maxj = 0., 0, 0
                for ii in range(i*partsize, (i+1)*partsize):
                    for jj in range(j*partsize, (j+1)*partsize):
                        if tpmax < psf[ii, jj, k]:
                            tpmax = psf[ii, jj, k]
                            maxi, maxj = ii, jj
                # print(f"lambda[{k*10+400}], position[{i*3+j}]: {maxi},{maxj},{tpmax}")
                PsfMaxLocVal[k, i*3+j] = (maxi, maxj, tpmax)
                sum_k += tpmax
        for i in range(3):
            for j in range(3):
                PsfMaxLocVal[k, i*3+j][-1] /= sum_k
        # print(PsfMaxLocVal[k, ...])
        # break
    
    # Rel_PsfMaxLocVal = np.zeros_like(PsfMaxLocVal)
    # for k in range(psf.shape[-1]):
    #     for i in range(9):
    #         Rel_PsfMaxLocVal[k, i][0] = PsfMaxLocVal[k, i][0] - PsfMaxLocVal[k, 4][0]
    #         Rel_PsfMaxLocVal[k, i][1] = PsfMaxLocVal[k, i][1] - PsfMaxLocVal[k, 4][1]
    #         Rel_PsfMaxLocVal[k, i][2] = PsfMaxLocVal[k, i][2]
        # print(Rel_PsfMaxLocVal[k, ...])
    print(f"MaxLocVal done.")
    return PsfMaxLocVal


def get_zero_order_size(snapshot, Threshold=3.8820e-5, centerh=255, centerw=255):
    '''
        calculate the size of zero-order image
        
        input:

            1. snapshot image
            2. threshold
            3. centerh
            4. centerw

        return:

                tuple (left top x, left top y, right bottom x, right bottom y)
                left/right included
                   [1]
                    |
                    v
              [0]-> _________
                    |///////|
                    |0-order|
                    |///////|
                    --------- <-[2]
                            ^
                            |
                           [3]    
    '''
    img = snapshot.copy()
    img[img<Threshold] = 0
    img[img>=Threshold] = 1 # 
    colsum = np.sum(img, axis=0)
    rowsum = np.sum(img, axis=1)
    left_top_h, left_top_w, right_bottom_h, right_bottom_w = 0, 0, 0, 0
    for i in range(centerh, -1, -1):
        if rowsum[i] == 0:
            left_top_h = i + 1
            break
    for i in range(centerw, -1, -1):
        if colsum[i] == 0: 
            left_top_w = i + 1
            break
    for i in range(centerh, snapshot.shape[0]):
        if rowsum[i] == 0:
            right_bottom_h = i - 1
            break
    for i in range(centerw, snapshot.shape[1]):
        if colsum[i] == 0:
            right_bottom_w = i - 1
            break
    print(f"get zero-order size done.")
    return (left_top_h, left_top_w, right_bottom_h, right_bottom_w)


def construct_H(PsfMaxLocVal, Shape, Lambda=31):
    '''
        input:

            1. PsfMaxLocVal: (lambda, order position) -> (x, y, val)
            2. Shape: zero-order (left top x, left top y, right bottom x, right bottom y)
            3. Lambda: number of spectral bands
        
        return:

            1. voxel_to_pixels: defaultdict(list)  (x, y, z) -> list((x, y, val))
            2. pixel_to_voxels: defaultdict(list)  (x, y) -> list((x, y, z, val))
    '''
    pixel_to_voxels = defaultdict(list)
    voxel_to_pixels = defaultdict(list)
    zero_order_width = Shape[3] - Shape[1] + 1
    zero_order_height = Shape[2] - Shape[0] + 1

    for i in range(zero_order_height):
        for j in range(zero_order_width):
            for k in range(Lambda):
                psf_center_x, psf_center_y, _ = PsfMaxLocVal[k, 4]
                delt_x = i + Shape[0] - psf_center_x
                delt_y = j + Shape[1] - psf_center_y
                for ipart in range(9):
                    psf_point_x, psf_point_y, psf_point_val = PsfMaxLocVal[k, ipart]
                    voxel_to_pixels[(i,j,k)].append((psf_point_x + delt_x, psf_point_y + delt_y, psf_point_val))
                    pixel_to_voxels[(psf_point_x + delt_x, psf_point_y + delt_y)].append((i,j,k,psf_point_val))
                # print(voxel_to_pixels)
                # print(pixel_to_voxels)
                # return
    print(f"H matrix constructed.")
    return voxel_to_pixels, pixel_to_voxels


def MLEM(snapshot, Shape, pixel_to_voxels, voxel_to_pixels, GIF=False, guided_img=None, ksize=3, eps=1e-5, maxite=100):
    '''
        MLEM algorithm ( with/without Guided Image Filtering)
    '''

    H, W, L = (Shape[2]-Shape[0]+1, Shape[3]-Shape[1]+1, 31)
    datacube = np.ones((H, W, L))
    datacube_temp = np.ones((H, W, L))
    ite = 0
    st_time = perf_counter()
    prev_time = perf_counter()
    while True:
        datacube_temp = datacube.copy()
        for i in range(H):
            for j in range(W):
                for k in range(L):
                    denominator = sum([v for (_, _, v) in voxel_to_pixels[(i, j, k)]])
                    numerator = 0
                    for (x1, y1, v1) in voxel_to_pixels[i, j, k]:
                        predicted_pixval = 0
                        for (x2, y2, z2, v2) in pixel_to_voxels[x1, y1]:
                            predicted_pixval += datacube_temp[x2, y2, z2] * v2
                        numerator += v1 * snapshot[int(x1), int(y1)] / predicted_pixval
                    datacube[i, j, k] = datacube_temp[i, j, k] * numerator / denominator

        if GIF: # Guided Image Filtering
            for k in range(L):
                datacube[..., k] = GIF_filter(datacube[..., k], guided_img=guided_img, ksize=ksize)

        ite += 1
        relative_error = np.sum(datacube - datacube_temp)/np.sum(datacube)
        # print(f"iteration [{ite}], relative eroor = {relative_error}")
        yield ite, perf_counter()-prev_time, relative_error
        prev_time = perf_counter()
        if ite>=maxite or np.abs(relative_error) < eps:
            break
    # print(f"end of MLEM algorithm, total time = {perf_counter() - st_time} s")
    yield datacube, perf_counter()-st_time


if __name__ == '__main__':
    
    psf = get_psf("./psf_simulated_ctis/") # (H,W,31)
    PsfMaxLocVal = get_MaxLocVal(psf)
    snapshot = cv2.imread('./fake_and_real_food_ctis.png')
    snapshot = rgb2gray_01(snapshot)
    Shape = get_zero_order_size(snapshot)
    zero_order = snapshot[Shape[0]: Shape[2]+1, Shape[1]: Shape[3]+1]
    voxel_to_pixels, pixel_to_voxels = construct_H(PsfMaxLocVal, Shape)

    print(f"start of MLEM algorithm")
    for ite_data in MLEM(snapshot, Shape, pixel_to_voxels, voxel_to_pixels, GIF=True, guided_img=zero_order, ksize=5, eps=1e-10, maxite=1):
        if len(ite_data) == 3:
            ite, ti, relative_error = ite_data
            print(f"iteration [{ite}], time spent = {ti} s, relative eroor = {relative_error}")
        else:
            datacube, total_time = ite_data
            break
    print(f"end of MLEM algorithm, total time = {total_time} s")

    numlam = 1
    px, py = 33, 25
    real_datacube = np.load("./fake_and_real_food_hsi.npy")
    print(f"PSNR = {psnr(real_datacube, datacube)}")
    print(f"SSIM = {ssim(real_datacube, datacube)}")

    plt.figure(figsize=(4,4))
    plt.imshow(datacube[:,:,numlam], cmap='gray')
    plt.figure(figsize=(4,4))
    plt.imshow(real_datacube[:,:,numlam], cmap='gray')
    plt.figure(figsize=(6,4))
    plt.plot(normalize(datacube[px,py,:]), label='reconstruction')
    plt.plot(normalize(real_datacube[px,py,:]), label='ground truth')
    plt.legend()
    plt.show()


