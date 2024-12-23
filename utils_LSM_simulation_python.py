from skimage import io
import numpy as np
import random
import os

## Python으로 LSM simulation 만드는 코드
### Ground Truth 3D 이미지 한 장을 불러오고 시작해야 함
gt = io.imread('D:/210827_LSM_simulation/trash.tif')
S = 65535 * np.ones_like(gt)

### 이미지 폴더 단위로 저장 (GT는 따로 저장하기)
path_Y = 'D:/210827_LSM_simulation/231030_input_Y_110vol_ExcludePSF'
path_S = 'D:/210827_LSM_simulation/231030_synthetic_S_110vol_ExcludePSF'
os.makedirs(path_Y, exist_ok=True)
os.makedirs(path_S, exist_ok=True)

### Parameters
max_h = 2048       # 이미지의 세로 길이
thick = 10         # line artifact의 두께
count = 100        # obstacle 개수
stength_min = 0.5  # mask 최소값
stength_max = 1    # mask에서 1을 제외한 line artifact의 최대값

for i in range(len(S)):
    temp = np.random.randint(0, 2048-thick, size=100)
    for t in temp:
        S[i, t:t+thick, :] = 65535 * (stength_min + (stength_max - stength_min) * random.random())

    io.imsave(f'{path_S}/artifact_{i+1}.tif', S[i])
    Y = np.multiply(gt[i], S[i]/65535)
    io.imsave(f'{path_Y}/LSF_{i+1}.tif', Y.astype('uint16'))

