from common_header import *


def get_relative_pixels_coordinates_in_circle(radius):
    """
    상대 원내부 좌표들 구함
    :param radius: 원 반지름
    :return:
    """
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    pixel_coordinates_ = (x**2 + y**2)**0.5 <= radius
    relative_pixel_coordinates_ = np.transpose(np.nonzero(pixel_coordinates_)) - np.array([radius, radius])
    return relative_pixel_coordinates_


# 원 바지름 설정
RADIUS = 30


# 상대 좌표 받아 오기
relative_pixel_coordinates = get_relative_pixels_coordinates_in_circle(RADIUS)



# 입력 영상의 크기
WIDTH, HEIGHT = 1920, 1080

# 파일 경로
INPUT_IMAGE_PATH = r'images/19.jpg'

# 출력 파일 2개 이름
OUT_FILE1_PATH = r'./output/20162171_1.jpg'
OUT_FILE2_PATH = r'./output/20162171_2.jpg'

# 좌측 상단 메시지
MENU = '1/blurring, 2/sharpening, 3/prewitt, 4/sobel, 5/laplacian'

# 커널 이름
BLURRING = 'blurring'
SHARPENING = 'sharpening'
PREWITT = 'prewitt'
SOBEL = 'sobel'
LAPLACIAN = 'laplacian'

# 모드에 따른 커널값 설정
MODE = {
    BLURRING:
        np.array([
            [[1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9]],

            [[1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9]],

            [[1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9]]
        ], np.float32),

    SHARPENING:
        np.array([
            [[-1, -1, -1],
             [-1, -1, -1],
             [-1, -1, -1]],

            [[-1, -1, -1],
             [9, 9, 9],
             [-1, -1, -1]],

            [[-1, -1, -1],
             [-1, -1, -1],
             [-1, -1, -1]]
        ], np.float32),

    # 수평마스크, 수직마스크
    PREWITT:
        np.array([
            [
                [[-1, -1, -1],
                 [-1, -1, -1],
                 [-1, -1, -1]],

                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]
            ],
            [
                [[-1, -1, -1],
                 [0, 0, 0],
                 [1, 1, 1]],

                [[-1, -1, -1],
                 [0, 0, 0],
                 [1, 1, 1]],

                [[-1, -1, -1],
                 [0, 0, 0],
                 [1, 1, 1]]
            ]
        ], np.float32),

    # 수평마스크, 수직마스크
    SOBEL:
        np.array([
            [
                [[-1, -1, -1],
                 [-2, -2, -2],
                 [-1, -1, -1]],

                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],

                [[1, 1, 1],
                 [2, 2, 2],
                 [1, 1, 1]]
            ],
            [
                [[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]],

                [[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]],

                [[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]]
            ]
        ], np.float32),

    LAPLACIAN:
        np.array([
            [[-1, -1, -1],
             [-1, -1, -1],
             [-1, -1, -1]],

            [[-1, -1, -1],
             [8, 8, 8],
             [-1, -1, -1]],

            [[-1, -1, -1],
             [-1, -1, -1],
             [-1, -1, -1]]
        ], np.float32)
}

# print(absolute_pixels_coordinates_in_circle.shape, absolute_pixels_coordinates_in_circle.dtype, type(absolute_pixels_coordinates_in_circle))
# print(absolute_pixels_coordinates_in_circle.transpose().shape)

'''
>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>> i = np.array([[0, 1],  # indices for the first dim of `a`
              [1, 2]])
>> j = np.array([[2, 1],  # indices for the second dim
              [3, 3]])
>>> a[i, j]  # i and j must have equal shape
array([[ 2,  5],
       [ 7, 11]])
'''
