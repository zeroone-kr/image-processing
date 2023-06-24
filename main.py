import threading

from common_header import *
from pixel_manager import PixelManager2D
from config import *


# 파일 저장
def save_images(user_interaction_image_filename, non_processed_image_filename):
    """
    :param user_interaction_image_filename: 사용자 상호작용 이미지 파일 이름
    :param non_processed_image_filename: 처리한 부분을 제외한 이미지 파일 이름
    :return:
    """
    global touch_effect_layer, convolution_applied_layer

    processed_image = touch_effect_layer.copy()
    processed_image[processed_image[:, :, 2] != 127] = 255
    processed_image[processed_image[:, :, 2] == 127] = 0
    non_processed_image = cv2.bitwise_and(convolution_applied_layer, processed_image)
    print(convolution_applied_layer[1:-1, 1:-1].shape, non_processed_image[1:-1, 1:-1].shape)

    cv2.imwrite(user_interaction_image_filename, convolution_applied_layer[1:-1, 1:-1])
    cv2.imwrite(non_processed_image_filename, non_processed_image[1:-1, 1:-1])


def differential(coordinates):
    """
    프리윗과 소벨을 위한 함수
    :param coordinates: 클릭한 원 내부 좌표들
    :return: (dst_value: (x, y)에 들어갈 값 , x_s:처리한 좌표 x목록, y_s: 처리한 좌표 y목록)
    """
    global original

    dst_value = np.zeros_like(original)
    dst1_value, x_s, y_s = apply_filter(MODE[current_mask][0], coordinates)  # 수평 마스크
    dst2_value, _, _ = apply_filter(MODE[current_mask][1], coordinates)  # 수직 마스크
    dst1_value = dst1_value.astype(np.float32)
    dst2_value = dst2_value.astype(np.float32)
    dst_value = np.clip(cv2.sqrt(dst1_value ** 2 + dst2_value ** 2), 0, 255)
    return dst_value, x_s, y_s


def apply_filter(kernel, coordinates):
    """
    회선을 직접 수행 하는 함수
    :param kernel: 적용할 커널
    :param coordinates: 클릭한 원 내부 좌표들
    :return:
    """
    global original

    dst_value = np.zeros_like(original)
    x_s, y_s = [], []
    for coordinate in coordinates:
        x, y = coordinate[0], coordinate[1]
        if pixel_manager2d.get_touch_effect_layer_pixels(y, x, 2) != 0:
            continue

        # 패딩 영역 화소는 회선 제외
        if x < 1 or y < 1 or x > WIDTH or y > HEIGHT:
            continue

        roi = original[y - 1:y + 2, x - 1:x + 2].astype('float32')
        tmp = cv2.multiply(roi, kernel)
        b, g, r, _ = cv2.sumElems(tmp)
        dst_value[y, x, :] = np.clip([b, g, r], 0, 255)
        x_s.append(x)
        y_s.append(y)
    return dst_value, x_s, y_s


def get_coordinates(matrix):
    indices = np.where(matrix == 1)
    coordinates = np.array(list(zip(indices[1], indices[0])))
    return coordinates


def do_filtering(coordinates_in_circle):
    """
    모드에 따라 회선 수행 하도록 포워딩
    :param x: 클릭한 마우스 x좌표
    :param y: 클릭한 마우스 y좌표
    :param coordinates_in_circle: 클릭한 좌표 (x, y)를 중심 원 내부 좌표
    :return:
    """

    if current_mask == PREWITT or current_mask == SOBEL:
        dst_value, x_s, y_s = differential(coordinates_in_circle)
    else:
        dst_value, x_s, y_s = apply_filter(MODE[current_mask], coordinates_in_circle)

    apply_image(dst_value, x_s, y_s)


def apply_image(dst_value, x_s, y_s):
    """
    실제 이미지에 변경한 값 적용
    :param dst_value: (x, y)에 들어갈 값
    :param x_s: 처리한 좌표 x목록
    :param y_s: 처리한 좌표 y목록
    :return:
    """
    global convolution_applied_layer
    convolution_applied_layer[y_s, x_s] = dst_value[y_s, x_s]
    pixel_manager2d.set_touch_effect_layer_pixels(y_s, x_s, 2, 127)


def mouse_callback(event, x, y, flags, param):
    """
    마우스 이벤트 처리 함수
    :param event: 이벤트
    :param x: 마우스 클릭 x좌표
    :param y: 마우스 클릭 y좌표
    :param flags:
    :param param:
    :return:
    """
    global drawing, prev_x, prev_y, convolution_applied_prev_x, convolution_applied_prev_y
    layer = pixel_manager2d.get_layer()

    # 마우스 왼쪽 버튼 클릭할 때 회선 처리
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # 해당 화소가 처리 안된 경우만
        if pixel_manager2d.get_touch_effect_layer_pixels(y, x, 2) == 0:
            cv2.circle(layer, (x, y), RADIUS, 1, -1)
            coordinates = get_coordinates(layer)
            do_filtering(coordinates)
            pixel_manager2d.clear_layer()
            convolution_applied_prev_x, convolution_applied_prev_y = x, y


    # 마우스 움직일때 회선 처리
    elif event == cv2.EVENT_MOUSEMOVE:
        # print('event_mouse -> ', y, x)
        if prev_x != x or prev_y != y:
            pixel_manager2d.clear_mouse_cursor_layer()

        cv2.circle(mouse_cursor_layer, (x, y), RADIUS, (0, 0, 127), -1)

        # 해당 화소가 처리 안된 경우만
        if drawing and pixel_manager2d.get_touch_effect_layer_pixels(y, x, 2) == 0:
            if convolution_applied_prev_y == -1 and convolution_applied_prev_x == -1:
                pass
            else:
                cv2.circle(layer, (prev_x, prev_y), RADIUS, 1, -1)
                coordinates = get_coordinates(layer)
                do_filtering(coordinates)
                pixel_manager2d.clear_layer()

                cv2.circle(layer, (x, y), RADIUS, 1, -1)
                coordinates = get_coordinates(layer)
                do_filtering(coordinates)
                pixel_manager2d.clear_layer()

                cv2.line(layer, (prev_x, prev_y), (x, y), 1, 2 * RADIUS)
                coordinates = get_coordinates(layer)
                do_filtering(coordinates)
                pixel_manager2d.clear_layer()

            convolution_applied_prev_x, convolution_applied_prev_y = x, y
        prev_x, prev_y = x, y

    # 마우스 왼쪽 버튼을 떼었을 때 회선 처리
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 해당 화소가 처리 안된 경우만
        if pixel_manager2d.get_touch_effect_layer_pixels(y, x, 2) == 0:
            cv2.circle(layer, (x, y), RADIUS, (0, 0, 1), -1)
            coordinates = get_coordinates(layer)
            do_filtering(coordinates)
            convolution_applied_prev_x, convolution_applied_prev_y = -1, -1
            pixel_manager2d.clear_layer()


if __name__ == '__main__':

    # 그리기 중인지 확인
    drawing = False

    # 입력 영상 로드, 회선 처리 대상
    original = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_COLOR)

    # 입력 영상 크기 조정
    original = cv2.resize(original, (WIDTH, HEIGHT))
    original = cv2.copyMakeBorder(original, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    print(original[0, 0, :])
    print(original[HEIGHT + 1, WIDTH + 1, :])
    # print(original.shape)

    # 화소 처리 여부 관리 하는 녀석 (2D)
    pixel_manager2d = PixelManager2D(original, HEIGHT, WIDTH)

    # 사용자 상호 작용을 위한 레이어
    user_interaction_layer = original.copy()

    # 터치 효과 저장 레이어
    touch_effect_layer = pixel_manager2d.get_touch_effect_layer()

    # 원 레이어
    mouse_cursor_layer = pixel_manager2d.get_mouse_cursor_layer()

    # convolution 적용 레이버
    convolution_applied_layer = pixel_manager2d.get_convolution_applied_layer()

    # 이전 마우스 좌표
    prev_x, prev_y = 0, 0
    convolution_applied_prev_x, convolution_applied_prev_y = -1, -1

    # 회선 마스크 초기화
    current_mask = BLURRING

    # 윈도우 생성 및 마우스 이벤트 설정
    cv2.namedWindow('Touch Effect')
    cv2.setMouseCallback('Touch Effect', mouse_callback)

    while True:

        cv2.addWeighted(convolution_applied_layer, 1, mouse_cursor_layer, 0.5, 0, user_interaction_layer)
        cv2.putText(user_interaction_layer, MENU, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # 영상 출력
        cv2.imshow('Touch Effect', user_interaction_layer)

        # 키보드 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_mask = BLURRING

        elif key == ord('2'):
            current_mask = SHARPENING

        elif key == ord('3'):
            current_mask = PREWITT

        elif key == ord('4'):
            current_mask = SOBEL

        elif key == ord('5'):
            current_mask = LAPLACIAN

        elif key == ord('q'):  # 'q' 키를 누르면
            save_images(OUT_FILE1_PATH, OUT_FILE2_PATH)
            break
