from common_header import *


class PixelManager2D:
    """
    회선 처리 완료 좌표들 관리 하는 클래스
    """

    def __init__(self, image, width, height):
        self.layer = np.zeros((width, height))    # 마우스 움직 이는 영역 저장
        self.touch_effect_layer = np.zeros_like(image)  # 지금 까지 처리 한거 저장
        self.mouse_cursor_layer = np.zeros_like(image)  # 마우스 포인터
        self.convolution_applied_layer = image.copy()

    def get_convolution_applied_layer(self):
        return self.convolution_applied_layer

    def get_layer(self):
        return self.layer

    def clear_layer(self):
        self.layer[:] = 0

    def get_mouse_cursor_layer(self):
        return self.mouse_cursor_layer

    def clear_mouse_cursor_layer(self):
        self.mouse_cursor_layer[:] = 0

    def set_touch_effect_layer_pixels(self, y, x, channel, value):
        self.touch_effect_layer[y, x, channel] = value

    def get_touch_effect_layer_pixels(self, y, x, channel):
        return self.touch_effect_layer[y, x, 2]

    def get_touch_effect_layer(self):
        return self.touch_effect_layer
