# display_utils.py
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


def display_frame(label_widget, rgb_frame):
    h, w, ch = rgb_frame.shape
    bytes_per_line = ch * w
    qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_img)
    label_widget.setPixmap(pixmap.scaled(
        label_widget.width(),
        label_widget.height(),
        Qt.AspectRatioMode.KeepAspectRatio
    ))
