import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QColor, QImage
from PyQt5.QtWidgets import QWidget


class Canvas(QWidget):
    def __init__(self, parent, width, height, resolution=64):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.setFixedSize(self.width, self.height)
        self.resolution = resolution

        self.image = QImage(resolution, resolution, QImage.Format.Format_Grayscale8)
        self.image.fill(Qt.white)
        self.converted_image = None

        self.drawing = False
        self.last_point_canvas = QPoint()
        self.last_point_screen = QPoint()
        self.last_width = 0

    def set_resolution(self, resolution):
        self.resolution = resolution
        self.image = QImage(resolution, resolution, QImage.Format.Format_Grayscale8)
        self.image.fill(Qt.white)

    def paintEvent(self, event):
        # Rysowanie płótna
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        self.drawing = True
        self.last_point_screen = event.pos()
        self.last_point_canvas = self.scale_point_to_canvas(event.pos())

    def scale_point_to_canvas(self, point):
        return point.x() / (self.width / self.resolution), point.y() / (self.height / self.resolution)

    def drawPoint(self, pos, width):
        pixel_pos = (int(pos[0]), int(pos[1]))
        width_px = int(width)
        for dy in range(-width_px, width_px + 2):
            for dx in range(-width_px, width_px + 2):
                if width_px != 0:
                    if pixel_pos[0] + dx < 0 or pixel_pos[0] + dx >= self.resolution or \
                            pixel_pos[1] + dy < 0 or pixel_pos[1] + dy >= self.resolution:
                        continue

                    color = QColor(self.image.pixelColor(pixel_pos[0] + dx, pixel_pos[1] + dy))

                    distx = pos[0] - (pixel_pos[0] + dx)
                    disty = pos[1] - (pixel_pos[1] + dy)
                    dist = np.sqrt(distx ** 2 + disty ** 2) * 0.1
                    value = 20 - dist * 255 / width_px
                    # print(dist)
                    newR = round(max(min(color.red(), color.red() - value), 0))
                    newG = round(max(min(color.green(), color.green() - value), 0))
                    newB = round(max(min(color.blue(), color.blue() - value), 0))
                    color.setRgb(newR, newG, newB)

                    self.image.setPixelColor(pixel_pos[0] + dx, pixel_pos[1] + dy, color)

    def drawLine(self, p1, p2, width):
        self.drawPoint(p2, width)

    def mouseMoveEvent(self, event):
        if self.drawing:
            pos_screen = event.pos()
            pos_canvas = self.scale_point_to_canvas(event.pos())

            dist = pos_screen - self.last_point_screen
            module = np.sqrt(dist.x() ** 2 + dist.y() ** 2)

            width = max(self.resolution * 0.3 / (module + 1), 1)
            width = self.last_width * 0.9 + width * 0.1
            self.last_width = width

            self.drawLine(self.last_point_canvas, pos_canvas, width)
            self.last_point_screen = pos_screen
            self.last_point_canvas = pos_canvas
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.update()

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def setImage(self, image):
        _image = image.reshape(self.resolution, self.resolution).astype(np.uint8)

        self.image = QImage(_image, self.resolution, self.resolution, QImage.Format.Format_Grayscale8)
        self.update()

    def showResult(self, image, result, prediction):
        try:
            plt.close()
            plt.imshow(image.reshape(self.resolution, self.resolution), cmap='gray', vmin=0, vmax=255)
            plt.title(f"Rozpoznano: {result}")
            plt.xlabel(f"Prawdopodobieństwo: {max(prediction) * 100:.2f}%")
            plt.show()
        except:
            print("Nie można wyświetlić obrazka")

    def getConvertedImage(self):
        width = self.image.width()
        height = self.image.height()

        data = self.image.bits().asstring(width * height)
        self.converted_image = np.frombuffer(data, dtype=np.uint8).astype(np.float32).reshape(1, width * height)
        return self.converted_image
