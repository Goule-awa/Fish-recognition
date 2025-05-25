import sys
import torch
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from model.fishnet import FishNet


class FishApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model()

    def initUI(self):
        # 主界面设置
        self.setWindowTitle('智能垂钓鱼类识别系统')
        self.setGeometry(300, 300, 800, 600)

        # 控件布局
        self.image_label = QLabel(self)
        self.result_label = QLabel('识别结果：', self)
        self.btn_load = QPushButton('上传图片', self)
        self.btn_cam = QPushButton('实时识别', self)

        # 布局管理
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_cam)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 事件绑定
        self.btn_load.clicked.connect(self.load_image)

    def load_model(self):
        self.model = FishNet(num_classes=15)
        self.model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu'))
        self.model.eval()
        self.labels = open('data/class_labels.txt').read().splitlines()

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择垂钓图片', '', 'Image files (*.jpg *.png)')
        if fname:
            # 显示图片
            pixmap = QPixmap(fname)
            self.image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))

            # 预处理
            image = Image.open(fname).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(image).unsqueeze(0)

            # 推理
            with torch.no_grad():
                output = self.model(input_tensor)
                pred = torch.argmax(output).item()

            # 显示结果
            self.result_label.setText(f'识别结果：{self.labels[pred]}')
            self.check_protected_species(self.labels[pred])

    def check_protected_species(self, species):
        # 保护物种检查逻辑
        protected_list = ['中华鲟', '长江刀鱼']
        if species in protected_list:
            QMessageBox.warning(self, '保护物种警告',
                                f'检测到{species}！请立即放生！\n根据《长江保护法》第XX条规定...')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FishApp()
    ex.show()
    sys.exit(app.exec_())