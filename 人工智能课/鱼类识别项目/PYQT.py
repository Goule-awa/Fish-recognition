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
        # ����������
        self.setWindowTitle('���ܴ�������ʶ��ϵͳ')
        self.setGeometry(300, 300, 800, 600)

        # �ؼ�����
        self.image_label = QLabel(self)
        self.result_label = QLabel('ʶ������', self)
        self.btn_load = QPushButton('�ϴ�ͼƬ', self)
        self.btn_cam = QPushButton('ʵʱʶ��', self)

        # ���ֹ���
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_cam)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # �¼���
        self.btn_load.clicked.connect(self.load_image)

    def load_model(self):
        self.model = FishNet(num_classes=15)
        self.model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu'))
        self.model.eval()
        self.labels = open('data/class_labels.txt').read().splitlines()

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'ѡ�񴹵�ͼƬ', '', 'Image files (*.jpg *.png)')
        if fname:
            # ��ʾͼƬ
            pixmap = QPixmap(fname)
            self.image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))

            # Ԥ����
            image = Image.open(fname).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(image).unsqueeze(0)

            # ����
            with torch.no_grad():
                output = self.model(input_tensor)
                pred = torch.argmax(output).item()

            # ��ʾ���
            self.result_label.setText(f'ʶ������{self.labels[pred]}')
            self.check_protected_species(self.labels[pred])

    def check_protected_species(self, species):
        # �������ּ���߼�
        protected_list = ['�л���', '��������']
        if species in protected_list:
            QMessageBox.warning(self, '�������־���',
                                f'��⵽{species}��������������\n���ݡ���������������XX���涨...')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FishApp()
    ex.show()
    sys.exit(app.exec_())