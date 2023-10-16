import sys
import os

import numpy as np
from typing import Optional

from PySide6.QtCore import Slot,Qt
from PySide6.QtWidgets import (QApplication,QWidget,QMainWindow,
                               QLabel,QScrollArea,QPushButton,
                               QHBoxLayout,QGridLayout,QListWidget,
                               QSplitter)
from PySide6.QtGui import QPixmap,QImage

from skimage.segmentation import slic,mark_boundaries
from skimage.io import imread
from PIL import Image

def ndarray2QPixmap(image:np.ndarray):
    return QPixmap(Image.fromarray(image.astype(np.uint8)).toqimage())

class ScrollAreaH(QScrollArea):
    def __init__(self,parent=None):
        super().__init__(parent)
    def wheelEvent(self, event):
        if event.modifiers()==Qt.ShiftModifier:
            delta=-event.angleDelta().y()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()+delta//4)
        else:
            super().wheelEvent(event)

class LabelImg(QLabel):
    def __init__(self,parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton:
            scroll_area=self.parent().parent()
            if isinstance(scroll_area,ScrollAreaH):
                x=event.position().x()
                y=event.position().y()
                location=(x,y)
        super().mousePressEvent(event)
        print(location)
    
class LabelingTools(QWidget):
    def __init__(self):
        super().__init__()
        self.IMAGE_PATH=r'data/image'
        self.LABEL_PATH=r'data/label'
        self.image=[os.path.join(self.IMAGE_PATH,y)for y in os.listdir(self.IMAGE_PATH)]
        self.label=[os.path.join(self.IMAGE_PATH,y)for y in os.listdir(self.LABEL_PATH)]
        self.cur_image=None
        self.cur_label=None
        self.cur=0

        self.initUI()
        self.load_image()

    def update_list_view(self):
        items=os.listdir(self.IMAGE_PATH)
        self.list_view.clear()
        self.list_view.addItems(items)

    def initUI(self):
        self.layout=QGridLayout(self)

        self.splitter_1=QSplitter()
        self.splitter_2=QSplitter()

        self.scro_1,self.scro_2,self.scro_3=ScrollAreaH(),ScrollAreaH(),ScrollAreaH()
        self.label_1,self.label_2,self.label_3=LabelImg(self.scro_1),LabelImg(self.scro_2),LabelImg(self.scro_3)
        self.scro_1.setWidget(self.label_1)
        self.scro_2.setWidget(self.label_2)
        self.scro_3.setWidget(self.label_3)
        
        self.list_view=QListWidget()
        self.update_list_view()

        self.splitter_1.setOrientation(Qt.Orientation.Horizontal)
        self.splitter_2.setOrientation(Qt.Orientation.Vertical)

        self.splitter_2.addWidget(self.scro_1)
        self.splitter_2.addWidget(self.scro_3)

        self.splitter_1.addWidget(self.list_view)
        self.splitter_1.addWidget(self.scro_2)
        self.splitter_1.addWidget(self.splitter_2)

        self.splitter_1.setSizes([1,1,1])
        self.layout.addWidget(self.splitter_1,0,0)

        self.setLayout(self.layout)
        self.resize(800,800)

    def load_image(self):
        all_image_len=len(self.image)
        if all_image_len==0:
            print("No Image", file=sys.stderr)
            return
        cur_image_path=self.image[self.cur%all_image_len]
        self.cur_image=imread(cur_image_path)
        self.cur+=1
        segments=slic(self.cur_image,400,10)
        
        image_with_segments=(mark_boundaries(self.cur_image,segments,color=(1,0,0))*255).astype(np.uint8)
        true_ground_image=np.zeros(shape=self.cur_image.shape)
        if cur_image_path in self.label:
            self.cur_label=np.loadtxt(cur_image_path)
        else:
            _unique_segments=len(np.unique(segments))
            self.label=np.zeros(_unique_segments)
        
        pixmap_1=ndarray2QPixmap(self.cur_image)
        pixmap_2=ndarray2QPixmap(image_with_segments)
        pixmap_3=ndarray2QPixmap(true_ground_image)

        self.label_1.setMinimumSize(pixmap_1.size())
        self.label_2.setMinimumSize(pixmap_2.size())
        self.label_3.setMinimumSize(pixmap_3.size())
        
        self.label_1.setPixmap(pixmap_1)
        self.label_2.setPixmap(pixmap_2)
        self.label_3.setPixmap(pixmap_3)

if __name__=='__main__':
    app=QApplication(sys.argv)
    win=LabelingTools()
    win.show()
    app.exec()