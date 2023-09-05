import cv2
import numpy as np
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


# Kiểm tra sự hiện diện của GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load data
data = []
label = []

# id để đếm số lượng nhãn
id = 0
for filename in os.listdir("anh_sau_xu_li"):
    for i in range(1, 21):
        img = cv2.imread("anh_sau_xu_li/" + filename + f"/anh_{i}.jpg",0)
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (128, 128))
        data.append(img)
        label.append(id)
    id += 1

data = np.array(data)
label = np.array(label)

# Xử lý ảnh
data = data.reshape(id * 20, 128, 128, 1)
data = data / 255
label = to_categorical(label)

# Khởi tạo biến tăng cường dữ liệu
datagen = ImageDataGenerator(
    rotation_range=10,  # Xoay ảnh trong khoảng -10 đến +10 độ
    # width_shift_range=0.1,  # Dịch chuyển chiều ngang trong khoảng -10% đến +10% của chiều rộng ảnh
    # height_shift_range=0.1,  # Dịch chuyển chiều dọc trong khoảng -10% đến +10% của chiều cao ảnh
    shear_range=0.1,  # Làm méo dạng ảnh trong khoảng -10 đến +10 độ
    # zoom_range=0.1,  # Phóng to/thu nhỏ ảnh trong khoảng từ 0.8 đến 1.2 lần
    horizontal_flip=True,  # Lật ngang ảnh
    fill_mode='nearest'  # Điền các pixel bị thiếu do biến đổi bằng các pixel lân cận gần nhất
)


# Tạo mô hình

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

# model.add(Dropout(0.5))

model.add(Dense(id, activation='softmax'))

model.summary()

# Compile mô hình
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Sử dụng generator trong quá trình huấn luyện
model.fit(datagen.flow(data, label, batch_size=32), epochs=30)

# Lưu mô hình
model.save('model.h5')
