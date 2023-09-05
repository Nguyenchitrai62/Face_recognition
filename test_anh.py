import cv2
import numpy as np
import os
from keras.models import load_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = load_model('model.h5')

# Đường dẫn đến ảnh đầu vào
image_path = f"                          "  # nhập đường dẫn ảnh cần test vào đây

# Đọc ảnh từ đường dẫn
img = cv2.imread(image_path)
# nhận dạng khuôn mặt

faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

if len(faces)==0:
    print("không phát hiện thấy khuôn mặt")
else:
    for (x, y, w, h) in faces:
        # cắt ảnh khuôn mặt
        img = img[y:y+h, x:x+w]
        
        # resize ảnh khuôn mặt về kích thước 128x128
        img = cv2.resize(img,(128, 128))
        
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.equalizeHist(img)

        
        img = img.reshape(1, 128, 128, 1)
        
        # chuẩn hóa về đoạn 0 1 tăng độ chính xác   
        img = img / 255

        # dự đoán nhãn
        prediction = model.predict(img)
        
        # lấy nhãn có xác xuất dự đoán cao nhất
        label = np.argmax(prediction)
        id=0
        for filename in os.listdir("dataset"):
            if label == id:
                name=filename
                break
            id += 1
        

        print(name,"tỉ lệ chính xác",prediction[0][label])
                

