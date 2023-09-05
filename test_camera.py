import cv2
import numpy as np
import os
from keras.models import load_model

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

model = load_model('model.h5')

# Khởi tạo đối tượng VideoCapture để lấy dữ liệu từ camera
cap = cv2.VideoCapture(0)

# Đọc dữ liệu từ camera và hiển thị lên màn hình
while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    
    # nhận dạng khuôn mặt
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # cắt ảnh khuôn mặt
        img = frame[y:y+h, x:x+w]
        
        # vẽ hình chữ nhật
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
        
        # resize ảnh khuôn mặt về kích thước 128x128
        img = cv2.resize(img,(128, 128))

        # chuyển ảnh sang xám
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cân bằng histogram
        img = cv2.equalizeHist(img)
        
        img = img.reshape(1, 128, 128, 1)
        
        img = img / 255
        
        # dự đoán nhãn
        prediction = model.predict(img)
        label = np.argmax(prediction)
        
        if (prediction[0][label] > 0.9):
            # lấy nhãn có xác xuất dự đoán cao nhất
            id=0
            for filename in os.listdir("anh_sau_xu_li"):
                if label == id:
                    name=filename
                    break
                id += 1
            
            cv2.putText(frame, str(name), (x - 10,y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            print(name,"       ",prediction[0][label])
        else:
            cv2.putText(frame, "" , (x - 10,y - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        
        
    # Hiển thị ảnh lên màn hình
    cv2.imshow('img', frame)

    # Nhấn phím 'esc' để thoát khỏi vòng lặp
    if cv2.waitKey(1) == ord("\x1b"):
        break

# Giải phóng tài nguyên và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
