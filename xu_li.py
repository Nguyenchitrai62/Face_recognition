import cv2
import os

# khởi tạo bộ phân loại cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lặp qua các tệp tin trong thư mục
for folder_name in os.listdir("dataset"):
    i=1
    for file_name in os.listdir("dataset/" + folder_name):
        img = cv2.imread("dataset/" + folder_name +'/'+ file_name)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces)==1:
            os.rename("dataset/" + folder_name +'/'+ file_name, "dataset/" + folder_name + f"/anh_{i}.jpg")  # Đổi tên tệp tin
            i+=1
        else:
            os.remove("dataset/" + folder_name +'/'+ file_name)
    
for filename in os.listdir("dataset"):
    for i in range(1,21):
        # load ảnh đầu vào
        img = cv2.imread("dataset/" + filename + f"/anh_{i}.jpg")

        # nhận dạng khuôn mặt
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # cắt ảnh khuôn mặt
            img = img[y:y+h, x:x+w]
            
            # resize ảnh khuôn mặt về kích thước 128x128
            img = cv2.resize(img, (128, 128))
            
            # kiểm tra nếu chưa có folder thì tạo folder mới
            new_folder_path = os.path.join("anh_sau_xu_li", filename)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            
            # lưu ảnh khuôn mặt vào file
            cv2.imwrite("anh_sau_xu_li/"+ filename + f"/anh_{i}.jpg", img)
        
                
print("đã xong")