import cv2
import os
import tkinter as tk
import numpy as np
from tkinter import messagebox
import shutil

def submit_text():
    global entry_text
    entry_text = entry.get()
    root.destroy()

root = tk.Tk()
root.geometry("400x200+600+250")

label = tk.Label(root, text="Nhập tên", font = ("Calibri",24,'bold'))
label.pack()
label.place(x = 130, y = 0)

entry = tk.Entry(root, font=26)
entry.pack()
entry.place(x = 80, y = 70, width=250, height=30)

button = tk.Button(root, text="Xác nhận", command=submit_text)
button.pack()
button.place(x =140 ,y = 120 , width=100, height=40)

root.mainloop()

# kiểm tra nếu chưa có folder thì tạo folder mới
new_folder_path = os.path.join("anh_sau_xu_li", entry_text)
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

i=10000

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# Khởi tạo đối tượng VideoCapture để lấy dữ liệu từ camera
cap = cv2.VideoCapture(0)

# Đọc dữ liệu từ camera và hiển thị lên màn hình
while True:
    if cv2.waitKey(1) == ord("b"):
        i=1
        
    # Đọc khung hình từ camera
    ret, frame = cap.read()

    if i == 10000:
        cv2.putText(frame, "press 'b' to start" , (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # nhận dạng khuôn mặt
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    
    # vẽ hình chữ nhật xung quanh khuôn mặt
    for (x, y, w, h) in faces:
                
        # cắt ảnh khuôn mặt
        img = frame[y:y+h, x:x+w]
        
        # resize ảnh khuôn mặt về kích thước 128x128
        img = cv2.resize(img, (128, 128))

        # đo chất lượng của ảnh
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        chat_luong = np.var(laplacian)
        
        # nếu đủ chất lượng thì lưu trong anh_sau_xu_li
        if len(faces) == 1 and  i < 401 and i % 10 == 0:
            if chat_luong > 800:
                # lưu ảnh khuôn mặt vào file
                cv2.imwrite(f"anh_sau_xu_li/" + entry_text + f"/anh_{int( i / 10 )}.jpg", img)
                i += 1
        
        if i % 10 != 0 and i < 401:
            i += 1
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
      
    # Hiển thị ảnh lên màn hình
    text="processing " + str(int( (i-1) / 10 )) +"/40"
    if i <= 401:
        cv2.putText(frame, text , (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('img', frame)

    if i == 401:
        messagebox.showinfo("thông báo","Đã lấy ảnh " + entry_text + " thành công" )    
        break
    

    # Nhấn phím 'esc' để thoát khỏi vòng lặp
    if cv2.waitKey(1) == ord("\x1b"):
        break
    
if i != 401:
    shutil.rmtree("anh_sau_xu_li/" + entry_text)

# Giải phóng tài nguyên và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()




