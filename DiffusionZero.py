#假面真探4.0
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import cv2
import tensorflow as tf
import numpy as np
from openvino.inference_engine import IECore
from openvino.runtime import Core, Tensor
import time


model_xml = 'model.xml'
model_bin = 'model.bin'

face_detection_xml = 'D:/lz/intel/face-detection-0205/FP32/face-detection-0205.xml'
face_detection_bin = 'D:/lz/intel/face-detection-0205/FP32/face-detection-0205.bin'



ie = IECore()
IE = Core()
exec_net = IE.compile_model(model_xml, device_name='MULTI:CPU,GPU')

face_net = IE.compile_model(face_detection_xml, device_name = 'MULTI:CPU,GPU')


# 进行预测
def predict(image):
    input_data = np.transpose(image, (0,1,2))  # 转换为 CHW 格式
    input_data = np.expand_dims(input_data, axis=0)  # 添加 batch 维度
    # 进行推理
    preds = exec_net([input_data])[exec_net.outputs[0]]
    return preds


def face_detection(image):
    input_data = np.transpose(image, (2,0,1))
    input_data = np.expand_dims(input_data, axis = 0)
    preds = face_net([input_data])[face_net.outputs[0]]
    return preds

def select_file():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png"), 
                                                 ("Video files", "*.mp4; *.avi")])
    count = 0
    if path:
        # 根据文件类型确定是图像还是视频
        if path.endswith(('.jpg', '.jpeg', '.png')):
            # 显示选择的图像
            img = Image.open(path).resize((416, 416))
            img_tk = ImageTk.PhotoImage(img)
            image_label.configure(image=img_tk)
            image_label.image = img_tk  
            img_bgr = np.array(img)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            position = face_detection(img_bgr) [0]
            img_bgr = np.array(img_bgr) / 255.0
            processed_img = cv2.resize(img_bgr[int(position[1]):int(position[3]),int(position[0]):int(position[2])],(224,224))
            # 保持对图片对象的引用，否则图片不会正常显示
            # 进行模型预测
            result = predict(processed_img)
            print(result)
            if result[0][0] >= 0.7:
                result_label.configure(text="FAKE")
            elif result[0][1] >= 0.7:
                result_label.configure(text = "REAL")
        elif path.endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(path)
            # 存储预测为"No"的帧数
            yes_count = 0
            # 存储视频总帧数
            total_frames = 0
            # 逐帧读取视频并进行处理
            infer_request_curr = exec_net.create_infer_request()
            infer_request_next = exec_net.create_infer_request()
            input_node = exec_net.inputs[0]
            output_node = exec_net.outputs[0]
            if cap.isOpened():
                ret, frame_curr = cap.read()
                if ret:
                    #图像前处理
                    h,w,c= frame_curr.shape
                    frame_curr_resize = cv2.resize(frame_curr, (416,416))
                    position = face_detection(frame_curr_resize)[0]
                    left_curr = int(position[0] * w / 416)
                    top_curr = int(position[1] * h /416)
                    right_curr = int(position[2] * w / 416)
                    bottom_curr = int(position[3] * h / 416)
                    processed_frame_curr = cv2.resize(frame_curr,(224,224))
                    processed_frame_curr = np.array(processed_frame_curr)/255.0
                    processed_frame_curr = processed_frame_curr.astype(np.float32) 
                    processed_frame_curr = np.expand_dims(processed_frame_curr, axis=0)
                    blob = Tensor(processed_frame_curr)
                    #推理
                    infer_request_curr.set_tensor(input_node, blob)
                    infer_request_curr.start_async()
                while True:
                    start = time.time()
                    ret, frame_next = cap.read()
                    if ret:
                        h,w,c = frame_next.shape
                        frame_next_resize = cv2.resize(frame_next, (416,416))
                        position = face_detection(frame_next_resize)[0]
                        left_next = int(position[0] * w / 416)
                        top_next = int(position[1] * h /416)
                        right_next = int(position[2] * w / 416)
                        bottom_next = int(position[3] * h / 416)
                        processed_frame_next = cv2.resize(frame_next,(224,224))
                        processed_frame_next = np.array(processed_frame_next)/255.0
                        processed_frame_next = processed_frame_next.astype(np.float32)
                        processed_frame_next = np.expand_dims(processed_frame_next, axis=0)
                        blob = Tensor(processed_frame_next)  
                        infer_request_next.set_tensor(input_node, blob)
                        infer_request_next.start_async()
                        infer_request_curr.wait()
                        infer_result = infer_request_curr.get_tensor(output_node)
                        end = time.time()
                        result = torch.tensor(infer_result.data)
                        print(result)
                        # 在这里根据模型预测结果进行相应操作
                        if result[0][0] >= 0.7:
                            yes_count += 1
                            frame_curr = cv2.rectangle(frame_next,(left_curr, top_curr),(right_curr, bottom_curr),(55,55,255),2)
                            cv2.putText(frame_curr, 'FAKE' , (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (55,55,255), 3)
                            cv2.imshow("video",frame_curr)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        else:
                            frame_curr = cv2.rectangle(frame_next,(left_curr, top_curr),(right_curr, bottom_curr),(55,255,155),2)
                            cv2.putText(frame_curr, 'REAL' , (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (55,255,155), 3)
                            cv2.imshow("video",frame_curr)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                        total_frames += 1
                        infer_request_curr, infer_request_next = infer_request_next, infer_request_curr
                        frame_curr = frame_next
                        processed_frame_curr = processed_frame_next
                        left_curr,top_curr,right_curr,bottom_curr = left_next,top_next,right_next,bottom_next
                        count += 1/(end - start)


                    if not ret:break
            print(count / total_frames)
            cap.release()
            cv2.destroyAllWindows()
            result_label.configure(text= yes_count / total_frames)

# 创建主窗口
root = tk.Tk()
root.title("DiffusionZero")
root.geometry("400x400")

# 创建选择文件按钮
select_button = tk.Button(root, text="选择文件", command=select_file)
select_button.pack(pady=10)

# 显示选择的图像
image_label = tk.Label(root)
image_label.pack()

# 显示模型预测结果
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# 启动主事件循环
root.mainloop()
