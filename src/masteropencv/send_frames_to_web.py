# import cv2
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# import uvicorn
# app = FastAPI()

# camera = cv2.VideoCapture(0)


# def generate_frames():

#     while True:

#         success, frame = camera.read()

#         if not success:
#             break

#         # Convert frame to JPEG
#         success, buffer = cv2.imencode(
#             ".jpg",
#             frame
#         )

#         if not success:
#             continue

#         frame = buffer.tobytes()

#         # Send frame as MJPEG
#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n"
#             + frame
#             + b"\r\n"
#         )


# @app.get("/video")
# def video():
#     return StreamingResponse(
#         generate_frames(),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )

# if __name__ == "__main__":
#     uvicorn.run(
#         app,
#         host="127.0.0.1",
#         port=8000
#     )

import os
from ultralytics import solutions, YOLO
import cv2
import shutil

if os.path.exists("results"):
    shutil.rmtree("results")

os.makedirs("results", exist_ok=True)

searcher = solutions.VisualAISearch(
    data="images",
    device="cpu"
)
model = YOLO("yolo26n.pt")

cam = cv2.VideoCapture(0)

def getClassInfo(image):
    res = model(image)
    for cls in res[0].boxes.cls:
        class_id = int(cls)
        class_name = res[0].names[class_id]
        # print("Predicted Class Name : ", class_name)
        return searcher(class_name)
    

def sendFrameToSearch(image):
    classInfo = getClassInfo(image)
    return classInfo

while True:
    success, frame = cam.read()
    if not success:
        break
    cv2.imshow("Video Stream", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        res = sendFrameToSearch(frame)
        print(type(res), res)
        for file_name in res[0:10]:
            shutil.copy(os.path.join("images", file_name),
                        os.path.join("results", file_name))
            
        break
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
