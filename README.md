 # Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program
### Developed by: Devadhaarini.D
### Register number: 212223230040
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline

# Load images
model = cv2.imread('image_01.png', 0)
withglass = cv2.imread('image_02.png', 0)
group = cv2.imread('image_03.jpeg', 0)

# Display images
plt.imshow(model, cmap='gray')
plt.show()
plt.imshow(withglass, cmap='gray')
plt.show()
plt.imshow(group, cmap='gray')
plt.show()

# Load Haar cascades with full paths
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Check if cascades loaded successfully
if face_cascade.empty():
    print("Error loading face cascade!")
if eye_cascade.empty():
    print("Error loading eye cascade!")

# Face detection function
def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

result = detect_face(withglass)
plt.imshow(result, cmap='gray')
plt.show()

result = detect_face(group)
plt.imshow(result, cmap='gray')
plt.show()

# Adjusted detection
def adj_detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

result = adj_detect_face(group)
plt.imshow(result, cmap='gray')
plt.show()

# Eye detection function
def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

result = detect_eyes(model)
plt.imshow(result, cmap='gray')
plt.show()

result = detect_eyes(withglass)
plt.imshow(result, cmap='gray')
plt.show()

# Live webcam detection
cap = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots()

ret, frame = cap.read()
if ret:
    frame = detect_face(frame)
    im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Video Face Detection')

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame = detect_face(frame)
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.10)
    # Optional: Add a break condition (like pressing 'q') if using OpenCV window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
plt.close()
```
## Output
![Screenshot 2025-05-30 070750](https://github.com/user-attachments/assets/78ded806-8147-4716-91d3-38791b7efc5d)
![Screenshot 2025-05-30 070758](https://github.com/user-attachments/assets/c6109db8-98a1-4aaa-8d3c-09dc87d01ed3)
![Screenshot 2025-05-30 070806](https://github.com/user-attachments/assets/1d54d541-98c9-46f3-88c0-c2295fc22906)
![Screenshot 2025-05-30 070816](https://github.com/user-attachments/assets/35e8a8db-aad3-4fc8-9540-ee652fd463bb)
![Screenshot 2025-05-30 070824](https://github.com/user-attachments/assets/9efc7f52-147e-479b-af06-0baadf8f2cde)
![Screenshot 2025-05-30 070832](https://github.com/user-attachments/assets/9fec8a39-5c41-4dc1-aa09-ee7a73bc70bb)
![Screenshot 2025-05-30 070840](https://github.com/user-attachments/assets/3d0d905e-123f-481d-b3b9-74c283a33850)
![Screenshot 2025-05-30 070849](https://github.com/user-attachments/assets/e10a8f2d-ea3c-4a7b-bf6e-f0b068d1d5c7)
![Screenshot 2025-05-30 070905](https://github.com/user-attachments/assets/65ebf8e1-9330-4e50-be46-4bf89bdc8d8d)

## Result
Thus, to write a Python program using OpenCV to perform image manipulations for the given objectives is executed sucessfully.
