from imageai.Detection import ObjectDetection
import cv2
from classifier import classifier

detector = ObjectDetection()
model_path = "yolo-tiny.h5"
input_path = "input.png"
output_path = "output.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)
image = cv2.imread(input_path)
cnt = 0
for eachItem in detection:
    cnt+=1
    print(eachItem["name"] , " : ", eachItem["percentage_probability"],eachItem)
    x,y,w,h = eachItem['box_points']
    label = classifier.predict(image[y:h,x:w])
    img = cv2.rectangle(image,(x,y),(w,h),(0,0,255),2)
    cv2.putText(image,'{0}'.format(label),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
cv2.putText(image,'No of bottles got detected {0}'.format(cnt),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
cv2.imwrite('{0}'.format(output_path),image)
cv2.imshow('Output',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
