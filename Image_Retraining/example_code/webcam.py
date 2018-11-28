import example_code.label_image_redux as img
import cv2
import os

cap = cv2.VideoCapture(0)
img_name = "temp.png"

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imwrite(img_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print(frame.shape)
    # print(img.label_image(img_name))
    if cv2.waitKey(1) & 0xFF == ord('c'):
        os.system('python label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --output_layer=final_result --image=temp.png --input_layer=Placeholder')
    os.remove(img_name)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()





