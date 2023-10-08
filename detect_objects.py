import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt 

# Load the pre-trained model
model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_1024x1024/1")
#Image size
input_size = (1024, 1024)
#Threshold for detection confidence
score_threshold = 0.5
# Define label map(models/research/object_detection/data/mscoco_label_map.pbtxt)
label_map = {17: 'cat', 18: 'dog'}

image_np = cv2.imread('image.jpg')

# Convert the input image to a tensor
image_tensor = tf.convert_to_tensor(image_np)

# Add a batch dimension to the tensor
image_tensor = tf.expand_dims(image_tensor, axis=0)
detections = model(image_tensor)

cat_dog_scores = []
cat_dog_boxes = []
for i in range(len(detections['detection_scores'][0])):
    if detections['detection_classes'][0][i] in [17, 18]:
        if detections['detection_scores'][0][i] > score_threshold:
            cat_dog_scores.append(detections['detection_scores'][0][i])
            cat_dog_boxes.append(detections['detection_boxes'][0][i])

for i in range(len(cat_dog_boxes)):
    ymin, xmin, ymax, xmax = cat_dog_boxes[i]
    im_height, im_width, _ = image_np.shape
    left, right, top, bottom = xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height
    cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=2)
    
    # Add text to the rectangle
    label = 'Cat' if detections['detection_classes'][0][i] == 17 else 'Dog'
    confidence = detections['detection_scores'][0][i]
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image_np, text, (int(left), int(top) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Display the output image
cv2.imwrite('output_image.jpg', image_np)
plt.imshow(image_np)
plt.show()