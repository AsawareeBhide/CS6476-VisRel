import torchvision
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import SVOs

COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_prediction(img_path, threshold):
    img = Image.open(img_path) # Load the image
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class

def object_detection_api(img_path, sub, obj, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

    sub_list=[]
    obj_list=[]

    for i in range(len(boxes)):
        if(pred_cls[i]==sub):
            sub_list.append(i)
        if(pred_cls[i]==obj):
            obj_list.append(i)

    for i1 in range(len(sub_list)):
        for j1 in range(len(obj_list)):
            if(sub_list[i1]==obj_list[j1]):
                continue
            plt.clf()
            img1 = np.array(img)
            i = sub_list[i1]
            cv2.rectangle(img1, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
            cv2.putText(img1,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class

            i = obj_list[j1]
            cv2.rectangle(img1, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
            cv2.putText(img1,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
            plt.imshow(img1)
            plt.xticks([])
            plt.yticks([])
            plt.show()

def show_all_pairs(img_path, text):
    tokens = nlp(text)
    # tokens = nlp("the pizza is being eaten by her")
    svos = findSVOs(tokens)
    # print(svos)
    subject=str(tokens[0])
    obj=str(tokens[2])
    print(tokens)
    print(subject)
    print(obj)

#object_detection_api(img_path, subject, obj, rect_th=15, text_th=7, text_size=5, threshold=0.8)
subject = "car"
obj = "handbag"
object_detection_api('./girl_cars.jpg', subject, obj, rect_th=15, text_th=7, text_size=5, threshold=0.8)
show_all_pairs('./girl_cars.jpg',"handbag near car")
