# Visual Tracking Using UAVS
Advancements in computer processing power and programming languages have opened a
new area of using computer vision within autonomous vehicles. Computer vision techniques
are designed to operate using visible light cameras to perform specific task much like humans do. Examples of current uses of computer vision include optical character recognition, object tracking and face detection. With improvements in machine learning techniques, computer vision task has become easier to achieve with higher accuracy rates.

Object tracking is very beneficial for unmanned aerial vehicles (UAVs) in military, policing
and wildlife. For example, in a police chase a helicopter may be deployed to track the
escaping vehicle. Currently it is very expensive to purchase and maintain helicopters.
Another cost is training a pilot which is a high skilled profession. UAVs can be manufactured to be much smaller in size than a helicopter as they no longer need to contain space for a pilot. UAVs can greatly reduce these costs as they are cheaper to produce can be more easily mass produced and don’t require a pilot for them to operate.

## Single Shot MultiBox Detector
Single Shot MultiBox Detector (SSD) is a method of detecting objects within images using a
single neural network. Released in 2016 it provided significant performance increase over
Faster R-CNN and accuracy improvements over You Only Look Once (YOLO) [1]. SSD uses a single neural net for predicting object classes and bounding boxes, compared to Faster R-CCN that used two. This increases the overall speed of SSD. Improvements in accuracy are achieved by making predictions at each feature map within the CNN layers. This is demonstrated in the architecture of SSD found in Figure 1.

![Figure 1 - Comparison of SSD and YOLO Architectures](https://i.imgur.com/B2P200z.jpg)
**Figure 1 - Comparison of SSD and YOLO Architectures [11]**

The largest difference between Faster R-CNN and SSD is with the region of interest (ROI) proposal. SSD performs ROI on the feature map of the whole image unlike Faster R-CNN which perform ROI before the convolutional layers with the region proposal network. For generating ROIs from the last feature map of the convolution layers, SSD use a similar concept to anchors as found in Faster R-CNN. For each pixel in the feature map SSD generates boxes of different ratios. Demonstrated in Figure 2.

![Figure 2 - SSD ROI Generation](https://imgur.com/TfQVPb1.jpg)
**Figure 2 - SSD ROI Generation**

## Training
Training using a pretrianed model was decied as training off a pre-trained model has
benefits over starting from scratch such as much quicker training times and less data needed
for the training. ssd_mobilenet_v1_coco was selected for the pretrained model due to its
good performance. 

Before labelling the collected images all images were resized to 640x360 and then labeled using LabelImg [2]. LabelImg exports the bounding boxes for each image into an XML file. However, TensorFlow requires a TFRecord file to train with. Generating a TFRecord requires the bounding boxes for each image to be available in a singular CSV file. To overcome this issue a short Python script was obtained that converts the XML files to a single CSV file [3]. To speed up the conversation process the script was slightly modified. A for loop was added that scans though all the files and converts each XML file. This avoids having to type in each individual file name that needs to be converted, saving a lot of time as there are over 250 files to be processed.

For this project line 31 was changed from if row_label == 'raccoon': to if row_label == 'car':. This change ensures the cars within the dataset are correctly labelled. 
Next, the pretrained model was downloaded to obtain the checkpoint and configuration files. The ssd_mobilenet_v1_coco model is available from TensorFlow’s GitHub [4]. The
configuration file (ssd_mobilenet_v1_coco.config) required a few changes before training.
These changes include directory and file locations. 

The final file required for training is mscoco_label_map.pbtxt this file contains the label data
for the objects are being trained. ssd_mobilenet_v1_coco initially has 90 objects in its
classifier. Once the training has been run it will only feature one object in its classifier, car.
This is ideal for this project as only cars need to be detected.

With all the files and datasets for training setup correctly the training process was started.
The training process was attempted multiple times but was only successfully two times. Training was ran on a Nvidia P4000 on Paperspace [5].

Paperspace is a cloud computing service that provides access to computers that are equippedwith high performance GPUs. All the image data and configuration files were uploaded to the machine for training. Paperspace is a pay per hours service. For a Nvidia P4000 the rate is $0.60/hr. To fund the cost of using Paperspace a promotion code was found on a development website that provided $10 of free credit. This is enough to cover the cost of training this model. Training on the Paperspace machine ran for 3hr 33min and 20760 steps. It could have been run for a shorter time as the TotalLoss drop below 1 at 2hr 44min, as show in Figure 3. The Paperspace machine provides much greater steps per second (1.62 steps/s) than the GTX 1080 Ti (0.89 step/s) even though the Quadro P4000 is a weaker card. This is due to the larger amount of VRAM on the Quadro P4000 (16GB).

![Figure 3 - TensorFlow Training TotalLoss Graph](https://imgur.com/0cFVn8S.jpg)
**Figure 3 - TensorFlow Training TotalLoss Graph**

## Results
This video show a comparion between ssd_mobilenet_v1_coco provided by TensorFlow and the custom trained model.
[![SSD Model Demo](https://img.youtube.com/vi/75i9xqIBunI/0.jpg)](https://www.youtube.com/watch?v=75i9xqIBunI)

The final custom model provided very good results when it was tested. Figure 29 shows the
comparison between the two models on the same test video. ssd_mobilenet_v1_coco is on
the left and the custom model is on the right.
![Figure 4 - Comparison Between ssd_mobilenet_v1_coco and custom model](https://imgur.com/35IddNu.jpg)
![Figure 4 - Comparison Between ssd_mobilenet_v1_coco and custom model](https://imgur.com/PXLyjzg.jpg)
![Figure 4 - Comparison Between ssd_mobilenet_v1_coco and custom model](https://imgur.com/2gNQi0E.jpg)
**Figure 4 - Comparison Between ssd_mobilenet_v1_coco and custom model**

As seen in Figure 4 the trained model offers significant improvement in areas where
ssd_mobilenet_v1_coco failed to detect the car. The model is much better than expected as
only a small dataset was used to train the model. Even with this improvement the model still
isn’t perfect. It was only able to detect the car in 92% of the frames, see Figure 5.
|Model|Average FPS|Percentage of Average Frames Detected|Confidence per Detected Frame|
|--|--|--|--|
|Custom Trained Model|13.5|92.0|0.949|
|ssd_mobilenet_v1_coco|12.9|18.5|0.562|
**Figure 5 - Result Data Comparing Custom Model and ssd_mobilenet_v1_coco**

The area which the custom model lacked accuracy was directly above the vehicle. This
uncertainty is due to only having 6 images out of the 232 that feature the roof of cars. The
issue could be rectified by simply adding more top view images. Lighting condition also
affect the results. When the car is driving through a shaded area the model doesn’t always
detect the vehicle as shown in Figure 6. Due to time constraints with the project it was
decided to not spend more time developing the model as the time it would to take to achieve a small improvement wasn’t necessary and the training already accomplish the desired result.
![Figure 6 - Lighting Conditions Example](https://imgur.com/64AccrK.jpg)
**Figure 6 - Lighting Conditions Example**

# Future Work
The model can be further improved. Collecting more image for the training dataset may yield even better results. Also, relabelling the existing bounding boxes may improve the result as some of them were roughly done to speed up the labelling process, they may be too large around the object. It’s worth looking in YOLOv3 Tiny as this CNN is design to be extremely fast. 

# References
[1] SSD: Single Shot MultiBox Detector [https://arxiv.org/abs/1512.02325](https://arxiv.org/abs/1512.02325)
[2] tzutalin/LabelImg [https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)
[3]  datitran/Raccoon Detector Dataset [https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py](https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py)
[4] tenforflow/Tensorflow detection model zoo [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
[5] Paperspace [https://www.paperspace.com/](https://www.paperspace.com/)


