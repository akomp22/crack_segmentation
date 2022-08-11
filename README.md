# image_segmentation
unet which detect cracks with varying scale.

---


Basic data consists of images taken during fatigue test of metal specimens that was carried out in National Aviation University (Ukraine). During fatigue test, specimens captured with specified time interval. Result of each specimen test is row of images with gradually growing crack. Images for each specimen was preprocessed so that crack highlighted on basic of dynamical changes on each image row.  


### Examples of initial images and images after preprocess(zoomed to crack region) 
![alt text](https://github.com/akomp22/image_segmentation/blob/master/img/1.PNG)
![alt text](https://github.com/akomp22/image_segmentation/blob/master/img/2.PNG)


After preprocessing image from different specimens test was combined in single data set. For each image, mask was created using matlab code because it has simple function for drawing on image. Using created data set, u-net model was trained. Loss function was modified specifically for current task. As crack area is small relative to image size (1080x768) mask has much more 0 pixels than 1 pixels. Usual loss class weighting could be used but such approach would train model, which ignore small cracks. For that reason, loss weight was specifically for each image example (during training butch size was choose equal to 1). Then, on each training iteration, algorithm compare quantity of 0 and 1 pixels of training mask and adjust loss weights according to class to class ratio. Lower, result of model output from validation image is shown.


### Input image with crack
![alt text](https://github.com/akomp22/image_segmentation/blob/master/img/gh20_2.png)

### Output mask
![alt text](https://github.com/akomp22/image_segmentation/blob/master/img/gh20_0.png)




### Overlay
![alt text](https://github.com/akomp22/image_segmentation/blob/master/img/gh20_3%20(1).png)

