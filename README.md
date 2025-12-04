# Car Cras hDetector
**Overview**
Road accidents are a leading cause of injury and death worldwide, yet emergency responses are often delayed due to the absence of immediate detection and reporting systems. In many cases, surveillance or dashcam footage captures accidents, but no automated mechanism exists to recognize and alert responders in real time. This delay can cost lives, particularly in areas of low traffic or visibility. Developing an automated accident detection system can drastically improve response times and reduce fatalities by enabling instant alerts to authorities or nearby hospitals. This project leverages computer vision to identify accidents directly from images or video footage. By creating a reliable vision-based classifier, we can contribute to safer roads and more efficient emergency response infrastructure, especially in developing or rural regions lacking advanced detection systems.

**Dataset**
We will be using a publicly available dataset from Kaggle called “Car Crash or Collision Prediction Dataset”. It includes 10,000 images of dashcams with a lot of them involving crashes and the rest being normal dashcam images. It also contains an Excel file which tells us which image had a collision and which did not.

Link: https://www.kaggle.com/datasets/mdfahimbinamin/car-crash-or-collision-prediction-dataset

**Method**
1. Upon downloading the dataset, we have the 10,000 images which have a mixture of crashes and no crashes and a .xlsx file which tell us which of the images are crashes and which of them are not.
2. I firstly created split.py to split the paths of my dataset into which images are crashes and which of them are not so that I can use PyTorch's ImageLoader function to load my training and testing data
3. In the CNN.ipynb file, I start of by resizing all the images and centering them to help the model learn well.
4. After loading the training and testing datasets, I created my own CNN model with 4 convolutional layers, each followed by batch normalization, and ending with 2 linear layers into one binary output of a crash or not.
5. I then used Binary Cross Entropy for my loss function and an optimizer with learning rate of 0.001
6. After this, I trained my model across 20 epochs and tested the accuracies per epoch and the loss per epoch
7. On creating the grpahs for each, I was able to see my model was not overfitting and was acheiving a test accuracy of around 82%
8. I printed the Confusion Matrix, Precision, Recall, F1 score and the ROC-AUC curve and saw that the model performed relatively well across the testing set with a great recall score.

**Future Applications**
With this model and integrating it into dashcams in cars, we will be able to provide a extra layer of safety for drivers where if they ever get into a crash, the model will know and be able to inform local authorities. In the future, I can look to try integrating this model into an actual dashcam and maybe checking for a crash every 10-15 seconds for quick response times going forward.
