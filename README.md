# Face_Recognition
Developing a model to recognize faces using PCA to decrease dimensions  of images and Mean Square Error to detect faces.
#Principal Component Analysis(PCA)
* The goal: Find another basis of the vector space, which treats variations of data better. 

* Don’t choose the number of components manually. Instead of that, use the option that allows you to set the variance of the input that is supposed to be explained by the generated components
![image](https://user-images.githubusercontent.com/75946833/174461680-d017cbbd-2c76-42ae-b3ed-a00d72dbfe54.png)
* First Function:Convert the RGB image to Grayscale image (Black and white)

![22](https://user-images.githubusercontent.com/75946833/174462002-62a62cbf-b018-4f5d-80db-036187ec8a7e.JPG) ![11](https://user-images.githubusercontent.com/75946833/174462023-5e510e98-2828-43e2-8195-fa47caa38d80.JPG)
* Second Function: After generating model Pass the input image to PCA function to transform it.
* Third Function: Get the MSE between every two images and return the error
* Forth Function: We take 4 images from user One test and 3 for check. First, we will convert the 4 Images to Grayscale, we will resize our images, vectorize each image as a vector(array) then Transform 4 images and check their MSE between the input image and others.
* To secure our camera data If someone was hack the camera to take the images of the employees  



