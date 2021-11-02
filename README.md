# Segmented-Images-to-Landscap-Images-Using-GANs
In this project we decided to implement Gans model to generate landscape images from segmented images . We used pix2pix model which is an arcticture to do Image-to-image translation with a conditional GAN .
The main goal of this project was learning how to implement and train GANs model. 

## Dataset
We chose to use **ADE20k-Outdoors dataset**. The ADE20K semantic segmentation dataset contains more than 20K scene-centric images exhaustively annotated with pixel-level objects and object parts labels. There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed.

## Choosing the model
We used the architecture of pix2pix model .
### The Generator 
The architecture of generator is a modified U-Net -the U-Net model is a simple fully convolutional neural network which consists of two parts:
* **Contracting Path:** we apply a series of conv layers and downsampling layers (max-pooling) layers to reduce the spatial size

* **Expanding Path:** we apply a series of upsampling layers to reconstruct the spatial size of the input. The two parts are connected using a concatenation layers among different levels. This allows learning different features at different levels. At the end we have a simple conv 1x1 layer to reduce the number of channels to 1.
![Unet](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/Unet.png)

* Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
* Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
* There are skip connections between the encoder and decoder (as in U-Net).

#### Define the generator loss
GANs learn a loss that adapts to the data, while cGANs learn a structured loss that penalizes a possible structure that differs from the network output and the target image/
The genrator loss per image is a sigmoid cross-entropy loss of the generated images and an array of ones.
in addition we take L1 loss between the generated image and the target image.

The formula to calculate the total generator loss: generator loss = gan_loss + LAMBDA * l1_loss

Where in our case LAMBDA = 100.

#### Training the generator
The training procedure for the generator: the generator get the segmented image as an input , the output of the generator used for both calculate the MAE cost function of the difference between the target image and the generated image. In addition, th output of the generator is also used as the input of the discriminator.

Hence, when aplying gradient decent the gererator is generator being update accoding to both MAE and cross entorpy losses.
![](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/gen_loss.png)

### The Discriminator
The Discriminator is a PatchGAN, meaning after feeding one input image to the network, it gives you the probabilities of two things: either real or fake, but not in scalar output. It used the NxN output vector (in our case N=30).

* Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
* The shape of the output after the last layer is (batch_size, 30, 30, 1)
* Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
* Discriminator receives 2 inputs.
* Input image and the target image, which it should classify as real.
* Input image and the generated image (output of generator), which it should classify as fake.
* Concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))

#### Define the discriminator loss

The discriminator loss function takes 2 inputs: real images, generated images
real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since these are the real images)
generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since these are the fake images)
Then the total_loss is the sum of real_loss and the generated_loss


The training procedure for the discriminator is shown below: the discriminator works in 2 pathes: first path is comapre target images and input images to 1s matrix to check if it is a real images, and second path is comapring the generated images to 0s matrix to check if it is a fake image. Both error defined by cross entropy and a sigmoid to get probabilities.
![](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/disc_loss.png)

## Results
We trained the model on 40 epochs the results we got:

**FID**

![fid](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/fid.png)

we can see our FID gets better and better over time. However the FID is still very high .

**losses per epoch**
![loss_per_epoch](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/loss_per_epoch.png)


We can tell that the genertor indeed trained and the generator loss is maybe noisy but decreasing as we do more steps. In the other hand the discriminator may not be as good as the discriminator which make sense becasue at the beginning - the generaor was not so good so the discriminator was able to predict the image is fake easily. As the generator became better and better the discriminator find it harder to predict wether the image is real or fake.

* The value log(2) = 0.69 is a good reference point for the losses, as it indicates a perplexity of 2: That the discriminator is on average equally uncertain about the two options.
* For the gen_gan_loss a value below 0.69 means the generator is doing better than random at fooling the descriminator. In our case all our value of gen loss is under 0.69.
* For the disc_loss a value below 0.69 means the discriminator is doing better than random, on the combined set of real+generated images. In our case we can tell that it is not balanced- somtimes we get result lower that 0.69 and sometimes not, which shows that as much that the generator becomes better the discriminator is struggeling to predict.
* As training progresses the gen_l1_loss should go down and this is the case so it is good for us.

**Evalutaion using testset**

![](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/results.png)
![](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/results2.png)
![](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/results3.png)
![](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/results4.png)
![](https://github.com/shaniklein/Segmented-Images-to-Landscap-Images-Using-GANs/blob/main/images/results5.png)


The model may not be the best model - it is pretty clear the images we generated is not real images.However, we did succeed in capture the colors of the images such as sky,grass ,trees and ect, we did succeed to capute the details in the images and structures of objects. The images are not just random but startining to look more like a real image. In addition, the model did not get saturated and we can tell by those generated imaged that it indeed seems that with some more training the model will be able to get better and better.
## Conclustions
We leant a lot in this project- we learnt how to build a GAN model from scratch - how to define the tranining function ,how to define each block in the generator and the discriminator using upsamling and downsamnping. We learnt how to evaluate and imrpove our model (which is a whole different way from classification or regression)

Our model may not be the best model there is , to make our model better we could use more training to the model and maybe use more sophisticated augmentation but unfortentlly that took us too much GPU and our GPU limitation on google colab was expired.

Hope you like our projects and learnt from it too !



### References
* ADE20k dataset - https://www.kaggle.com/residentmario/ade20k-outdoors
* How to Implement the Frechet Inception Distance (FID) for Evaluating GANs- https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
* pix2pix architecture - https://arxiv.org/abs/1611.07004
* Image-to-Image Translation with Conditional Adversarial Networks - https://arxiv.org/abs/1611.07004
