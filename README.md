# Semantic Segmentation of Defects in Asphalt

## Abstract

Timely repair of defects present on road asphalt is an important task to increase the comfort and safety of road users. In addition, it allows to avoid intense levels of wear on the asphalt, which also result in higher repair costs. The objective of this work is to construct a defect recognizer in digital images of asphalt by means of Deep learning that has an important function of solving image classification problems.  A convolutional neural network (CNN) model is used for its construction. The neural network addressed in this work was U-Net. It enabled IoU 0,9482.

## Contents

[Introduction](#Introduction) 
- 1.1 Problem Definition 
- 1.2 Objective 
- 1.3 Motivation 
- 1.4 Document Structure 

[Context and Technologies](#ContextandTechnologies) 
- 2.1 Neural Networks 
- 2.2 Activation function 
- 2.3 Max pooling 
- 2.4 Repropagation 
- 2.5 Pytorch 
- 2.5 Dataset 
- 2.6 GPU 

[Approach](#Approach) 
- 3.1 Project Steps 
- 3.2 Semantic Segmentation 
- 3.3 U-Net 

Development](#Development) 
- 4.1 Pre-processing of images 
- 4.2 Optimization Adam 
- 4.3 Cross Entropy 
- 4.4 Intersection over union (IoU) 
- 4.5 Save the template 

[Tests](#Tests) 
- 5.1 Analysis of results 

[Conclusions](#Conclusions) 

[Bibliography](#Bibliography)


# Introduction

Computer vision is a field with great potential within Artificial Intelligence (AI). The main purpose of computer vision is the understanding of images by a computer, for example, the way the computer recognizes an object in an image. In computer vision [19], the focus is on the detection of basic geometric structures and object shapes commonly found in digital images. This leads to a study of the basics of image processing and analysis, as well as vector space views and computational geometry of images.

This area of computer vision is connected with the field of Computer Graphics by defining a mathematical image, and by several techniques used in image processing. Image processing consists of modifying the images in such a way that it facilitates the learning of the machine learning model, in English Machine Learning (ML). Modifications in noise reduction, image size format, brightness, color, resulting in image quality. Russell [8] defines this process in three parts:

- Low level: removing noise and highlighting important data;
- Intermediate level: segmentation of images into regions; identify curves, straight lines and measurements;
- High level: compare images against a database.

Szeliski [9] explains about the difficulty of computer vision to highlight an inverse problem, in which we seek to recover some unknowns given insufficient information to specify the solution. This makes us resort to physics-based and probabilistic models to disambiguate between possible solutions.

1.1 Problem definition

Highways play a key role in the transport of goods and people, which in general has a huge impact on economic activities in the country. The poor quality of a road damages the vehicle such as wheels and tires, in the meantime, the money spent to replace the damaged parts is the least of the problems, the big issue is the insecurity that generates in traffic, uncontrolled vehicles, for having their tires burst. or making risky maneuvers trying to avoid holes putting their own lives at risk. Thus, for economic growth and road safety, it is essential to repair them regularly.

1.2 Objective

The objective of this work is to create, train, test and implement an algorithm to be able to perform the recognition of asphalt segmentation, that is, in other words, it is to identify cracks in the asphalt. For this, deep learning (DL) or deep learning in Portuguese will be used. See Figure 1.

Within the field of artificial intelligence there is a sub-area that is ML which is based on the idea that systems learn from data, identify patterns and make decisions with minimal human intervention. On the other hand, DL is an ML subarea capable of learning from a large amount of data without or with human intervention.

1.3 Motivation

I. Gooddellow and Y. Benzio [1] define artificial intelligence as a field that thrives on practical applications and active research topics. The software is seen as a way to automate routine work, understand speech or images, make diagnoses in medicine and support basic scientific research. Artificial intelligence is not only the ability to learn and understand from experience, it is also capable of acquiring and retaining knowledge in models, containing the ability to respond quickly and well to new situations.

Machine learning can be done either supervised or unsupervised. Supervised refers to the fact that it is performed by a supervision of a human, for example, a student and a teacher. No longer supervised, the network learns by itself, without supervision, in which it learns from its own mistakes.

In this work, supervised learning is used for image segmentation, in which the instructor checks if the network is close to a solution. And always being optimized whenever necessary to obtain a better learning rate.

We will present a concept that has been solidifying over the years, which is semantic segmentation (more details in chapter 3) that classifies an image into pixels, in which we start with an image and end with a model.

1.4 Document Structure

The report is composed as follows: Introduction, where the general content for project implementation is being explained; Context, technologies and tools, where the problems encountered, the proposed solution for implementation and the tools to be used will be clearly explained; approach, where it actually describes in depth what was implemented; Development, describes in detail the development sequence of the project; Tests, specifies the tests carried out and providing the necessary justifications; Deployment, which will put the model into production on an android system; Conclusions, makes the last conclusions about the work, explains other possibilities that could be made and says if the work will have any continuity.

# Context and Technologies

