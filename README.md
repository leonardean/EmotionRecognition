EmotionRecognition
==================

Real time emotion recogniser using web camera based on FACS.
##Abstract
In the past decades, the technology of computer vision has grown rapidly so that tracking objects, which has strong potential to contribute to human computer interaction, has become achievable.

Inspired by “Telling Lies” written by Paul Ekman, a final goal of being able to automatically recognize different emotions with the technology of computer vision rose for my individual project. Emotions have been playing an important role in human communications, and emotion recognition could be potentially very widely used in areas such as entertainment, human computer interaction, surveillance and even lie detection if emotions could be recognized precisely enough. The problems involved include face tracking, feature extraction, and emotion classification.

There are existing researches into this area. In the domain of psychology, how to determine an emotion based on the combination of muscle movements (facial expression) has been published and scientifically approved. In the domain of computer science, people have managed to classify different emotions with various accuracies in ways of machine learning, Bayesian method etc. However, those existing approaches have one or more limitations such as requiring fixed face position, requiring continuous frames, and need of pre-processing.

The thesis describes the related researches, experiments of different approaches and how a working implementation was constructed to recognize 7 fundamental emotions (neutral, happy, sad, disgust, fear, surprise, angry) in real time using a web camera.

##Algorithm Keywords (why not overview? because I'm lazy)
* AAM (Active Appearance Model) for face tracking
  * Action Units
  * Facial Characteristic Points
* SVM (Support Vector Machine Based on FACS)
  * Vector building
  * Model building
  * Classification

##Post
![post](http://www.mftp.info/20140202/1392036054x1927127332.png)
