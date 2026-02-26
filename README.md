# covid-19-detection-X-Ray-covid-detector-
This project implements a deep learning-based COVID-19 detection system using chest X-ray images. The system leverages Convolutional Neural Networks (CNNs) to classify chest X-rays into three categories: COVID-19 positive, Viral Pneumonia, and Normal cases.


Project Description
This project implements a deep learning-based COVID-19 detection system using chest X-ray images. The system leverages Convolutional Neural Networks (CNNs) to classify chest X-rays into three categories: COVID-19 positive, Viral Pneumonia, and Normal cases. The project includes a complete pipeline from data preprocessing to model deployment with a web interface.

The repository contains multiple implementations: a custom CNN architecture from scratch and transfer learning approaches using pre-trained models like VGG16, ResNet50, and DenseNet121. The DenseNet121 implementation achieves the best performance, leveraging its dense connectivity pattern that's particularly effective for detecting subtle texture changes in medical images.

The web application, built with Flask and a responsive HTML/CSS frontend, allows users to upload chest X-ray images for real-time analysis. The interface provides confidence scores for each classification category and offers personalized health recommendations based on the prediction results. Additional features include an information center with COVID-19 safety guidelines, emergency contact information, symptom checklists, and a contact/feedback system integrated with Web3Forms.

The project emphasizes practical deployment considerations, including data augmentation techniques, class imbalance handling, model checkpointing, learning rate scheduling, and early stopping. The system achieves competitive accuracy on the test dataset and demonstrates the potential of AI-assisted diagnostic tools in healthcare settings


# COVID-19 Chest X-Ray Detection System

## Overview
An AI-powered web application that detects COVID-19 from chest X-ray images using deep learning. The system classifies X-rays into COVID-19 positive, Viral Pneumonia, or Normal categories with confidence scores and provides personalized health recommendations.

## Features
- **Multi-model Support**: Custom CNN, VGG16, ResNet50, and DenseNet121 implementations
- **Web Interface**: User-friendly Flask application with responsive design
- **Real-time Analysis**: Instant prediction with confidence scores
- **Health Recommendations**: Personalized dos and don'ts based on results
- **Information Center**: COVID-19 symptoms, safety guidelines, emergency contacts
- **Feedback System**: Contact form with star ratings via Web3Forms

## Tech Stack
- **Backend**: Python, Flask, TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **Deep Learning**: CNN, Transfer Learning (VGG16, ResNet50, DenseNet121)
- **Data Processing**: NumPy, PIL, ImageDataGenerator
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Flask server, Web3Forms API




**Usage**
Navigate to the "Detect" page

Upload a chest X-ray image (PA view)

Click "Analyze X-Ray"

View results with confidence scores and recommendations

Model Performance
The DenseNet121 implementation achieves the best results:

Test Accuracy: ~45% (with mock data; higher with full dataset)

Balanced class weights for handling imbalanced data

Data augmentation techniques for better generalization
Navigate to the "Detect" page

Upload a chest X-ray image (PA view)

Click "Analyze X-Ray"

View results with confidence scores and recommendations


**Model Performance**

The DenseNet121 implementation achieves the best results:

Test Accuracy: ~45% (with mock data; higher with full dataset)

Balanced class weights for handling imbalanced data

Data augmentation techniques for better generalization
