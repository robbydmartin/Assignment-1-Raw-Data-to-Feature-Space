# -*- coding: utf-8 -*-
"""
Created on Thu May 29 08:38:42 2025

@author: rober
"""

import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to display image in color.
def displayColorImage(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

#Function to display the color channels for each image.
def displayColorChannels(image):
    plt.imshow(image[:,:,0])
    plt.axis('off')
    plt.show()

    plt.imshow(image[:,:,1])
    plt.axis('off')
    plt.show()

    plt.imshow(image[:,:,2])
    plt.axis('off')
    plt.show()

    plt.imshow(image[:,:,2],'gray')
    plt.axis('off')
    plt.show()
    
# Function to invert the color channels.
def convertToColorImage(image):
    imageConverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return imageConverted
    
# Function to display a grayscaled image.
def displayGrayscaleImage(image):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.show()

# Function to convert image to grayscale.
def convertToGrayscale(image):
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscaleImage
    
# Function to resize image
def resizeImage(image):
    height, width = image.shape
    width = round((width*(256/height)) / 16) * 16
    height = 256
    imageResized = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    imageResized = cv2.normalize(imageResized.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
    #print(imageResized.shape)
    return imageResized

# Function to create 16x16 block vectors for images
def createBlockFeatureVectors(image, label):
    n = 16
    height, width = image.shape    
    data = []
    file_count = 0
    path = '../Data/%s' % label + '_16x16_block_%i' % file_count + '.csv'
    
    for i in range(0, height, n):
        for j in range(0, width, n):
            block = image[i:i+16, j:j+16]
            flattened = block.flatten()
            data.append(flattened)

    data = np.vstack(data)

    df = pd.DataFrame(data)
    
    if (label == 'cardinal'):
        df[len(flattened) + 1 ] = 0
    elif (label == 'sparrow'):
        df[len(flattened) + 1 ] = 1
    else:
        df[len(flattened) + 1 ] = 2

    while (os.path.isfile(path)):
        file_count += 1
        path = '../Data/%s' % label + '_16x16_block_%i' % file_count + '.csv'
        
    df.to_csv(path, index=False, header=False)
    return df

# Function to create sliding block vectors for images
def createSlidingFeatureVectors(image, label):
    height, width = image.shape
    data = []
    file_count = 0
    path = '../Data/%s' % label + '_sliding_block_%i' % file_count + '.csv'
    
    for i in range(0, height-16):
        for j in range(0, width-16):
            block = image[i:i+16, j:j+16]
            flattened= block.flatten()
            data.append(flattened)

    data = np.vstack(data)

    df = pd.DataFrame(data)
    
    if (label == 'cardinal'):
        df[len(flattened) + 1] = 0
    elif (label == 'sparrow'):
        df[len(flattened) + 1] = 1
    else:
        df[len(flattened) + 1] = 2
        
    while (os.path.isfile(path)):
        file_count += 1
        path = '../Data/%s' % label + '_sliding_block_%i' % file_count + '.csv'

    df.to_csv(path, index=False, header=False)
    return df

# Function to open file and add contents to a vector
def openFile(fileName):
    file = open(fileName)
    reader = pd.read_csv(fileName)
    file.close()
    return reader

#Function to count number of observations
def observationCount(dataSet):
    rows, columns = dataSet.shape
    return rows

#Function to count number of features
def featureCount(dataSet):
    rows, columns = dataSet.shape
    return columns - 1
  
#Function to determine high dimensionality  
def dimensionality(dataSet):
    features = featureCount(dataSet)
    observations = observationCount(dataSet)
    
    if (features > observations):
        return True
    else:
        return False

#Function to calculate and create list of mean values for each feature
def featureMeanDataset(dataSet):
    meanList = []

    for i in range(0, len(dataSet.columns) - 1):
        meanList.append(np.mean(dataSet.iloc[:,i]))
           
    return meanList

#Function to calculate and create list of variance for each feature
def featureVarianceDataset(dataSet):
    varianceList = []
    
    for i in range(0, len(dataSet.columns) - 1):
        varianceList.append(np.var(dataSet.iloc[:, i]))
    
    return varianceList

#Function to calculate and create list of standard deviation for each feature
def featureStandDevDataset(dataSet):
    stdList = []
    
    for i in range(0, len(dataSet.columns) - 1):
        stdList.append(np.std(dataSet.iloc[:, i]))   
        
    return stdList

#Function to merge two feature vectors and shuffle rows
def mergeTwoFeatureVectors16(featureVector1, featureVector2, label):
    completeVector = []

    completeVector.append(featureVector1)
    completeVector.append(featureVector2)

    completeVector = np.vstack(completeVector)
    
    fs = pd.DataFrame(completeVector)
    fs = fs.sample(frac=1).reset_index(drop=True)

    fs.to_csv('../Data/%s' % label + '_16x16_block.csv', index=False, header=False)
    return completeVector
    

#Function to merge three feature vectors and shuffle rows
def mergeThreeFeatureVectors16(featureVector1, featureVector2, featureVector3, label):
    completeVector = []

    completeVector.append(featureVector1)
    completeVector.append(featureVector2)
    completeVector.append(featureVector3)
    
    completeVector = np.vstack(completeVector)
    
    fs = pd.DataFrame(completeVector)
    fs = fs.sample(frac=1).reset_index(drop=True)
    
    fs.to_csv('../Data/%s' % label + '_16x16_block.csv', index=False, header=False)
    return completeVector
    
#Function to merge two feature vectors and shuffle rows
def mergeTwoFeatureVectorsSliding(featureVector1, featureVector2, label):
    completeVector = []

    completeVector.append(featureVector1)
    completeVector.append(featureVector2)

    completeVector = np.vstack(completeVector)
    
    fs = pd.DataFrame(completeVector)
    fs = fs.sample(frac=1).reset_index(drop=True)

    fs.to_csv('../Data/%s' % label + '_sliding_block.csv', index=False, header=False)
    return completeVector
    
# Function to merge three feature vectors and shuffle rows
def mergeThreeFeatureVectorsSliding(featureVector1, featureVector2, featureVector3, label):
    completeVector = []
    
    completeVector.append(featureVector1)
    completeVector.append(featureVector2)
    completeVector.append(featureVector3)
    
    completeVector = np.vstack(completeVector)
    
    fs = pd.DataFrame(completeVector)
    fs = fs.sample(frac=1).reset_index(drop=True)
    
    fs.to_csv('../Data/%s' % label + '_sliding_block.csv', index=False, header=False)
    return completeVector

def main():
    # Open and display original bird images   
    cardinal = cv2.imread("../Images/image0.jpg")
    sparrow = cv2.imread("../Images/image1.jpg")
    robin = cv2.imread("../Images/image2.jpg")

    cardinal = convertToColorImage(cardinal)
    sparrow = convertToColorImage(sparrow)
    robin = convertToColorImage(robin)
    
    displayColorImage(cardinal)
    displayColorImage(sparrow)
    displayColorImage(robin)
    
    # Display the color channels - Cardinal
    displayColorChannels(cardinal)
    
    # Display the color channels - Sparrow
    displayColorChannels(sparrow)
    
    # Display the color channels - Robin
    displayColorChannels(robin)
    
    # Convert to grayscale and display image dimensions
    cardinalG = convertToGrayscale(cardinal)
    sparrowG = convertToGrayscale(sparrow) 
    robinG = convertToGrayscale(robin)
    
    displayGrayscaleImage(cardinalG)
    print("Cardinal grayscale image dimensions: ", cardinalG.shape)
    
    displayGrayscaleImage(sparrowG)
    print("Sparrow grayscale image dimensions: ", sparrowG.shape)
    
    displayGrayscaleImage(robinG)
    print("Robin grayscale image dimensions: ", robinG.shape, "\n")
    
    # Raw data (image) resizing
    cardinalResized = resizeImage(cardinalG)
    print("The resized cardinal grayscale image dimensions are: ", cardinalResized.shape)
    sparrowResized = resizeImage(sparrowG)
    print("The resized sparrow grayscale image dimensions are: ", sparrowResized.shape)
    robinResized = resizeImage(robinG)
    print("The resized robin grayscale image dimensions are: ", robinResized.shape, "\n")
    
    # Display the resized images.
    displayGrayscaleImage(cardinalResized)
    displayGrayscaleImage(sparrowResized)
    displayGrayscaleImage(robinResized)
    
    # Create feature space using 16x16 block function
    cardinal16x16 = createBlockFeatureVectors(cardinalResized, 'cardinal')
    sparrow16x16 = createBlockFeatureVectors(sparrowResized, 'sparrow')
    robin16x16 = createBlockFeatureVectors(robinResized, 'robin')
    
    # Create feature space using sliding block function
    cardinalSlide = createSlidingFeatureVectors(cardinalResized, 'cardinal')
    sparrowSlide = createSlidingFeatureVectors(sparrowResized, 'sparrow')
    robinSlide = createSlidingFeatureVectors(robinResized, 'robin')

################################## STATISTICAL MEAURES AND GRAPHS (16x16 Block)

    cardinalDataSet1 = pd.read_csv('../Data/cardinal_16x16_block_0.csv')
    sparrowDataSet1 = pd.read_csv('../Data/sparrow_16x16_block_0.csv')
    robinDataSet1 = pd.read_csv('../Data/robin_16x16_block_0.csv')

    print("Number of cardinal features (16x16 block): ", featureCount(cardinalDataSet1))
    print("Number of cardinal observations (16x16 block): ", observationCount(cardinalDataSet1))
    print("Number of sparrow features (16x16 Block): ", featureCount(sparrowDataSet1))
    print("Number of sparrow observations (16x16 block): ", observationCount(sparrowDataSet1))
    print("Number of robin features (16x16 Block): ", featureCount(robinDataSet1))
    print("Number of robin observations (16x16 block): ", observationCount(robinDataSet1), "\n")

    cardinalHighDimensional1 = dimensionality(cardinalDataSet1)
    sparrowHighDimensional1 = dimensionality(sparrowDataSet1)
    robinHighDimensional1 = dimensionality(robinDataSet1)

    if (cardinalHighDimensional1):
        print("The cardinal image data set (16x16 block) is high dimensional.")
    else:
        print("The cardinal image data set (16x16 block) is not high dimensional.")
    if (sparrowHighDimensional1):
        print("The sparrow image data set (16x16 block) is high dimensional.")
    else:
        print("The sparrow image data set (16x16 block) is not high dimensional.")
    if (robinHighDimensional1):
        print("The robin image data set (16x16 block) is high dimensional.\n")
    else:
        print("The robin image data set (16x16 block) is not high dimensional.\n")
        
    # Calculate and display mean values of features for each bird
    cardinalMean1 = featureMeanDataset(cardinalDataSet1)
    sparrowMean1 = featureMeanDataset(sparrowDataSet1)
    robinMean1 = featureMeanDataset(robinDataSet1)

    plt.plot(cardinalMean1, label= 'Cardinal')
    plt.plot(sparrowMean1, label= 'Sparrow')
    plt.plot(robinMean1, label= 'Robin')
    plt.xlabel("Feature Numbers")
    plt.ylabel("Mean Values")
    plt.legend()
    plt.show()

    # Calculate and display the variance of the features for each bird
    cardinalVariance1 = featureVarianceDataset(cardinalDataSet1)
    sparrowVariance1 = featureVarianceDataset(sparrowDataSet1)
    robinVariance1 = featureVarianceDataset(robinDataSet1)
    plt.plot(cardinalVariance1, label = 'Cardinal')
    plt.plot(sparrowVariance1, label = 'Sparrow')
    plt.plot(robinVariance1, label = 'Robin')
    plt.xlabel('Feature Numbers')
    plt.ylabel('Variance')
    plt.legend()
    plt.show()

    # Calculate and display standard deviations of features for each bird
    cardinalStd1 = featureStandDevDataset(cardinalDataSet1)
    sparrowStd1 = featureStandDevDataset(sparrowDataSet1)
    robinStd1 = featureStandDevDataset(robinDataSet1)
    plt.plot(cardinalStd1, label = 'Cardinal')
    plt.plot(sparrowStd1, label = 'Sparrow')
    plt.plot(robinStd1, label = 'Robin')
    plt.xlabel("Feature Numbers")
    plt.ylabel("Standard Deviation Values")
    plt.legend()
    plt.show()

################################ STATISTICAL MEAURES AND GRAPHS (Sliding Block)

    cardinalDataSet2 = pd.read_csv('../Data/cardinal_sliding_block_0.csv')
    sparrowDataSet2 = pd.read_csv('../Data/sparrow_sliding_block_0.csv')
    robinDataSet2 = pd.read_csv('../Data/robin_sliding_block_0.csv')

    print("Number of cardinal features (sliding block): ", featureCount(cardinalDataSet2))
    print("Number of cardinal observations (sliding block): ", observationCount(cardinalDataSet2))
    print("Number of sparrow features (sliding block): ", featureCount(sparrowDataSet2))
    print("Number of sparrow observations (sliding block): ", observationCount(sparrowDataSet2))
    print("Number of robin features (sliding block): ", featureCount(robinDataSet2))
    print("Number of robin observations (sliding block): ", observationCount(robinDataSet2), "\n")

    cardinalHighDimensional2 = dimensionality(cardinalDataSet2)
    sparrowHighDimensional2 = dimensionality(sparrowDataSet2)
    robinHighDimensional2 = dimensionality(robinDataSet2)

    if (cardinalHighDimensional2):
        print("The cardinal image data set (sliding block) is high dimensional.")
    else:
        print("The cardinal image data set (sliding block) is not high dimensional.")
    if (sparrowHighDimensional2):
        print("The sparrow image data set (sliding block) is high dimensional.")
    else:
        print("The sparrow image data set (sliding block) is not high dimensional.")
    if (robinHighDimensional2):
        print("The robin image data set (sliding block) is high dimensional.\n")
    else:
        print("The robin image data set (sliding block) is not high dimensional.\n")

    # Calculate and display mean values of features for each bird
    cardinalMean2 = featureMeanDataset(cardinalDataSet2)
    sparrowMean2 = featureMeanDataset(sparrowDataSet2)
    robinMean2 = featureMeanDataset(robinDataSet2)
    plt.plot(cardinalMean2, label = 'Cardinal')
    plt.plot(sparrowMean2, label = 'Sparrow')
    plt.plot(robinMean2, label = 'Robin')
    plt.xlabel("Feature Numbers")
    plt.ylabel("Mean Values")
    plt.legend()
    plt.show()

    # Calculate and display the variance of the features for each bird
    cardinalVariance2 = featureVarianceDataset(cardinalDataSet2)
    sparrowVariance2 = featureVarianceDataset(sparrowDataSet2)
    robinVariance2 = featureVarianceDataset(robinDataSet2)
    plt.plot(cardinalVariance2, label = 'Cardinal')
    plt.plot(sparrowVariance2, label = 'Sparrow')
    plt.plot(robinVariance2, label = 'Robin')
    plt.xlabel('Feature Numbers')
    plt.ylabel('Variance')
    plt.legend()
    plt.show()

    # Calculate and display standard deviations of features for each bird
    cardinalStd2 = featureStandDevDataset(cardinalDataSet2)
    sparrowStd2 = featureStandDevDataset(sparrowDataSet2)
    robinStd2 = featureStandDevDataset(robinDataSet2)
    plt.plot(cardinalStd2, label = 'Cardinal')
    plt.plot(sparrowStd2, label = 'Sparrow')
    plt.plot(robinStd2, label = 'Robin')
    plt.xlabel("Feature Numbers")
    plt.ylabel("Standard Deviation Values")
    plt.legend()
    plt.show()
    
    #Feature Space construction using the 16x16 block data
    cardinalSparrowMerged = mergeTwoFeatureVectors16(cardinalDataSet1, sparrowDataSet1, 'cardinal_sparrow_features')
    cardinalSparrowRobinMerged = mergeThreeFeatureVectors16(cardinalDataSet1, sparrowDataSet1, robinDataSet1, 'cardinal_sparrow_robin_features')
    
    #Select three random features
    a = 7
    b = 11
    c = 112
    
    plt.scatter(cardinalSparrowMerged[:,a], cardinalSparrowMerged[:,b], c=cardinalSparrowMerged[:, 256], s=1)
    plt.xlabel('Feature %i' % a)
    plt.ylabel('Feature %i' % b)
    plt.show()
    
    plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    fig1 = ax.scatter3D(cardinalSparrowRobinMerged[:, a], cardinalSparrowRobinMerged[:, b], cardinalSparrowRobinMerged[:, c], s=5, c=cardinalSparrowRobinMerged[:, 256])
    ax.set_xlabel('Feature %i' % a)
    ax.set_ylabel('Feature %i' % b)
    ax.set_zlabel('Feature %i' % c)
    plt.show()
    
    #Feature space construction using the sliding block data
    cardinalSparrowMerged2 = mergeTwoFeatureVectorsSliding(cardinalDataSet2, sparrowDataSet2, 'cardinal_sparrow_features')
    cardinalSparrowRobinMerged2 = mergeThreeFeatureVectorsSliding(cardinalDataSet2, sparrowDataSet2, robinDataSet2, 'cardinal_sparrow_robin_features')

    plt.scatter(cardinalSparrowMerged2[:,a], cardinalSparrowMerged2[:,b], c=cardinalSparrowMerged2[:, 256], s=1)
    plt.xlabel('Feature %i' % a)
    plt.ylabel('Feature %i' % b)
    plt.show()

    plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    fig1 = ax.scatter3D(cardinalSparrowRobinMerged2[:, a], cardinalSparrowRobinMerged2[:, b], cardinalSparrowRobinMerged2[:, c], s=5, c=cardinalSparrowRobinMerged2[:, 256])
    ax.set_xlabel('Feature %i' % a)
    ax.set_ylabel('Feature %i' % b)
    ax.set_zlabel('Feature %i' % c)
    plt.show()
    
if __name__ == "__main__":
    main()
    
