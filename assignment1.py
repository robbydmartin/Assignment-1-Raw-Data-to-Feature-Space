# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:45:58 2024

@author: Robert Martin
"""

import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import numpy as np


#Function to resize image
def resizeImage(image):
    height, width = image.shape
    width = round((width*(256/height)) / 16) * 16
    height = 256
    imageResized = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    imageResized = cv2.normalize(imageResized.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
    #print(imageResized.shape)
    return imageResized

#Function to create 16x16 block vectors
def createBlockFeatureVectors(imageG):
    nn = 16
    height, width = imageG.shape
    data = []
    
    for ii in range(0, height, nn):
        for jj in range(0, width, nn):
            block = imageG[ii:ii+16, jj:jj+16]
            flattened = block.flatten()
            data.append(flattened)

    data = np.vstack(data)

    df1 = pd.DataFrame(data)
    df1[len(flattened) + 1 ] = 0

    df1.to_csv('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out1.csv', index=False)
   
#Function to open file and add contents to a vector
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
    
    for ii in range(0, len(dataSet.columns) - 1):
        meanList.append(np.mean(dataSet.iloc[:,ii]))
        
    return meanList

#Function to calculate and create list of variance for each feature
def featureVarianceDataset(dataSet):
    varianceList = []
    
    for ii in range(0, len(dataSet.columns) - 1):
        varianceList.append(np.var(dataSet.iloc[:, ii]))
    
    return varianceList

#Function to calculate and create list of standard deviation for each feature
def featureStandDevDataset(dataSet):
    stdList = []
    
    for ii in range(0, len(dataSet.columns) - 1):
        stdList.append(np.std(dataSet.iloc[:, ii]))   
        
    return stdList
      
cardinal = cv2.imread("Images/image0.jpg")
sparrow = cv2.imread("Images/image1.jpg")
robin = cv2.imread("Images/image2.jpg")

cardinal = cv2.cvtColor(cardinal, cv2.COLOR_BGR2RGB)
sparrow = cv2.cvtColor(sparrow, cv2.COLOR_BGR2RGB)
robin = cv2.cvtColor(robin, cv2.COLOR_BGR2RGB)

plt.imshow(cardinal)
plt.axis('off')
plt.show()

plt.imshow(sparrow)
plt.axis('off')
plt.show()

plt.imshow(robin)
plt.axis('off')
plt.show()

# Display the color channels - Cardinal
plt.imshow(cardinal[:,:,0])
plt.axis('off')
plt.show()

plt.imshow(cardinal[:,:,1])
plt.axis('off')
plt.show()

plt.imshow(cardinal[:,:,2])
plt.axis('off')
plt.show()

plt.imshow(cardinal[:,:,2],'gray')
plt.axis('off')
plt.show()

#Display the color channels - Sparrow
plt.imshow(sparrow[:,:,0])
plt.axis('off')
plt.show()

plt.imshow(sparrow[:,:,1])
plt.axis('off')
plt.show()

plt.imshow(sparrow[:,:,2])
plt.axis('off')
plt.show()

plt.imshow(sparrow[:,:,2],'gray')
plt.axis('off')
plt.show()

#Display the color channels - Robin
plt.imshow(robin[:,:,0])
plt.axis('off')
plt.show()

plt.imshow(robin[:,:,1])
plt.axis('off')
plt.show()

plt.imshow(robin[:,:,2])
plt.axis('off')
plt.show()

plt.imshow(robin[:,:,2],'gray')
plt.axis('off')
plt.show()

# Convert to grayscale and display image dimensions
cardinalG = cv2.cvtColor(cardinal, cv2.COLOR_BGR2GRAY) 
sparrowG = cv2.cvtColor(sparrow, cv2.COLOR_BGR2GRAY) 
robinG = cv2.cvtColor(robin, cv2.COLOR_BGR2GRAY)

plt.imshow(cardinalG, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()
print("Cardinal grayscale image dimensions: ", cardinalG.shape)

plt.imshow(sparrowG, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()
print("Sparrow grayscale image dimensions: ", sparrowG.shape)

plt.imshow(robinG, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()
print("Robin grayscale image dimensions: ", robinG.shape)

# Raw data (image) resizing
cardinalResized = resizeImage(cardinalG)
sparrowResized = resizeImage(sparrowG)
robinResized = resizeImage(robinG)

# Plot the images
plt.imshow(cardinalResized, cmap=plt.get_cmap('gray'))
#plt.axis('off')
plt.show()

plt.imshow(sparrowResized, cmap=plt.get_cmap('gray'))
#plt.axix('off')
plt.show()

plt.imshow(robinResized, cmap=plt.get_cmap('gray'))
#plt.axis('off')
plt.show()

############################################################## CARDINAL 16X16

mm1 = []
for ii in range(0, 256, 16):
    for jj in range(0, 144, 16):
        blk1 = cardinalG[ii:ii+16,jj:jj+16]
        flt1 = blk1.flatten()
        mm1.append(flt1)

mm1 = np.vstack(mm1)

df1 = pd.DataFrame(mm1)
df1[len(flt1)] = 0   # label 0 for cardinal

df1.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out1.csv", index=False, header=False)

############################################################## SPARROW 16X16

mm2 = []
for ii in range(0, 256, 16):
    for jj in range(0, 144, 16):
        blk2 = sparrowG[ii:ii+16,jj:jj+16]
        flt2 = blk2.flatten()
        mm2.append(flt2)

mm2 = np.vstack(mm2)

df2 = pd.DataFrame(mm2)
df2[len(flt2)] = 1   # label 1 for sparrow

df2.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out2.csv", index=False, header=False)

############################################################## ROBIN 16X16

mm3 = []
for ii in range(0, 256, 16):
    for jj in range(0, 336, 16):
        blk3 = robinG[ii:ii+16,jj:jj+16]
        flt3 = blk3.flatten()
        mm3.append(flt3)

mm3 = np.vstack(mm3)

df3 = pd.DataFrame(mm3)
df3[len(flt3)] = 2   # label 2 for robin

df3.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out3.csv", index=False, header=False)

############################################################## CARDINAL

mm4 = []
for ii in range(0, 256):
    for jj in range(0, 144):
        blk4 = cardinalG[ii:ii+16,jj:jj+16]
        flt4 = blk4.flatten()
        mm4.append(flt4)

mm4 = np.vstack(mm4)

df4 = pd.DataFrame(mm4)
df4[len(flt4)] = 0   # label 0 for cardinal

df4.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out4.csv", index=False, header=False)

############################################################## SPARROW

mm5 = []
for ii in range(0, 256):
    for jj in range(0, 144):
        blk5 = sparrowG[ii:ii+16,jj:jj+16]
        flt5 = blk5.flatten()
        mm5.append(flt5)

mm5 = np.vstack(mm5)

df5 = pd.DataFrame(mm5)
df5[len(flt5)] = 1   # label 1 for sparrow

df5.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out5.csv", index=False, header=False)

############################################################## ROBIN

mm6 = []
for ii in range(0, 256):
    for jj in range(0, 336):
        blk6 = robinG[ii:ii+16,jj:jj+16]
        flt6 = blk6.flatten()
        mm6.append(flt6)

mm6 = np.vstack(mm6)

df6 = pd.DataFrame(mm6)
df6[len(flt6)] = 2   # label 2 for robin

df6.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out6.csv", index=False, header=False)

cardinalDataSet = openFile('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out1.csv')
sparrowDataSet = openFile('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out2.csv')
robinDataSet = openFile('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 1/Data/out3.csv')

observationCount(cardinalDataSet)
featureCount(cardinalDataSet)

cardinalHighDimensional = dimensionality(cardinalDataSet)
sparrowHighDimensional = dimensionality(sparrowDataSet)
robinHighDimensional = dimensionality(robinDataSet)

if (cardinalHighDimensional):
    print("The cardinal image data set is high dimensional.")
else:
    print("The cardinal image data set is not high dimensional.")
if (sparrowHighDimensional):
    print("The sparrow image data set is high dimensional.")
else:
    print("The sparrow image data set is not high dimensional.")
if (robinHighDimensional):
    print("The robin image data set is high dimensional.")
else:
    print("The robin image data set is not high dimensional.")

# Calculate and display mean values of features for each bird
cardinalMean = featureMeanDataset(cardinalDataSet)
sparrowMean = featureMeanDataset(sparrowDataSet)
robinMean = featureMeanDataset(robinDataSet)
plt.plot(cardinalMean, label = 'Cardinal')
plt.plot(sparrowMean, label = 'Sparrow')
plt.plot(robinMean, label = 'Robin')
plt.xlabel("Feature Numbers")
plt.ylabel("Mean Values")
plt.legend()
plt.show()

cardinalVariance = featureVarianceDataset(cardinalDataSet)
sparrowVariance = featureVarianceDataset(sparrowDataSet)
robinVariance = featureVarianceDataset(robinDataSet)

# Calculate and display standard deviations of features for each bird
cardinalStd = featureStandDevDataset(cardinalDataSet)
sparrowStd = featureStandDevDataset(sparrowDataSet)
robinStd = featureStandDevDataset(robinDataSet)
plt.plot(cardinalStd, label = 'Cardinal')
plt.plot(sparrowStd, label = 'Sparrow')
plt.plot(robinStd, label = 'Robin')
plt.xlabel("Feature Numbers")
plt.ylabel("Standard Deviation Values")
plt.legend()
plt.show
