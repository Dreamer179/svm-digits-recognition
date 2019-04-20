import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn import datasets, svm, metrics
import sqlite3
import os

# Load digits dataset
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#Training with HOG features
X_train_feature = []
for i in range(len(X_train)):
    feature = hog(X_train[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_train_feature.append(feature)
X_train_feature = np.array(X_train_feature,dtype = np.float32)
print(X_train_feature.shape)

#Choose model svm
model = LinearSVC(C=100)
#model = svm.SVC(gamma=0.001)
model.fit(X_train_feature,y_train) #fit data into model

########################## Read image and recognize each digit #####################
image = cv2.imread("digit.png")
im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im,thre = cv2.threshold(im_gray,254,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((1,10),np.uint8)
dilation = cv2.dilate(thre,kernel,iterations = 4) #gom cac ky tu tren 1 hang
ret,contours,hierachy1 = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#initializate a database to save digit 
db = open('digit.db','w')
db.close()
path = os.path.dirname('digit.db')
con = sqlite3.connect(path)
with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS digits")
    cur.execute("CREATE TABLE digits(y INT, x1 INT, y1 INT, w1 INT, h1 INT, digit TEXT, hang INT)")

#Predict digits, save to database recognized digits and its coordinates
y_list = []
for i in contours:
    (x,y,w,h) = cv2.boundingRect(i)
    y_list.append(y)
    thre1 = thre[y:y+h,x:x+w]
    _,contours1,hierachy = cv2.findContours(thre1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for j in contours1:
        (x1,y1,w1,h1) = cv2.boundingRect(j)
        cv2.rectangle(image,(x+x1,y+y1),(x+x1+w1,y+y1+h1),(0,255,0),1)
        roi = thre[y+y1:y+y1+h1,x+x1:x+x1+w1]
        roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),block_norm="L2")
        nbr = model.predict(np.array([roi_hog_fd], np.float32))  #ham du doan bang svm
        digits = (y,x1,y1,w1,h1,str(int(nbr[0])))
        with con:
            cur = con.cursor()
            cur.execute("INSERT INTO digits(y,x1,y1,w1,h1,digit) VALUES(?, ?, ?, ?, ?, ?)", digits)
        cv2.putText(image, str(int(nbr[0])), (x+x1, y+y1),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        cv2.imshow("output",image)
cv2.imwrite("output.png",image)
y_list.sort()
print(y_list)

#Save digits in to .txt file with features
fileopen = open('recognized_digits.txt','w')
with con:
    cur = con.cursor()
    for i in range(len(y_list)):
        print('Line',i+1, ':')
        fileopen.write("Line %d : "%(i+1))
        cur.execute('SELECT digit FROM digits WHERE y = ' + str(y_list[i]) + ' ORDER BY x1')
        rows = cur.fetchall()
        for row in rows:
            print(str(row)[2],' ')
            fileopen.write(str(row)[2]+' ')
        fileopen.write(" (%d digits)"%(len(rows)))
        fileopen.write('\n')
        print('(', len(rows), 'digits', ')')
fileopen.close()
cv2.waitKey()
cv2.destroyAllWindows()