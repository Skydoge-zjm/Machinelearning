import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import k_means_methods as kmm

"""
S1: read in .png as 3d-array
S2: kmeans and compute J
    take 200 times of random_centroids_init's choice
S3: use elbow method to get the proper num of k 
S4: output a picture of k colors
"""


# read in
image = mpimg.imread('12-3.png')
# print(image.shape)
# print(image)

# n = int(input("Please input the number of cluster you need: "))
X = kmm.ImageDivision(image=image)  # 创建实例

"""
return_value_1: 
    0 : 正常
    -1 : 迭代次数超过迭代阈值时仍不收敛

return_value_2:
    0 : 正常
    
"""
return_value_1 = X.kmeans()
return_value_2 = 1
if return_value_1 == 0:
    print("K-means finished.")
elif return_value_1 == -1:
    print("CANNOT converge within the iteration threshold.You can try the solutions below:")
    print("- Turn up the iteration threshold.")
    print("- Check the code.")

if return_value_2 == 1:
    print("Please check the process of K-means.")
elif return_value_2 == 0:
    print("Generation of clustered picture finished.")

