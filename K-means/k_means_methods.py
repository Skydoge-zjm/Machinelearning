import numpy as np
import math
from PIL import Image


class ImageDivision:
    """
    Introduction & Procedure:
    每个点的属性：RGB（3），坐标（2），u（属于第u个簇）
    用cluster_map (size: height * width, value: u)来表达每个点所归属的簇序号
    S1: k从1取到10，进行尝试。
    S2: 对每个k，随机选取初始簇心100轮。
    S3: 在RGB三维空间中，计算所有点到每个簇心的距离。
    S4: 对每个点找到最近簇心，聚类，更新cluster_map。
    S5: 若簇心改变量小于收敛阈值，进入下一步，否则返回S3。
    S6: 绘制分类后的K色图。

    Parameters:
    input: RGB_value:  array(3d)
     _____________.
    /____________/|
    |           | |  height(rows)
    |           | |
    |___________|/ RGB(3)
      width(cols)
    height:  int
    width:  int
    converge_threshold:  float   # 收敛阈值，可选参数，默认：0.0001
    iteration_threshold:  int  # 迭代阈值，可选参数，默认：100
    若迭代次数超过迭代阈值时仍不收敛，返回异常值：-1

    return:
        return_value_1: (kmeans)
            0 : 正常
            -1 : 迭代次数超过迭代阈值时仍不收敛

        return_value_2: (draw)
            0 : 正常

    """

    def __init__(self, image, converge_threshold=0.0001, iteration_threshold=100):
        self.image = image  # 归一化后所有像素点的RGB值矩阵
        self.height = int(image.shape[0])
        self.width = int(image.shape[1])
        self.cluster_map = np.zeros((self.height, self.width))  # create cluster_map matrix  240, 320
        self.converge_threshold = converge_threshold
        self.iteration_threshold = iteration_threshold

    def kmeans(self):
        for i in range(3, 7):  # k from 2 to 10
            k = i
            for j in range(5):  # 5 tries of random_centroids_init
                centroids_rgb = self.__random_centroids_init(k)
                # centroids_rgb = self.image[centroids_indexes]
                # print(centroids_rgb)
                change = 10
                iteration = 0  # 迭代计数器
                while change >= self.converge_threshold:
                    self.cluster_map = self.__update_cluster_map(k, centroids_rgb)  # 更新 cluster map
                    temp = self.__update_centroids_rgb(k)  # 将新簇心RGB暂存入temp
                    change = self.__calculate_change(centroids_rgb, temp, k)  # 上一轮迭代与本轮迭代后簇心rgb变化量
                    centroids_rgb = temp  # 更新簇心RGB
                    if iteration > self.iteration_threshold:
                        return -1
                    iteration += 1

                # print(self.cluster_map)
                self.draw(i, j)
        return 0

    def draw(self, s, t):
        divided_image = np.zeros(shape=(self.height, self.width, 3))
        for i in range(self.height):
            for j in range(self.width):
                a = self.cluster_map[i, j]
                if a == 0:  # 红
                    divided_image[i, j, :] = [255, 0, 0]
                elif a == 1:
                    divided_image[i, j, :] = [0, 255, 0]
                elif a == 2:
                    divided_image[i, j, :] = [0, 0, 255]
                elif a == 3:
                    divided_image[i, j, :] = [255, 255, 255]
                elif a == 4:
                    divided_image[i, j, :] = [255, 255, 0]

        divided_image = Image.fromarray(np.uint8(divided_image))
        filename = 'divided_image_' + str(s) + '_' + str(t) + '.jpg'
        divided_image.save(filename)
        return 0

    # 随机选出k个初始聚类中心
    def __random_centroids_init(self, k):
        """
        return:  list: coordinates of the k centroids
        """
        centroid_indexes = []
        centroid_rgb = []
        for i in range(k):
            centroid_index_height = np.random.choice(range(self.height))
            centroid_index_width = np.random.choice(range(self.width))
            centroid_indexes.append((centroid_index_height, centroid_index_width))
        # print(centroid_indexes)
        for i in range(k):
            centroid_rgb.append(self.image[centroid_indexes[i]])
        return centroid_rgb

    def __calculate_change(self, old_rgb, new_rgb, k):
        """
        :param old_rgb: list[list]
        :param new_rgb: list[list]
        :return: num
        """

        old_rgb = np.array(old_rgb)
        new_rgb = np.array(new_rgb)
        a = np.power(old_rgb - new_rgb, 2)
        change = math.sqrt(np.sum(a)) / float(k)
        return change

    def __update_cluster_map(self, k, centroids_rgb):
        """
        过程：
        计算image中每个点到标号为i的簇心的欧氏距离的平方
            将簇心RGB[1, 1, 3]广播为[height, width, 3]的三维矩阵a
            a与image作减法后平方沿第三维加和得到b
        形成k个二维矩阵，矩阵i中的某一位置元素表示image中这一位置的RGB坐标与簇心i的距离
        沿第三维度叠加，形成3d-array[height, width, k]
        沿第三维取最大值，并记录每一个最大值所处的第三维坐标，即为new_cluster_map
        :param k: int  簇心数
               centroids_rgb: list 元素为元组  len()=k  簇心坐标
        :return: 更新后的cluster map, matrix [height, width]
        """
        y = np.zeros(shape=(self.height, self.width))
        flag = 1
        for i in range(k):
            x = centroids_rgb[i]  # list, len()=3
            a = np.array([[x]])
            b = np.sum(np.power(self.image - a, 2), axis=2)
            if flag == 1:
                y = b
                flag = 0
            else:
                if np.ndim(y) == 2:
                    y = np.stack((y, b), axis=2)
                else:
                    y = np.concatenate((y, np.expand_dims(b, axis=2)), axis=2)
        new_cluster_map = np.argmin(y, axis=2)
        # print(new_cluster_map)
        return new_cluster_map

    def __update_centroids_rgb(self, k):
        """
        过程：
        temp: 临时列表 ,len()=k, 元素为列表
        遍历image所有位置：
            将标号为t的位置的rgb存入temp第t项的列表中
        最终，列表temp中有k个列表，每个列表中有若干个3元素列表
        对temp每一项（列表）的所有元素求出平均RGB
        :param k:
        :return:
        """

        temp = []
        new_centroids_rgb = []
        for i in range(k):
            temp.append([])
        for i in range(self.height):
            for j in range(self.width):
                index = (i, j)
                temp[self.cluster_map[index]].append(self.image[index].tolist())
        for i in range(k):
            n = len(temp[i])
            temp_1 = np.array(temp[i])  # 被归入i簇的所有点的RGB，再转换为array
            temp_2 = np.mean(temp_1, axis=0).tolist()
            new_centroids_rgb.append(temp_2)

        return new_centroids_rgb

