import numpy as np



class Point2D():
    def __init__(self, point):
        """
        :param point: [x, y]
        """
        self.theta = np.atan2(point[1], point[0])
        self.dis = np.linalg.norm(point)
        self.define_domain = [self.theta - np.pi/2, self.theta + np.pi/2]

    def get_distance(self, theta):
        distance = np.cos(theta - self.theta)
        if distance >= 0:
            return distance
        else:
            return -1


def get_matrix(points, resolution = 1):
    """
    :param points: a list of Point2D
    :param resolution:
    :return:
    """
    row = len(points)
    col = 360//resolution
    number = [0 for _ in range(col)]
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            theta = -180 + j * resolution
            temp = points[i].get_distance(theta)
            result[i][j] = temp
            if temp > -0.1:
                number[j] += 1
    return result, number

def max_number_and_min_variance(matrix, number, point_num, maxSigma=5, min_number_of_hyperplanes=20):
    index = []
    number_index = np.argsort(-1 * number)
    max_number = number[number_index[0]]
    if max_number < min_number_of_hyperplanes:
        return 0, 0, 0, 0, [], 0, True
    for id in number_index:
        if number[number_index[id]] == max_number:
            index.append(id)
        else:
            break
    min_variance = np.inf

    id = 0
    mu_result = 0
    ratio_result = 0
    pointID_result = []

    for item in index:
        array = matrix[:, item]
        avg, sigma, ratio, number, pointID = fitting_with_one_hyperplane(point_num, array, number[item], maxSigma)
        if sigma < min_variance:
            id = item
            min_variance = sigma
            mu_result = avg
            ratio_result = ratio
            pointID_result = pointID
    return mu_result, min_variance, ratio_result, number[id], pointID_result, id, False

def fitting_with_one_hyperplane(point_num, array, number, maxSigma=5):
    total = 0
    pointID = []
    for i in range(point_num):
        if array[i] >= -0.5:
            total += array[i]
    avg = total / number
    sigma = 0
    numberInTwoHyperplanes = 0
    for i in range(point_num):
        if array[i] >= -0.5:
            dis2 = (array[i]-avg) ** 2
            sigma = sigma + dis2
            if dis2 <= (maxSigma**2):
                numberInTwoHyperplanes += 1
                pointID.append(i)
    sigma = sigma / number
    ratio = numberInTwoHyperplanes / number
    return avg, sigma, ratio, number, pointID

def whether_need_more_hyperplanes(mu, sigma, ratio, number, minNumber):
    if ratio*number >= minNumber and ratio > 0.5:
        return False
    else:
        return True

def finding_max_probability_hyperplane(minNumber, point_num, array, number, maxSigma=5):
    maxHyperplanes = number / minNumber
    pointID = []
    deltarho = max(array) / maxHyperplanes
    rho = [i*deltarho for i in range(maxHyperplanes)]
    mu = 0
    last_mu = 0

    while abs(last_mu - mu) < 0.2:
        rho_num = [0 for _ in range(maxHyperplanes)]
        temp = [[] for _ in range(maxHyperplanes)]
        for i in range(point_num):
            for j in range(maxHyperplanes):
                if array[i] >= -0.5 and abs(array[i] - rho[j]) <= maxSigma:
                    rho_num[j] += 1
                    temp[j].append(array[i])
        for k in range(maxHyperplanes):
            rho[k] = np.mean(np.array(temp[k]))
        last_mu = mu
        mu_index = np.argsort(np.array(rho_num))[-1]
        mu = rho[mu_index]

    for i in range(point_num):
        if array[i] >= -0.5 and abs(array[i] - mu) <= maxSigma:
            pointID.append(i)

    return mu, pointID

def fitting_with_more_hyperplanes():
    pass

def updateMatrix(pointID, Matrix, number):
    result = Matrix
    num = number
    for item in pointID:
        for j in range(len(number)):
            if result[item][j] >= -0.5:
                result[item][j] = -1
                num[j] -= 1
    return result, num


def get_initial_hyperplanes(points, resolution=1, maxSigma=5, minNumber=20, maxHyperplanes=10):
    hyperplanes = []
    point_num = len(points)
    new_points = [Point2D(item) for item in points]
    # points_used = [False for _ in range(point_num)]
    # multi_hyperplanes_index = []
    # single_hyperplane_index = []
    Matrix, number = get_matrix(new_points, resolution)

    while True:
        # 找包含点最多的theta值（可能有多个）的索引构成列表，计算方差，找方差最小的哪一个
        #            假设只有一个平面，根据上面计算的均值和方差，计算被两个平面夹住的点占总点的比率
        mu, sigma, ratio, number, pointID, id, breakFlag = max_number_and_min_variance(Matrix, number, point_num, maxSigma, minNumber)
        if breakFlag or len(hyperplanes) >= maxHyperplanes:
            break
        theta = (id * resolution - 180) * np.pi / 180
        if whether_need_more_hyperplanes(mu, sigma, ratio, number, minNumber):
            # 处理含平行线的
            # fitting_with_more_hyperplanes()
            mu, pointID = finding_max_probability_hyperplane(minNumber, point_num, Matrix[:, id], number[id], maxSigma)
            hyperplanes.append(np.array([mu, theta]))
        else:
            # 不含平行线
            hyperplanes.append(np.array([mu, theta]))
        # 更新matrix
        Matrix, number = updateMatrix(pointID, Matrix, number)

    return np.array(hyperplanes)







