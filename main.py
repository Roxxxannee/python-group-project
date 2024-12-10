# Test code for IEEE course final project
# Fan Cheng, 2024

import minimatrix as mm
# Test code for IEEE course final project
from minimatrix import Matrix, arange, zeros, zeros_like, ones, ones_like, nrandom, nrandom_like


mat = Matrix(data=[[1,2,3], [6,4,5], [7,8,9]])
print("Original 3×3 Matrix mat:")
print(mat)

print("\nTranspose of mat:")
print(mat.T())

print("\nDeterminant of mat:")
print(mat.det())

print("\nInverse of mat:")
print(mat.inverse())

print("\nTesting addition and subtraction:")
mat2 = Matrix(data=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
print("mat + mat2:")
print(mat + mat2)
print("mat - mat2:")
print(mat - mat2)

m24 = arange(0, 24, 1)
print("Original 1×24 Matrix m24:")
print(m24)

print("\nReshape m24 to 3×8:")
print(m24.reshape((3, 8)))

print("\nReshape m24 to 24×1:")
print(m24.reshape((24, 1)))

print("\nReshape m24 to 4×6:")
print(m24.reshape((4, 6)))


zero_matrix = zeros((3, 3))
print("3×3 zeros matrix:")
print(zero_matrix)
print(zeros_like(m24))

one_matrix = ones((3, 3))
print("3×3 ones matrix:")
print(one_matrix)
print(ones_like(m24))

random_matrix = nrandom((3, 3))
print("3×3 random matrix:")
print(random_matrix)
print(nrandom_like(m24))


m, n = 1000, 100
X = nrandom((m, n))  # m×n 随机矩阵
w = nrandom((n, 1))  # n×1 随机向量
e = nrandom((m, 1))  # m×1 随机误差向量
Y = X.dot(w) + e  # 计算 Y = Xw + e

# 最小二乘法估计 ŵ = (X^T X)^-1 X^T Y
X_T = X.T()
XTX = X_T.dot(X)

XTX_inv = XTX.inverse()
XTY = X_T.dot(Y)
w_hat = XTX_inv.dot(XTY)
print("Original w:")
print(w)
print("\nEstimated ŵ:")
print(w_hat)




