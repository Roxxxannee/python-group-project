class Matrix:
    r"""
    自定义的二维矩阵类

    Args:
        data: 一个二维的嵌套列表，表示矩阵的数据。即 data[i][j] 表示矩阵第 i+1 行第 j+1 列处的元素。
              当参数 data 不为 None 时，应根据参数 data 确定矩阵的形状。默认值: None
        dim: 一个元组 (n, m) 表示矩阵是 n 行 m 列, 当参数 data 为 None 时，根据该参数确定矩阵的形状；
             当参数 data 不为 None 时，忽略该参数。如果 data 和 dim 同时为 None, 应抛出异常。默认值: None
        init_value: 当提供的 data 参数为 None 时，使用该 init_value 初始化一个 n 行 m 列的矩阵，
                    即矩阵各元素均为 init_value. 当参数 data 不为 None 时，忽略该参数。 默认值: 0

    Attributes:
        dim: 一个元组 (n, m) 表示矩阵的形状
        data: 一个二维的嵌套列表，表示矩阵的数据

    Examples:
        mat1 = Matrix(dim=(2, 3), init_value=0)
        print(mat1)
        [[0 0 0]
         [0 0 0]]
        mat2 = Matrix(data=[[0, 1], [1, 2], [2, 3]])
        print(mat2)
        [[0 1]
        [1 2]
        [2 3]]
    """

    def __init__(self, data=None, dim=None, init_value=0):
        if data is None and dim is None:
            raise ValueError("Either data or dim should be provided.")
        if data is not None:
            self.data = data
            self.dim = (len(data), len(data[0]))
        else:
            n, m = dim
            self.dim = (n, m)
            self.data = [[init_value for _ in range(m)] for _ in range(n)]

    def __repr__(self):
        repr_str = "[["
        for i, row in enumerate(self.data):
            repr_str += " ".join(map(str, row))
            if i < len(self.data) - 1:
                repr_str += "]\n ["
            else:
                repr_str += "]]"
        return repr_str

    def shape(self):
        r"""
        返回矩阵的形状 dim
        """
        return self.dim

    def reshape(self, newdim):
        r"""
        将矩阵从(m,n)维拉伸为newdim=(m1,n1)
        该函数不改变 self

        Args:
            newdim: 一个元组 (m1, n1) 表示拉伸后的矩阵形状。如果 m1 * n1 不等于 self.dim[0] * self.dim[1],
                    应抛出异常

        Returns:
            Matrix: 一个 Matrix 类型的返回结果, 表示 reshape 得到的结果
        """
        m1, n1 = newdim
        if m1 * n1 != self.dim[0] * self.dim[1]:
            raise ValueError("newdim should have the same product as the original dim")
        elem_list = [self.data[i][j] for i in range(self.dim[0]) for j in range(self.dim[1])]
        reshape_matrix = [elem_list[i:i + n1] for i in range(0, len(elem_list), n1)]
        return Matrix(data=reshape_matrix, dim=newdim)

    def dot(self, other):
        r"""
        矩阵乘法：二维矩阵乘以二维矩阵
        按照公式 A[i, j] = \sum_k B[i, k] * C[k, j] 计算 A = B.dot(C)

        Args:
            other: 参与运算的另一个 Matrix 实例

        Returns:
            Matrix: 计算结果

        Examples:
            >A = Matrix(data=[[1, 2], [3, 4]])
            > A.dot(A)
            >[[ 7 10]
              [15 22]]
        """
        if self.dim[1] != other.dim[0]:
            raise ValueError("Matrix dimensions do not match for dot product")
        C = Matrix(dim=(self.dim[0], other.dim[1]))
        for i in range(self.dim[0]):
            for j in range(other.dim[1]):
                C.data[i][j] = sum(self.data[i][k] * other.data[k][j] for k in range(self.dim[1]))
        return C

    def T(self):
        r"""
        矩阵的转置

        Returns:
            Matrix: 矩阵的转置

        Examples:
            > A = Matrix(data=[[1, 2], [3, 4]])
            > A.T()
            > [[1 3]
               [2 4]]
            > B = Matrix(data=[[1, 2, 3], [4, 5, 6]])
            > B.T()
            > [[1 4]
               [2 5]
               [3 6]]
        """
        n, m = self.dim
        T_matrix = Matrix(dim=(m, n))
        for i in range(n):
            for j in range(m):
                T_matrix.data[j][i] = self.data[i][j]
        return T_matrix

