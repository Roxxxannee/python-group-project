
import copy
import random
class Matrix:
	def __init__(self, data=None, dim=None, init_value=0):
		if data is None and dim is None:
			raise ValueError("Either data or dim should be provided.")
		if data is not None:
			self.data = data
			self.dim = (len(data),len(data[0]))
		else:
			n,m = dim
			self.dim = (n,m)
			self.data = [[init_value for _ in range(m)] for _ in range(n)]



	def __repr__(self):
		repr_str = "[["
		for i,row in enumerate(self.data):
			repr_str += " ".join(map(str,row))
			if i < len(self.data) - 1:
				repr_str += "]\n ["
			else:
				repr_str += "]]"
		return repr_str


	def shape(self):
		return self.dim

	def reshape(self,newdim):
		if newdim[0]*newdim[1] !=self.dim[0]*self.dim[1]: #判断新旧矩阵的元素个数是否相同
			return "VallueErroe"
		else:
			newMatrix=Matrix(dim=newdim)
			value_lst=[]
			for j in range(self.dim[0]):
				for i in self.data[j]:
					value_lst.append(i) #将所有的元素展平成一个列表
			for i in range(newdim[0]):
				newMatrix.data[i]=value_lst[newdim[1]*i:newdim[1]*(i+1)]#将每个元素按顺序填进新矩阵
			return newMatrix.data

	def dot(self, other):
		if self.dim[1] != other.dim[0]:
			raise ValueError("Matrix dimensions do not match for dot product")
		C = Matrix(dim = (self.dim[0], other.dim[1]))
		for i in range(self.dim[0]):
			for j in range(other.dim[1]):
				C.data[i][j] = sum(self.data[i][k] * other.data[k][j] for k in range(self.dim[1]))
		return C


	def T(self):
		n,m = self.dim
		T_matrix = Matrix(dim=(m,n))
		for i in range(n):
			for j in range(m):
				T_matrix.data[j][i] = self.data[i][j]
		return T_matrix


	def sum(self, axis=None):
		if axis == 0:  # 对列求和
			return Matrix(data=[[sum(self.data[i][j] for i in range(self.dim[0])) for j in range(self.dim[1])]])
		elif axis == 1:  # 对行求和
			return Matrix(data=[[sum(self.data[i][j] for j in range(self.dim[1]))] for i in range(self.dim[0])])
		elif axis is None:  # 所有元素的和
			return Matrix(data=[[sum(self.data[i][j] for i in range(self.dim[0]) for j in range(self.dim[1]))]])
	def copy(self):
		return Matrix(copy.deepcopy(self.data))

	def Kronecker_product(self, other):
		value_lst=[] #元素按顺序展开的列表
		result_matrix=Matrix(dim=(self.dim[0]*other.dim[0],self.dim[1]*other.dim[1]))
		def Kronecker_one_row(p,q):
			for j in range(self.dim[1]):
				for i in range(other.dim[1]):
					value_lst.append(self.data[p][j]*other.data[q][i]) #一行的效果
			return value_lst
		for y in range(self.dim[0]):
			for x in range(other.dim[0]):
				Kronecker_one_row(y,x)  #多行的效果
		for i in range(result_matrix.dim[0]):
			result_matrix.data[i]=value_lst[result_matrix.dim[0]*i:result_matrix.dim[0]*(i+1)]##将每个元素按顺序填进新矩阵
		return result_matrix.data

	def __getitem__(self, key):
		
		if isinstance(key[0],slice) and isinstance(key[1],slice):  #判断是否是切片
			row_slice,col_slice=key #赋值
			rows=self.data[row_slice]
			result=[row[col_slice] for row in rows]
			return result
		else:
			return self.data[key[0]][key[1]]

	def __setitem__(self, key, value):
		if isinstance(key[0], slice) and isinstance(key[1], slice):
			row_slice, col_slice = key
			row_start, row_stop= row_slice.start or 0, row_slice.stop or self.dim[0]
			col_start, col_stop= col_slice.start or 0, col_slice.stop or self.dim[1]
			row_size = row_stop - row_start
			col_size = col_stop - col_start
			if (row_size, col_size) != value.dim:
				return 'ValueError'
			for r in range(row_start, row_stop):
				for c in range(col_start, col_stop):
					self.data[r][c] = value.data[r - row_start][c - col_start]
			return self.data
		else:
			self.data[key[0]][key[1]] = value
			return self.data
		
	def I(n):
		return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

	def __pow__(self, n):
		if n == 0:
			return Matrix.I(n)
		else:
			result = copy.deepcopy(self)
			for i in range(n-1):
				result = result.dot(self)
			return result

	def __add__(self, other):
		C = Matrix(dim=self.dim)
		for i in range(self.dim[0]):
			for j in range(self.dim[1]):
				C.data[i][j] = self.data[i][j] + other.data[i][j]
		return C

	def __sub__(self, other):
		C = Matrix(dim=self.dim)
		for i in range(self.dim[0]):
			for j in range(self.dim[1]):
				C.data[i][j] = self.data[i][j] - other.data[i][j]
		return C

	def __mul__(self, other):
		C = Matrix(dim=self.dim)
		for i in range(self.dim[0]):
			for j in range(self.dim[1]):
				C.data[i][j] = self.data[i][j] * other.data[i][j]
		return C


	def __len__(self):
		return self.dim[0] * self.dim[1]

	def __str__(self):
		formatted_rows = []
		for row in self.data:
			formatted_row = " ".join(f"{elem:4d}" if isinstance(elem, int) else f"{elem:4.2f}" for elem in row)

			formatted_rows.append(f"[ {formatted_row} ]")

		return "\n".join(formatted_rows)

	def det(self):
		if len(self.data)!=len(self.data[0]):
			return False
		else:
			n=len(self.data)
			a =copy.deepcopy(self.data)
		det_val = 1
		for i in range(n):
			max_row_index = i
			max_element = abs(a[i][i])
			for k in range(i+1, n):
				if abs(a[k][i]) > max_element:
					max_row_index = k
					max_element = abs(a[k][i])
			if i != max_row_index:
				a[i],a[max_row_index] = a[max_row_index],a[i]
				det_val *= -1
			for k in range(i+1, n):
				ratio = a[k][i] / a[i][i]
				a[k] = [a[k][j] - ratio * a[i][j] for j in range(n)]
		for i in range(n):
			det_val *= a[i][i]
		return det_val

	def inverse(self):
		rows = len(self.data)
		if rows == 0 or len(self.data[0]) != rows:
			raise ValueError("矩阵必须是非奇异方阵")


		augmented_matrix = [row[:] for row in self.data]
		identity_matrix = [[1 if i == j else 0 for j in range(rows)] for i in range(rows)]


		for i in range(rows):
			augmented_matrix[i] += identity_matrix[i]

		for i in range(rows):
			max_row = i
			for k in range(i + 1, rows):
				if abs(augmented_matrix[k][i]) > abs(augmented_matrix[max_row][i]):
					max_row = k

			augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

			if abs(augmented_matrix[i][i]) < 1e-10:
				raise ValueError("该矩阵是奇异矩阵，无法求逆")

			pivot = augmented_matrix[i][i]
			for j in range(2 * rows):
				augmented_matrix[i][j] /= pivot

			for k in range(rows):
				if k != i:
					factor = augmented_matrix[k][i]
					for j in range(2 * rows):
						augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

		inverse_matrix = [row[rows:] for row in augmented_matrix]

		return Matrix(data=inverse_matrix)

	def rank(self):
		rows = len(self.data)
		if rows == 0:
			return 0
		cols = len(self.data[0])

		matrix = [row[:] for row in self.data]

		rank = 0
		for r in range(rows):

			if r >= cols:
				break
			max_row = r
			for i in range(r + 1, rows):
				if abs(matrix[i][r]) > abs(matrix[max_row][r]):
					max_row = i
			matrix[r], matrix[max_row] = matrix[max_row], matrix[r]

			if abs(matrix[r][r]) < 1e-10:
				continue

			for j in range(r + 1, cols):
				matrix[r][j] /= matrix[r][r]

			for i in range(r + 1, rows):
				factor = matrix[i][r]
				for j in range(r, cols):
					matrix[i][j] -= factor * matrix[r][j]

			rank += 1

		return rank

	def narray(dim, init_value=1):
		return Matrix(dim = dim, init_value = init_value)

	def arange(start, end, step):
		return Matrix(data=[list(range(start, end, step))])


	def zeros(dim):
		return Matrix(dim = dim, init_value = 0)

	def zeros_like(matrix):
		return Matrix.zeros(matrix.dim)

	def ones(dim):
		return Matrix(dim = dim, init_value = 1)

	def ones_like(matrix):

		return Matrix.ones(matrix.dim)

	def nrandom(dim):
		if isinstance(dim, int):
			return Matrix(data = [random.random()])
		else:
			return Matrix(data = [[random.random()for j in range(dim[1])] for i in range(dim[0])])
			
	def nrandom_like(matrix):
		return Matrix.nrandom(matrix.dim)

	def concatenate(items, axis=0):
		if axis == 0:
			if not all(item.dim[1] == items[0].dim[1] for item in items):
				raise ValueError("Matrix dimensions do not match for concatenation along axis 0")
			else:
				dim = (sum(item.dim[0] for item in items), items[0].dim[1])
				data = []
				for item in items:
					data.extend(item.data)
				return Matrix(dim=dim, data=data)
		elif axis == 1:
			if not all(item.dim[0] == items[0].dim[0] for item in items):
				raise ValueError("Matrix dimensions do not match for concatenation along axis 1")
			else:
				dim = (items[0].dim[0], sum(item.dim[1] for item in items))
				data = []
				for item in items:
					data.extend(item.T().data)
				return Matrix(dim=dim, data=data).T()
		else:
			raise ValueError("axis should be 0 or 1")

	def vectorize(func):
		def vectorized_function(x):
			new_data = [[0 for j in range(x.dim[1])] for i in range(x.dim[0])]
			for i in range(x.dim[0]):
				for j in range(x.dim[1]):
					new_data[i][j] = func(x.data[i][j])
			return Matrix(data=new_data)
		return vectorized_function


if __name__ == "__main__":
	print("test here")
