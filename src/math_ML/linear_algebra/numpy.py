import numpy as np

# one dimension array
test_array = np.array([1,2,3])
print(test_array)

# creating array with 3 integer
test_array2 = np.arange(5)
print(test_array2)

# create array that start with 1 until 20 and the incremet is 3
test_array3 = np.arange(1, 20, 3)
print(test_array3)

# membuat urutan (array) berisi nilai-nilai yang berada dalam rentang tertentu dengan jumlah titik atau elemen tertentu.
test_array4 = np.linspace(1, 100, 4)
print(test_array4)
# bisa pakein data type biar ga float
test_array5 = np.linspace(1, 100, 4, dtype=int)
print(test_array5)

# char array
test_array6 = np.array('i love mathematics')
print(test_array6)
print(test_array6.dtype)

# Return a new array of shape 3, filled with ones. 
ones_arr = np.ones(3)
print(ones_arr)
# bia pake dtype juga biar int

# Return a new array of shape 3, filled with zeroes.
zeros_arr = np.zeros(3)
print(zeros_arr)
# bia pake dtype juga biar int

# Return a new array of shape 3, without initializing entries.
empt_arr = np.empty(3)
print(empt_arr)

# Return a new array of shape 3 with random numbers between 0 and 1.
rand_arr = np.random.rand(3)
print(rand_arr)

# two dimension array
two_dim_arr = np.array([[1,2,3], [4,5,6]])
print(two_dim_arr)

# transfomr 1 dimension into 2 dimension array
test_array7 = np.array([1,2,3,4,5,6,7,8,])
# yang dalem kurung itu angka pertama buat ubah ke dimensi berapa
# yang angka ke 2 buat ksh tau ada berapa di value di array itu
test_array8 = np.reshape(test_array7, (2,4))
print(test_array8)

# cek dimensi array
test_array8.ndim

# cek shape
test_array8.shape

# cek total value
test_array8.size

# math operation in array
arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])

# Adding two 1-D arrays
addition = arr_1 + arr_2
print(addition)

# Subtracting two 1-D arrays
subtraction = arr_1 - arr_2
print(subtraction)

# Multiplying two 1-D arrays elementwise
multiplication = arr_1 * arr_2
print(multiplication)

# multiplying with vector
vector = np.array([1, 2])
vector * 1.6

# Select the third element of the array. Remember the counting starts from 0.
test_array9 = ([1, 2, 3, 4, 5])
print(test_array9[2])

# Select the first element of the array.
print(test_array9[0])

# Slice the array a to get the array [2,3,4]
sliced_arr = test_array9[1:4]
print(sliced_arr)

# Slice the array a to get the array [1,2,3]
sliced_arr = test_array9[:3]
print(sliced_arr)

# Slice the array a to get the array [3,4,5]
sliced_arr = test_array9[2:]
print(sliced_arr)

# Slice the array a to get the array [1,3,5]
sliced_arr = test_array9[::2]
print(sliced_arr)

# Indexing on a 2-D array
two_dim = np.array(([1, 2, 3],
          [4, 5, 6], 
          [7, 8, 9]))

# Select element number 8 from the 2-D array using indices i, j.
print(two_dim[2][1])

# Slice the two_dim array to get the first two rows
sliced_arr_1 = two_dim[0:2]
sliced_arr_1

# Similarily, slice the two_dim array to get the last two rows
sliced_two_dim_rows = two_dim[1:3]
print(sliced_two_dim_rows)

# cari value di index yang ke berapa dri smua array
sliced_two_dim_cols = two_dim[:,1]
print(sliced_two_dim_cols)

#stack array
a1 = np.array([[1,1], 
               [2,2]])
a2 = np.array([[3,3],
              [4,4]])
print(f'a1:\n{a1}')
print(f'a2:\n{a2}')

# Stack the arrays vertically
vert_stack = np.vstack((a1, a2))
print(vert_stack)

# Stack the arrays horizontally
horz_stack = np.hstack((a1, a2))
print(horz_stack)