# IMPORTING NUMPY
Import numpy as np # np IS USED AS ALIAS FOR NUMPY 

# CREATING ARRAYS USING NUMPY 
array1 = np.array([1,2,3,4,5,6])  #CREATED AN ARRAY NAMED AS array1
array2 = np.array([7,8,9,10,11])  #CREATED AN ARRAY NAMED AS array2

# TO CHECK THE TYPE OF THE LIST WE WRITE 
type(array1) #output : numpy.ndarray
type(array2) #output : numpy.ndarray

# CREATING MATRICES WITH VALUES 0 AND 1

zeros = np.zeros((2, 3))  #CREATES A MATRIX OF 2 ROWS AND 3 COLUMNS FILLED WITH ZEROS
ones = np.ones((2, 2))    #CREATES A MATRIX OF 2 ROWS AND 2 COLUMNS FILLED WITH ONES
print(zeros,ones)

output : (array([[0., 0., 0.],
                [0., 0., 0.]]),
          array([[1., 1.],
                [1., 1.]]))

# USING ARANGE FUNCTION TO PRINT VALUES BETWEEN A CERTAIN RANGE
array_range = np.arange(0, 10, 2)  #IN HERE 0 IS STARTING VALUE ,10 IS ENDING VALUE (DOES NOT INCLUDE THE END VALUE) AND 2 IS THE STEP COUNT (HOW MANY NUMBERS SHOULD I SKIP BETWEEN EACH) 
print(array_range)

output : array([0, 2, 4, 6, 8])

lin = np.linspace(0, 1, 5)
output : array([0. , 0.25, 0.5 , 0.75, 1.  ])

#0 IS THE START OF THE SEQUENCE,1 IS THE END OF THE SEQUENCE,5 IS THE NUMBER OF SAMPLES TO GENERATE.

# ATTRIBUTES 
print(array1.shape)    output : (3,)
print(array1.ndim)     output : 1
print(array1.dtype)    output : int64


# RESHAPING 
reshaped = array_range.reshape(5, 1)  #RESHAPES VALUES(DIMENSION)
print(reshaped)

output : array([[0],
               [2],
               [4],
               [6],
               [8]])
# INDEXING
print(array[0])      #RETURNS VALUES IN THOSE INDEXS 
print(array[1][0])

# Slicing
print(array_range[1:5])   #RETURNS FROM INDEX 1 TO 5

# Mathematical operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
print(a * b)
print(np.sqrt(a))  #SQUARE ROOT OF a

# Aggregations
print(np.sum(a))   #FINDS ADDITION OF ALL VALUES IN a
print(np.mean(a))  #FINDS MEAN OF ALL VALUES IN a
print(np.max(a))   #FINDS MAXIMUM VALUE IN a
print(np.min(a))   #FINDS MINIMUM VALUE IN a

# Random numbers
rand = np.random.rand(3)               #GENERATES RANDOM NUMBERS 
rand_int = np.random.randint(1, 10, 5)

# Boolean operations
arr = np.array([1, 2, 3, 4])
print(arr > 2)

# Filtering
print(arr[arr > 2])
