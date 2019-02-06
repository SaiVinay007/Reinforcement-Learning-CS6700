import numpy as np
import tensorflow as tf


def customOps(n):
    
    # Placeholder for input matrix with data type float
    customOps.mat = tf.placeholder(tf.float32, shape = [n, n])    
    

    '''
        1.Transpose the elements in the bottom-right triangle of matrix

        upper_left_part         :   makes all the elements at right-bottom of matrix to zeros
        bottom_right_part       :   makes all the elements at left-top of the transposed matrix to zeros
        diagonal_rev_part       :   get the North East facing diagonal of bottom_right_matrix
    
    '''
  
    upper_left_part = tf.reverse(tf.matrix_band_part(tf.reverse(customOps.mat, [-1]), 0, -1),[-1])
    bottom_right_part =  tf.reverse(tf.matrix_band_part(tf.reverse(tf.transpose(customOps.mat), [-1]), -1, 0), [-1])
    diagonal_rev_part = tf.reverse(tf.matrix_band_part(tf.reverse(bottom_right_part, [-1]), 0, 0),[-1])
        
    A = upper_left_part + bottom_right_part - diagonal_rev_part

    
    # 2.Take the maximum value along the columns of A to get a vector m
    # Gets the maximum element in each row of A
    m = tf.reduce_max(A, axis = 1)
    

    # 3.Softmax matrix
    # temp a list of softmax on sliced "m" and padded with zeros to make length 'n' 
    # B is the converted tensor form of temp
    temp = []
    for i in range(n):
        padding = [[0,i]]
        temp.append(tf.pad(tf.nn.softmax(tf.slice(m,[0],[n-i]), axis = 0), padding, mode='CONSTANT', name=None))
    B = tf.convert_to_tensor(temp)
        
     
    # 4.Sum along the rows of B to obtain vector ~v 1 
    v1 = tf.reduce_sum(B, axis = 0)
    
    
    # Sum along the columns of B to get another vector ~v 2 
    v2 = tf.reduce_sum(B, axis = 1)
    
    
    # 5.Concatenate the two vectors and take a softmax of this vector: ~v = softmax(concat(~v 1 , ~v 2 ))
    v  = tf.nn.softmax(tf.concat([v1, v2], axis = 0), axis = 0)
    
    
    # 6.Get the index number in vector ~v with maximum value
    index = tf.argmax(v, axis = 0)
    
    
    # 7.If index number is greater than n/3 gives sum(square(differences of vectors)) else gives sum(square(addition of vectors)): 
    finalVal = tf.cond(index > tf.cast((n/3),tf.int64), lambda : tf.reduce_sum(tf.square(v1-v2)), lambda : tf.reduce_sum(tf.square(v1+v2)))
    
    
    return finalVal


if __name__ == '__main__':
    mat = np.asarray([[0, 1, 2],
                      [1, 0, 3],
                      [2, 5, 4]])
    n = mat.shape[0]

    finalVal = customOps(n)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    outVal = sess.run(finalVal, feed_dict={customOps.mat: mat})
    print(outVal)
    sess.close()