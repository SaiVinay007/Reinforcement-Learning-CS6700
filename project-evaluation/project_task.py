import numpy as np
import tensorflow as tf


def customOps(n):
    
    customOps.A = tf.placeholder(tf.float32, shape = [n, n])    
    
    # 1.Transpose the elements in the bottom-right triangle of A

    '''
    A_transpose             : transpose of matrix A
    A_rev                   : flip left to right of matrix A
    A_transpose_rev         : flip left to right of A_transpose
    upper                   : upper triangular matrix of A_rev
    lower                   : lower triangular matrix of A_transpose_rev
    upper_rev               : flip left to right of upper
    lower_rev               : flip left to right of lower
    diagonal                : diagonal matrix of A_rev
    diagonal_rev            : flip left to right of diagonal
    B                       : Transpose the elements in the bottom-right triangle of A  
    '''

    A_transpose = tf.transpose(customOps.A)                         
    A_rev = tf.reverse(customOps.A, [-1])                       
    A_transpose_rev = tf.reverse(A_transpose, [-1])             
    upper =  tf.matrix_band_part(A_rev, 0, -1)                  
    lower = tf.matrix_band_part(A_transpose_rev, -1, 0)
    upper_rev = tf.reverse(upper, [-1])
    lower_rev = tf.reverse(lower, [-1])
    diagonal = tf.matrix_band_part(A_rev, 0, 0)  # diagonal matrix
    diagonal_rev = tf.reverse(diagonal, [-1])

    B = upper_rev + lower_rev - diagonal_rev
    

    # 2.Take the maximum value along the columns of A to get a vector m~ 
    
    m = tf.reduce_max(B, axis = 1)
    
    
    # 3.softmax matrix 
    
    B = tf.stack([m] for i in range(n))
    B_rev = tf.reverse(B, [-1])
    upper = tf.matrix_band_part(B_rev, 0, -1)     # Upper triangular matrix of 
    upper_rev = tf.reverse(upper, [-1])
    B = tf.nn.softmax(B, axis = 0)
            
    
    # 4.Sum along the rows of B to obtain vector ~v 1 
    
    v1 = tf.reduce_sum(B, axis = 1)
    
    
    # Sum along the columns of B to get another vector ~v 2 
    
    v2 = tf.reduce_sum(B, axis = 0)
    
    
    # 5.Concatenate the two vectors and take a softmax of this vector: ~v = softmax(concat(~v 1 , ~v 2 ))
    
    v  = tf.nn.softmax(tf.concat([v1, v2], axis = 0), axis = 0)
    
    
    # 6.Get the index number in vector ~v with maximum value
    
    index = tf.argmax(v, axis = 0)
    
    
    # 7.If index number is greater than n/3 : 
    #      finalVal = ||v 1 − ~v 2 || 2  
    # else :
    #      finalval = ||v 1 + ~v 2 || 2
    
    finalVal = tf.cond(index > tf.cast((n/3),tf.int64), lambda : tf.square(v1-v2), lambda : tf.square(v1+v2))
    
    
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
    outVal = sess.run(finalVal, feed_dict={finalVal: mat})
    print(outVal)
    sess.close()