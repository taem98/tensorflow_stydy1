import tensorflow as tf

sess = tf.Session()

p_holder1 = tf.placeholder(dtype=tf.float32)
p_holder2 = tf.placeholder(dtype=tf.float32)
p_holder3 = tf.placeholder(dtype=tf.float32)
 
val1 = 5
val2 = 10
val3 = 3
 
ret_val = p_holder1 * p_holder2 + p_holder3
 
feed_dict = {p_holder1: val1, p_holder2: val2, p_holder3: val3}
result = sess.run(ret_val, feed_dict=feed_dict)
 
print(result)