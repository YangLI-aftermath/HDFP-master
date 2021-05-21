import tensorflow as tf
x = tf.random.normal([16,1,1,256])
w = tf.random.normal([4,4,256,512])
y = tf.nn.conv2d_transpose(x,w,output_shape=4,strides=4,padding='SAME')
print('y',y)