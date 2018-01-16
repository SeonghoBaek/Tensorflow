import tensorflow as tf

from layer_utils import get_deconv2d_output_dims


def conv(input, name, filter_dims, stride_dims, padding='SAME',
         non_linear_fn=tf.nn.relu):
    input_dims = input.get_shape().as_list()
    print(name, 'in', input_dims)
    assert(len(input_dims) == 4)    # batch_size, height, width, num_channels_in
    assert(len(filter_dims) == 3)   # height, width and num_channels out
    assert(len(stride_dims) == 2)   # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    # Define a variable scope for the conv layer
    with tf.variable_scope(name) as scope:
        # Create filter weight variable
        conv_weight = tf.Variable(tf.random_normal([filter_h, filter_w, num_channels_in, num_channels_out], stddev=0.1, dtype=tf.float32))
        # Create bias variable
        conv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))
        # Define the convolution flow graph
        map = tf.nn.conv2d(input, conv_weight, strides=[1, stride_h, stride_w, 1], padding=padding)
        # Add bias to conv output
        map = tf.nn.bias_add(map, conv_bias)

        # Apply non-linearity (if asked) and return output
        activation = non_linear_fn(map)

        print(name, 'out', activation.get_shape().as_list())
        return activation
        pass


def deconv_v2(input, name, filter_dims, output_dims, padding='SAME', non_linear_fn=tf.nn.relu):
    input_dims = input.get_shape().as_list()
    print(name, 'in', input_dims)
    assert (len(input_dims) == 4)  # batch_size, height, width, num_channels_in
    assert (len(filter_dims) == 3)  # height, width and num_channels out
    assert (len(output_dims) == 3)

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    batch_size = input_dims[0]
    output_dim = [batch_size] + output_dims

    # Define a variable scope for the deconv layer
    with tf.variable_scope(name) as scope:
        # Create filter weight variable
        # Note that num_channels_out and in positions are flipped for deconv.
        deconv_weight = tf.Variable(
            tf.random_normal([filter_h, filter_w, num_channels_out, num_channels_in], stddev=0.1, dtype=tf.float32))
        # Create bias variable
        deconv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))

        # Define the deconv flow graph
        map = tf.nn.conv2d_transpose(input, deconv_weight, output_dim, strides=[1, 2, 2, 1], padding=padding)

        # Add bias to deconv output
        map = tf.nn.bias_add(map, deconv_bias)

        # Apply non-linearity (if asked) and return output
        activation = non_linear_fn(map)

        print(name, 'out', activation.get_shape().as_list())
        return activation
        pass


def deconv(input, name, filter_dims, stride_dims, padding='SAME',
           non_linear_fn=tf.nn.relu):
    input_dims = input.get_shape().as_list()
    print(name, 'in', input_dims)
    assert(len(input_dims) == 4)    # batch_size, height, width, num_channels_in
    assert(len(filter_dims) == 3)   # height, width and num_channels out
    assert(len(stride_dims) == 2)   # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims
    # Let's step into this function
    output_dims = get_deconv2d_output_dims(input_dims,
                                           filter_dims,
                                           stride_dims,
                                           padding)

    # Define a variable scope for the deconv layer
    with tf.variable_scope(name) as scope:
        # Create filter weight variable
        # Note that num_channels_out and in positions are flipped for deconv.
        deconv_weight = tf.Variable(tf.random_normal([filter_h, filter_w, num_channels_out, num_channels_in], stddev=0.1, dtype=tf.float32))
        # Create bias variable
        deconv_bias = tf.Variable(tf.zeros([num_channels_out], dtype=tf.float32))
        
        # Define the deconv flow graph
        map = tf.nn.conv2d_transpose(input, deconv_weight, output_dims, strides=[1, stride_h, stride_w, 1], padding=padding)
        
        # Add bias to deconv output
        map = tf.nn.bias_add(map, deconv_bias)
        
        # Apply non-linearity (if asked) and return output
        activation = non_linear_fn(map)

        print(name, 'out', activation.get_shape().as_list())
        return activation
        pass


def max_pool(input, name, filter_dims, stride_dims, padding='SAME'):
    assert(len(filter_dims) == 2) # filter height and width
    assert(len(stride_dims) == 2) # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims
    
    # Define the max pool flow graph and return output
    pool = tf.nn.max_pool(input, ksize=[1, filter_h, filter_w, 1], strides=[1, stride_h, stride_w, 1], padding=padding)

    return pool
    pass


def fc(input, name, out_dim, non_linear_fn=tf.nn.relu):
    assert(type(out_dim) == int)

    # Define a variable scope for the FC layer
    with tf.variable_scope(name) as scope:
        input_dims = input.get_shape().as_list()
        print(name, 'in', input_dims)

        # the input to the fc layer should be flattened
        if len(input_dims) == 4:
            # for eg. the output of a conv layer
            batch_size, input_h, input_w, num_channels = input_dims
            # ignore the batch dimension
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input, [batch_size, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input

        # Create weight variable
        fc_weight = tf.Variable(tf.random_normal([in_dim, out_dim], stddev=0.1, dtype=tf.float32))

        # Create bias variable
        fc_bias = tf.Variable(tf.zeros([out_dim], dtype=tf.float32))
        
        # Define FC flow graph
        output = tf.add(tf.matmul(flat_input, fc_weight), fc_bias)

        print(name, 'out', output.get_shape().as_list())
        # Apply non-linearity (if asked) and return output
        if non_linear_fn is None:
            return output
        else:
            activation = non_linear_fn(output)

        return activation
        pass
