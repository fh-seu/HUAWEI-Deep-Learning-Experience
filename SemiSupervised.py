## Wrapper for execution in PySpark - First block
def model_wrapper():
    
    from hops import hdfs
    from tempfile import TemporaryFile
    import numpy as np
    import os
    import time
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from keras.utils import to_categorical
    from sklearn.preprocessing import OneHotEncoder
    
    ## Early christmas gift from the mentors :)
    def open_hdfs(path):
        temp_file = TemporaryFile()
        with hdfs.get_fs().open_file(path, 'r') as data_file:
            temp_file.write(data_file.read())
            temp_file.seek(0)
        return temp_file

    def loadnp(path):
        temp_file = open_hdfs(path)
        data = np.load(temp_file)
        return data

    def save(data, path):
        with hdfs.get_fs().open_file(path, 'w') as data_file:
            data_file.write(data)

    def savetf(temp_file, path, close=True):
        with hdfs.get_fs().open_file(path, 'w') as data_file:
            temp_file.seek(0)
            data_file.write(temp_file.read())
        if close:
            temp_file.close()
            
    def savenp(data, path):
        temp_file = TemporaryFile()
        np.save(temp_file, data)
        savetf(temp_file, path)
    
    def log(message):
        hdfs.log(str(message))
    
    def vectorize(sequences,dimension=10):
        results = np.zeros((len(sequences),dimension))
        for i,sequence in enumerate(sequences):
            results[int(i),int(sequence)] = 1
        return results
    
    ## Load datasets
    data_path = 'hdfs://10.0.104.196:8020/Projects/Mentors/hackathon_AB/'
    
    # Unlabled images
    unlabled = loadnp(data_path + 'A.npy')
    
    # Full labled training set
    with loadnp(data_path + 'B_train_full.npz') as train:
        X_train, y_train = train['X'], train['y']
    
    # Labled trainig set provided in the same format as the final evaluation data set
    # A dict of python tuples {num_samples0: (x0, y0), ..., num_samplesN: (xN, yN)}
    with loadnp(data_path + 'B_train.npz') as train:
        train_eval = train['data'].item()
    
    # Test set
    with loadnp(data_path + 'B_test.npz') as test:
        X_test, y_test = test['X'], test['y']
    
    ##reshape data into use
    ##data_unlabled = unlabled.astype('float32')/255-0.5
    data_unlabled = unlabled.reshape(100000,32*32*3)#unlabled data
    
    ##data_labled_x = X_train.astype('float32')/255-0.5
    data_labled_x = X_train.reshape(4000,32*32*3)#labled data image
    data_labled_y = to_categorical(y_train)

    
    ##test_x = X_test.astype('float32')/255-0.5
    test_x = X_test.reshape(1000,32*32*3)#labled data image
    test_y = to_categorical(y_test)

    
    ##
    ## Your awesome ideas goes here
    
    ##################################################################################################################################
    def flat(x, output_name = "flatten"):
        """ takes a tensor of rank > 2 and return a tensor of shape [?,n]"""
        shape = x.get_shape().as_list()
    #    n = reduce(lambda x,y:x*y, shape[1:])
        n = np.prod(shape[1:])
        x_flat = tf.reshape(x, [-1,n])
        return tf.identity(x_flat, name = output_name)

    def max_pool_2x2(x, output_name = "max_pool"):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name = output_name)


    def unpool(value, output_shape, name='unpool'):
        """N-dimensional version of the unpooling operation from
        https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

        :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
        :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
        """
        with tf.name_scope(name):

            kernel_shape = (2,2)
            num_channels = value.get_shape()[-1]
            input_dtype_as_numpy = value.dtype.as_numpy_dtype()
            kernel_rows, kernel_cols = kernel_shape

            # build kernel
            kernel_value = np.zeros((kernel_rows, kernel_cols, num_channels, num_channels), dtype=input_dtype_as_numpy)
            kernel_value[0, 0, :, :] = np.eye(num_channels, num_channels)
            kernel = tf.constant(kernel_value)

            # do the un-pooling using conv2d_transpose
            out = tf.nn.conv2d_transpose(value,
                                            kernel,
                                            output_shape=output_shape,
                                            strides=(1, kernel_rows, kernel_cols, 1),
                                            padding='VALID')

        return out


    def run_layer(x, layer_spec, output_name):
        """ x: a tensor
            layer_spec: a dictionary containing the type of the layer (dense or conv2d) 
                        and the shape of the kernel ([a,b] for dense layer, [a,b,c,d] for conv2d
            return the output of the layer"""
        if layer_spec["type"] == "dense":
            W = tf.get_variable("W", layer_spec["kernel_shape"], initializer= tf.random_normal_initializer())
            z = tf.matmul(x,W, name=output_name)
        elif layer_spec["type"] == "conv2d":
            filters = tf.get_variable("W", layer_spec["kernel_shape"], initializer= tf.random_normal_initializer())
            z = tf.nn.conv2d(x, filters, strides=[1, 1, 1, 1], padding='SAME', name=output_name)
        return z


    def run_transpose_layer(x, layer_spec, output_name="inverse", output_shape=None):
        """ Performs the transposed operation of the layer described by layer_spec.
            x: a tensor whose shape is the same as the output of the layer
            layer_spec: dictionary describing the layer
            output shape: shape of the output tensor (only used for a "flat" layer)
            return a tensor whose shape is the same as the input of the layer
            """
        if layer_spec["type"] == "dense":
            W = tf.get_variable("W", layer_spec["kernel_shape"][::-1], initializer= tf.random_normal_initializer())
            z = tf.matmul(x,W, name=output_name)

        elif layer_spec["type"] == "conv2d":
            kernel_shape = layer_spec["kernel_shape"]
            kernel_shape = kernel_shape[0:2] + kernel_shape[-1:1:-1]
            filters = tf.get_variable("W", kernel_shape, initializer= tf.random_normal_initializer())
            z = tf.nn.conv2d(x, filters, strides=[1, 1, 1, 1], padding='SAME', name=output_name)

        elif layer_spec["type"] == "flat":
            z = tf.reshape(x, output_shape, name=output_name)

        elif layer_spec["type"] == "max_pool_2x2":
            z = unpool(x, output_shape, name=output_name)
        return z
    
    # Definition of the layers of the feedforward network. Four types of layers are avaiable :
    # conv2d, max_pool_2x2, dense, flat.
    # The convolution layers have always a stide of 1, and padding is set as "same".
    # The kernel shapes should be carefully defined to avoid non-compatibilities 
    # between the layers and their inputs.

    layers = [ {"type": "dense", "kernel_shape": [3072, 1000]},
               {"type": "dense", "kernel_shape": [1000, 500]},
               {"type": "dense", "kernel_shape": [500, 250]},
               {"type": "dense", "kernel_shape": [250, 250]},
               {"type": "dense", "kernel_shape": [250, 250]},
               {"type": "dense", "kernel_shape": [250, 10]}]

    #layers = [{"type": "conv2d", "kernel_shape": [5,5,1,32]}, # 28x28x1 -> 28x28x32
    #           {"type": "max_pool_2x2"},                      # 28x28x32 -> 14x14x32
    #           {"type": "conv2d", "kernel_shape": [5,5,32,64]},# 14x14x32 -> 14x14x64
    #           {"type": "max_pool_2x2"},                       # 14x14x64 -> 7x7x64
    #           {"type": "flat"},                               # 7x7x64 -> 3136
    #           {"type": "dense", "kernel_shape": [7*7*64, 1024]},# 3136 -> 1024
    #           {"type": "dense", "kernel_shape": [1024, 10]}]    # 1024 -> 10

    # hyperparameters that denote the importance of each layer
    denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

    checkpoint_dir = "checkpoints/"

    image_shape = [3072]

    num_epochs = 5
    starter_learning_rate = 0.01
    decay_after = 15  # epoch after which to begin learning rate decay
    batch_size = 4000
    num_labeled = 4000

    noise_std = 0.3  # scaling factor for noise used in corrupted encoder

    #==================================================================================================

    L = len(layers)  # number of layers

    tf.reset_default_graph()

    # feedforward_inputs (FFI): inputs for the feedforward network (i.e. the encoder).
    # Should contain the labeled images during train time and test images during test time.
    feedforward_inputs = tf.placeholder(tf.float32, shape=(None, np.prod(image_shape)), name = "FFI")

    # autoencoder_inputs (AEI): inputs for the autoencoder (encoder + decoder).
    # Should contain the unlabeled images during train time.
    autoencoder_inputs = tf.placeholder(tf.float32, shape=(None, np.prod(image_shape)), name = "AEI")

    outputs = tf.placeholder(tf.float32)# labels for the labeled images
    training = tf.placeholder(tf.bool) # If training or not

    FFI = tf.reshape(feedforward_inputs, [-1] + image_shape)
    AEI = tf.reshape(autoencoder_inputs, [-1] + image_shape)

    #==================================================================================================
    # Batch normalization functions
    ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
    bn_assigns = []  # this list stores the updates to be made to average mean and variance


    def update_batch_normalization(batch, output_name = "bn", scope_name = "BN"):
        dim = len(batch.get_shape().as_list())
        mean, var = tf.nn.moments(batch, axes=list(range(0,dim-1)))
        # Function to be used during the learning phase. Normalize the batch and update running mean and variance.
        with tf.variable_scope(scope_name, reuse= tf.AUTO_REUSE):
            running_mean = tf.get_variable("running_mean", mean.shape, initializer= tf.constant_initializer(0))
            running_var = tf.get_variable("running_var", mean.shape, initializer= tf.constant_initializer(1))

        assign_mean = running_mean.assign(mean)
        assign_var = running_var.assign(var)
        bn_assigns.append(ewma.apply([running_mean, running_var]))

        with tf.control_dependencies([assign_mean, assign_var]):
            z = (batch - mean) / tf.sqrt(var + 1e-10)
            return tf.identity(z, name = output_name)


    def batch_normalization(batch, mean=None, var=None, output_name = "bn"):
        dim = len(batch.get_shape().as_list())
        mean, var = tf.nn.moments(batch, axes=list(range(0,dim-1)))
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0])
        z = (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
        return tf.identity(z, name = output_name)

    #______________________________________________________________________________________
    # Encoder

    def encoder_bloc(h, layer_spec, noise_std, update_BN, activation):
        # Run the layer
        z_pre = run_layer(h, layer_spec, output_name="z_pre")

        # Compute mean and variance of z_pre (to be used in the decoder)
        dim = len(z_pre.get_shape().as_list())
        mean, var = tf.nn.moments(z_pre, axes=list(range(0,dim-1)))
        mean = tf.identity(mean, name = "mean")
        var = tf.identity(var, name = "var")

        # Batch normalization
        def training_batch_norm():
            if update_BN:
                z = update_batch_normalization(z_pre)
            else:
                z = batch_normalization(z_pre)
            return z

        def eval_batch_norm():
            with tf.variable_scope("BN", reuse= tf.AUTO_REUSE):
                mean = ewma.average(tf.get_variable("running_mean", shape = z_pre.shape[-1]))
                var = ewma.average(tf.get_variable("running_var", shape = z_pre.shape[-1]))
            z = batch_normalization(z_pre, mean, var)
            return z

        # Perform batch norm depending to the phase (training or testing)
        z = tf.cond(training, training_batch_norm, eval_batch_norm)
        z += tf.random_normal(tf.shape(z)) * noise_std
        z = tf.identity(z, name = "z")

        # Center and scale plus activation
        size = z.get_shape().as_list()[-1]
        beta = tf.get_variable("beta", [size], initializer= tf.constant_initializer(0))
        gamma = tf.get_variable("gamma", [size], initializer= tf.constant_initializer(1))

        h = activation(z*gamma + beta)
        return tf.identity(h, name = "h")


    def encoder(h, noise_std, update_BN):
        # Perform encoding for each layer
        h += tf.random_normal(tf.shape(h)) * noise_std
        h = tf.identity(h, "h0")

        for i, layer_spec in enumerate(layers):
            with tf.variable_scope("encoder_bloc_" + str(i+1), reuse = tf.AUTO_REUSE):
                # Create an encoder bloc if the layer type is dense or conv2d
                if layer_spec["type"] == "flat":
                    h = flat(h, output_name="h")
                elif layer_spec["type"] == "max_pool_2x2":
                    h = max_pool_2x2(h, output_name="h")
                else:
                    if i == L-1:
                        activation = tf.nn.softmax # Only for the last layer
                    else:
                        activation = tf.nn.relu
                    h = encoder_bloc(h, layer_spec, noise_std, update_BN = update_BN, activation = activation)

        y = tf.identity(h, name = "y")
        return y

    #__________________________________________________________________________________________________________
    # Decoder

    def g_gauss(z_c, u, output_name="z_est", scope_name = "denoising_func"):
    #    gaussian denoising function proposed in the original paper
        size = u.get_shape().as_list()[-1]
        wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
        with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE):
            a1 = wi(0., 'a1')
            a2 = wi(1., 'a2')
            a3 = wi(0., 'a3')
            a4 = wi(0., 'a4')
            a5 = wi(0., 'a5')

            a6 = wi(0., 'a6')
            a7 = wi(1., 'a7')
            a8 = wi(0., 'a8')
            a9 = wi(0., 'a9')
            a10 = wi(0., 'a10')

            mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
            v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

            z_est = (z_c - mu) * v + mu
        return tf.identity(z_est, name = output_name)


    def decoder_bloc(u, z_corr, mean, var, layer_spec=None):
        # Performs the decoding operations of a corresponding encoder bloc
        # Denoising
        z_est = g_gauss(z_corr, u)

        z_est_BN = (z_est - mean)/tf.sqrt(var + tf.constant(1e-10))
        z_est_BN = tf.identity(z_est_BN, name = "z_est_BN")

        # run transposed layer
        if layer_spec is not None:
            u = run_transpose_layer(z_est, layer_spec)
            u = batch_normalization(u, output_name = "u")

        return u, z_est_BN

    # ========================================================================================================
    # Graph building
    print("===  Building graph ===")

    # Encoder
    with tf.name_scope("FF_clean"):
        FF_y = encoder(FFI, 0, update_BN = False) # output of the clean encoder. Used for prediction
    with tf.name_scope("FF_corrupted"):
        FF_y_corr = encoder(FFI, noise_std, update_BN = False) # output of the corrupted encoder. Used for training.

    with tf.name_scope("AE_clean"):
        AE_y = encoder(AEI, 0, update_BN = True) # Clean encoding of unlabeled images
    with tf.name_scope("AE_corrupted"):
        AE_y_corr = encoder(AEI, noise_std, update_BN = False) # corrupted encoding of unlabeled images


    #__________________________________________________________________________________________________________
    # Decoder
        # Function used to get a tensor from encoder
    get_tensor = lambda input_name, num_encoder_bloc, name_tensor: tf.get_default_graph().get_tensor_by_name(
                input_name + "/encoder_bloc_" + str(num_encoder_bloc) + "/" + name_tensor + ":0")


    d_cost = []
    u = batch_normalization(AE_y_corr, output_name="u_L") 
    for i in range(L,0,-1):
        layer_spec = layers[i-1]

        with tf.variable_scope("decoder_bloc_" + str(i), reuse = tf.AUTO_REUSE):
            # if the layer is max pooling or "flat", the transposed layer is run without creating a decoder bloc.
            if layer_spec["type"] in ["max_pool_2x2", "flat"]:
                h = get_tensor("AE_corrupted",i-1,"h")
                output_shape = tf.shape(h)
                u = run_transpose_layer(u, layer_spec, output_shape=output_shape)
            else:
                z_corr, z = [get_tensor("AE_corrupted",i,"z"), get_tensor("AE_clean",i,"z")]
                mean, var = [get_tensor("AE_clean",i,"mean"), get_tensor("AE_clean",i,"var")]

                u, z_est_BN = decoder_bloc(u, z_corr, mean, var, layer_spec=layer_spec)
                d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[i])


    # last decoding step
    with tf.variable_scope("decoder_bloc_0", reuse = tf.AUTO_REUSE):
        z_corr = tf.get_default_graph().get_tensor_by_name("AE_corrupted/h0:0")
        z = tf.get_default_graph().get_tensor_by_name("AE_clean/h0:0")
        mean, var = tf.constant(0.0), tf.constant(1.0)
        u, z_est_BN = decoder_bloc(u, z_corr, mean, var)
        d_cost.append((tf.reduce_mean(tf.square(z_est_BN - z))) * denoising_cost[0])

    
    # Loss and accuracy
    u_cost = tf.add_n(d_cost) # decoding cost
    corr_pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(FF_y_corr), 1))  # supervised cost
    clean_pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(FF_y), 1))  # cost used for prediction

    loss = corr_pred_cost + u_cost  # total cost

    correct_prediction = tf.equal(tf.argmax(FF_y, 1), tf.argmax(outputs, 1))  # number of correct predictions
    log(correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)
    log(accuracy)


    # Optimization setting
    learning_rate = tf.Variable(starter_learning_rate, trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # add the updates of batch normalization statistics to train_step
    bn_updates = tf.group(*bn_assigns)
    with tf.control_dependencies([train_step]):
        train_step = tf.group(bn_updates)


    n = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print(str(n) + " trainable parameters")


    #========================================================================================================================
        # Learning phase
    print("===  Loading Data ===")
    ##data = input_data.read_data_sets("MNIST_data", n_labeled=num_labeled, one_hot=True)



    num_examples = 100000  ########################################
    num_iter = (num_examples//batch_size) * num_epochs  # number of loop iterations

    print("===  Starting Session ===")
    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    print("=== Training ===")


   



    i_iter = 0
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # get latest checkpoint (if any)
    if ckpt and ckpt.model_checkpoint_path:
        # if checkpoint exists, restore the parameters and set epoch_n and i_iter
        saver.restore(sess, ckpt.model_checkpoint_path)
        epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
    #    i_iter = (epoch_n+1) * (num_examples/batch_size)
        i_iter = (epoch_n+1) * (num_examples//batch_size) #py3 adapt
        print("Restored Epoch ", epoch_n)
    else:
        # no checkpoint exists. create checkpoints directory if it does not exist.
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        init = tf.global_variables_initializer()
        sess.run(init)

    for i in range(i_iter, num_iter):
        a=np.arange(4000)
        np.random.shuffle(a)
        labeled_images=data_labled_x[a[0:1000]]
        labels=data_labled_y[a[0:1000]]
        
        b=np.arange(4000)
        np.random.shuffle(b)
        unlabeled_images=data_unlabled[b[0:1000]]

       ## labeled_images, labels, unlabeled_images = data.train.next_batch(batch_size)
    
        
        T = time.time()
        sess.run(train_step, feed_dict={feedforward_inputs: labeled_images, 
                                        outputs: labels, 
                                        autoencoder_inputs: unlabeled_images, 
                                        training: True})
        T = time.time() - T

    #    msg = "\rIterations : {:} step time : {:}".format(i,T)
    #    sys.stdout.write(msg)
    #    sys.stdout.flush()

        if (i > 1) and ((i+1) % (num_iter/num_epochs) == 0):
            # Compute train loss, train accuracy and test accuracy for each epoch
            epoch_n = i//(num_examples//batch_size)

            train_loss = sess.run(loss, feed_dict={feedforward_inputs: labeled_images, 
                                                   outputs: labels, 
                                                   autoencoder_inputs: unlabeled_images, 
                                                   training: False})
            log(train_loss)

            train_acc  = sess.run(accuracy, feed_dict={feedforward_inputs: labeled_images, 
                                                   outputs: labels,
                                                   autoencoder_inputs: unlabeled_images,
                                                   training: False})

            log(train_acc)


            test_acc = sess.run(accuracy, feed_dict={feedforward_inputs: test_x, 
                                                     outputs: test_y,
                                                     autoencoder_inputs: unlabeled_images,
                                                     training: False})
            log(test_acc)

          #  print("\nEpoch ", str(int(epoch_n+0.1)), ": train loss = {:.5f}, train accuracy = {:.3}, test accuracy = {:.3}".format(train_loss,train_acc,test_acc))

            if (epoch_n+1) >= decay_after:
                # decay learning rate
                # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
                ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
                ratio = max(0, ratio / (num_epochs - decay_after))
                sess.run(learning_rate.assign(starter_learning_rate * ratio))

            saver.save(sess, checkpoint_dir + 'model.ckpt', epoch_n)


    final_accuracy = sess.run(accuracy, feed_dict={feedforward_inputs: test_x, 
                                              outputs: test_y, 
                                              training: False})
    print("Final Accuracy: ", final_accuracy, "%")
    
    log(final_accuracy)

    
    ##################################################################################################################################
    ##
    
    # Logging
    log("Hello machine learning!")
    
 
    # In the final evaluation save predictions to HDFS
    # Replace 't' in 'eval_gt' with your group number
    # savenp(predictions, hdfs.project_path() + 'eval_gt/predictions.npy')


## Launch experiment - Second block
from hops import experiment
experiment.launch(model_wrapper, name='Our winning model', local_logdir=True)
