from sklearn.metrics import confusion_matrix as cm
import tensorflow as tf
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
import sys

class DataMode(Enum):
    RAW = 1
    FEAT = 2

def ReshapeMyData(sDAT, sLEN, sTAR, n_features, maxLength): # reshapes stacked sequences to zero padded arrays

    end_index = np.cumsum(sLEN)
    start_index = np.insert(end_index, [0], [0])
    start_index = start_index[:-1]

    SparseLikeArray = np.empty((sLEN.shape[0], maxLength, n_features))

    for i in range(sLEN.shape[0]):
        sequence = sDAT[start_index[i]:end_index[i], :]
        slotForSequence = np.zeros((maxLength, n_features))
        slotForSequence[:sequence.shape[0], :] = sequence
        SparseLikeArray[i, :, :] = slotForSequence

    sLEN = sLEN.squeeze()
    sTAR = sTAR.squeeze()

    return SparseLikeArray, sLEN, sTAR

def selu(x): # selu activation function
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def last_relevant(output, length): # takes the last output of RNN
    # srce: https://danijar.com/variable-sequence-lengths-in-tensorflow/
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def readIt(self, subset, dataType): # reads the data from files

    if(self._my_option==1): #-select raw data
        dataType+='p'
    else:
        dataType+='n'

    L = pd.read_csv(self._path_ + 'L' + subset + dataType + '.csv', delimiter=',', header=None).\
        astype(dtype='int32').as_matrix()
    T = pd.read_csv(self._path_ + 'T' + subset + dataType + '.csv', delimiter=',', header=None).\
        astype(dtype='int32').as_matrix()
    I = pd.read_csv(self._path_ + 'I' + subset + dataType + '.csv', delimiter=';', decimal=',', header=None).\
        astype(dtype='float32').as_matrix()

    if   (self._my_option == 2): #-select features
        I = I[:, 1:7]
    elif (self._my_option == 3): #-select HLfeatures
        I = I[:, 0:3]

    print(I.shape)

    return I, L, T

def CreateCSV(data_type, featKind): # creates file with statistics of each trial

    now = datetime.utcnow().strftime("%Y-%m-%d-%H.%M.%S")

    if (data_type == DataMode.RAW):
        now += 'RAW'
    else:
        now += featKind

    filename = now + '.csv'
    par2save = ['layers', 'neurons', 'LR', 'dropout', 'act_fcn', 'epoch', 'kernel', 'stdDiagV', 'stdDiagT', \
                'fin_val_acc', 'best_val_acc', 'bestEpo', 'best_tst_acc']
    empdata = np.array([[0]*len(par2save)])
    df = pd.DataFrame(empdata, columns=par2save)
    df.to_csv(filename, sep=';', decimal=',', index=False)

    return now, filename

class RNNclassifier:

    def __init__(self):
        # ---------------- Settings -------------------------------------------------------------------
        self.SaveBestModel = True
        self.TrainModel    = True
        self.SampleHParams = True

    def start(self, my_option, my_gpu_memo, my_hml):
        featKind = ''
        #---------------- training params -----------------------------------------------------------------
        n_epochs = 1000      #-number of epochs to perform
        n_min_epochs = 100   #-minimal number of epochs to perform
        n_classes = 27       #-number of classes
        minibatch_size = 256 #-minibatch size

        lowLR  = 0.0000001
        highLR = 0.05
        dropout_table = [0.5, 0.7, 0.7, 1.0] #- 0.7 with higher probability
        act_fcn_table = [tf.nn.relu, tf.nn.softsign, tf.nn.tanh, tf.nn.elu, selu]
        kernel_init_table = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.variance_scaling_initializer(), None]
        kernel_init_index = 2

        howManyLoops = my_hml
        check_acc_steps = 5
        max_checks_without_progress = 100 / check_acc_steps

        #--------------- RNN parameters -------------------------------------------------------------------
        selected_cell = tf.nn.rnn_cell.GRUCell
        layers_limit = 5
        low_layers_limit = 1
        neurons_total_limit = 100

        # ---------------- shell options ---------------------
        if my_option == 0:  # performance test
            howManyLoops = 1
            data_type = DataMode.RAW
            dataType = data_type.name
            n_epochs = 1
            n_neurons = 49
            n_layers = 2
            dropout = 1.0
            act_fcn = tf.nn.softsign
            learning_rate = 0.000888231649292
            kernel_init = tf.contrib.layers.xavier_initializer()
            self.SampleHParams = False
            DoDropout = False
        elif my_option == 1:  # data type = RAW
            data_type = DataMode.RAW
            dataType = data_type.name
        elif my_option == 2:  # data type = features
            data_type = DataMode.FEAT
            dataType  = data_type.name
            dataType += 'allIneed'
            featKind = 'F5feat'
        elif my_option == 3:  # data type = HLfeatures
            data_type = DataMode.FEAT
            dataType = data_type.name
            dataType += 'allIneed'
            featKind = 'Fpcm' # pose-cog-max (HL features)

        path_ = '../data/'
        self._path_ = path_
        self._my_option = my_option

        #--- read the data into variables (data, lengths, targets)
        I_train, L_train, T_train, = readIt(self, 'train', dataType)
        I_test,  L_test,  T_test   = readIt(self, 'test',  dataType)
        I_val,   L_val,   T_val    = readIt(self, 'val',   dataType)

        #--- get the max length of sequence and number of inputs
        n_steps = np.max([max(L_train), max(L_test), max(L_val)])
        n_inputs = I_test.shape[1]

        #--- reshape the data from stacked sequences to zero padded arrays
        I_train, L_train, T_train = ReshapeMyData(I_train, L_train, T_train, n_inputs, n_steps)
        I_test,  L_test,  T_test  = ReshapeMyData(I_test,  L_test,  T_test,  n_inputs, n_steps)
        I_val,   L_val,   T_val   = ReshapeMyData(I_val,   L_val,   T_val,   n_inputs, n_steps)

        # get number of training samples
        TrSamples = L_train.shape[0]

        # create file for storing performance of each trial
        now, filename = CreateCSV(data_type, featKind)

        my_seed = int(datetime.utcnow().strftime('%f'))
        np.random.seed(seed=my_seed)

        GlobalPrevACC = 0.0
        #-------------------------------------------------------------
        #--- loop executes as many times as the asked number of trials
        for trialNO in range(howManyLoops):

            tf.reset_default_graph()

            if(self.SampleHParams):

                learning_rate = np.random.uniform(low=lowLR, high=highLR)
                n_neurons     = np.random.randint(neurons_total_limit-10+1) + 10
                n_layers      = np.random.randint(layers_limit) + low_layers_limit
                dropout       = np.random.choice(dropout_table)
                kernel_init_index = np.random.randint(0, len(kernel_init_table))
                kernel_init   = kernel_init_table[kernel_init_index]
                act_fcn       = np.random.choice(act_fcn_table)

                # --- HEURISTIC 1
                while n_neurons * n_layers > neurons_total_limit:
                    n_neurons -= 3

                # --- HEURISTIC 2
                if (n_layers == 1):
                    dropout = 1
                    DoDropout = False
                else:
                    DoDropout = True

            if(act_fcn==tf.nn.relu):
                I_train = (I_train+1)/2
                I_val   = (I_val+1)/2
                I_test  = (I_test+1)/2

            my_kernel = 'None'
            if (kernel_init_index == 0):
                my_kernel = 'Xavier'
            elif (kernel_init_index == 1):
                my_kernel = 'He'

            # info on sampled set of Hyperparameters
            print('[Loop: {}] neurons: {}, layers: {}, dropout: {}, act: {}, LR: {}, kernel: {}'.format\
                      (trialNO, n_neurons,n_layers,dropout,act_fcn, learning_rate,my_kernel))

            if(self.TrainModel):
                tf.set_random_seed(1)

                # add data to graph
                tfI_train = tf.convert_to_tensor(I_train, dtype=tf.float32)
                tfT_train = tf.convert_to_tensor(T_train, dtype=tf.int32)
                tfL_train = tf.convert_to_tensor(L_train, dtype=tf.int32)

                train_keep_prob = tf.Variable(dropout, dtype=tf.float32)

                queue = tf.RandomShuffleQueue(capacity=L_train.shape[0], min_after_dequeue=minibatch_size+10, dtypes=[tf.float32,tf.int32,tf.int32], shapes=[[n_steps,n_inputs],[],[]])
                enqueue = queue.enqueue_many((tfI_train, tfT_train, tfL_train))
                X_batch, y_batch, l_batch = queue.dequeue_many(minibatch_size)

                X = tf.placeholder_with_default(X_batch, [None, n_steps, n_inputs], name='X')
                y = tf.placeholder_with_default(y_batch, [None], name='y')
                seq_length = tf.placeholder_with_default(l_batch, [None], name='seq_length')
                what_keep_prob = tf.placeholder_with_default(train_keep_prob, [], name='what_keep_prob')

                layers = [selected_cell(num_units=n_neurons, activation=act_fcn, kernel_initializer=kernel_init)
                                    for _ in range(n_layers)]

                if DoDropout:
                    cells_drop = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=what_keep_prob, seed=1) for cell in layers]
                    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(cells_drop)
                else:
                    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)

                #- feed the RNN
                outputsD, statesD = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32, sequence_length=seq_length)

                last = last_relevant(outputsD, seq_length)
                logits = tf.layers.dense(last, n_classes,name='dense')

                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
                loss = tf.reduce_mean(xentropy)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

                #srce: https://stackoverflow.com/a/43486487
                gradients, variables =  zip(*optimizer.compute_gradients(loss))
                gradients = [None if gradient is None else tf.clip_by_norm(gradient, 1.0) for gradient in gradients]
                training_op = optimizer.apply_gradients(zip(gradients, variables))

                correct = tf.nn.in_top_k(logits, y, 1, name='correct')
                top_result = tf.argmax(logits, 1, name='top_result')
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

                init = tf.global_variables_initializer()
                saver = tf.train.Saver(max_to_keep=5)

                #--- QUEUE Runner is here -------------------------------------------
                n_threads = 1
                qr = tf.train.QueueRunner(queue, [enqueue] * n_threads)
                tf.train.add_queue_runner(qr)

                #--- Evaluate the graph ----------------------------------------------
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = my_gpu_memo
                with tf.Session(config=config) as sess:

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(coord=coord)
                    init.run()

                    bestACC = 0.0
                    bestEpo = 0.0
                    checks_without_progress = 0
                    acc_valid = 0.0
                    rec_cnt = 0

                    for epoch in range(n_epochs):
                        for miniBatch in range(TrSamples // minibatch_size):
                            sess.run(training_op)

                        if((epoch+1) % check_acc_steps == 0 or epoch == 0):
                            acc_valid = sess.run(accuracy, feed_dict={X: I_val, y: T_val, seq_length: L_val,
                                                                      what_keep_prob: 1.0})

                            print("epoch: {0}\tValid accuracy: {1:.4f}%".format(epoch + 1, 100 * acc_valid))

                            #save the best model and acc for the whole run
                            if(self.SaveBestModel and (acc_valid*100.0>93.0) and (acc_valid*100.0>GlobalPrevACC)):
                                GlobalPrevACC = acc_valid*100.0
                                saver.save(sess, "./ModelsRNN/"+now+".ckpt", global_step=epoch+1)
                                print('Saver')

                            if acc_valid >= bestACC:
                                checks_without_progress = 0
                                bestACC = acc_valid
                                bestEpo = epoch
                                acc_test = sess.run(accuracy, feed_dict={X: I_test, y: T_test, seq_length: L_test,
                                                                         what_keep_prob: 1.0})
                                getResVal = top_result.eval(
                                    feed_dict={X: I_val, y: T_val, seq_length: L_val, what_keep_prob: 1.0})
                                getResTest = top_result.eval(
                                    feed_dict={X: I_test, y: T_test, seq_length: L_test, what_keep_prob: 1.0})
                            else:
                                checks_without_progress += 1


                            if (100 * acc_valid < 0.1):
                                rec_cnt += 1
                            else:
                                rec_cnt = 0

                            if (rec_cnt >= 4):
                                print('Gradient vanished or exploded - early stopping!')
                                break

                            # - - - early stopping - heuristics (no progress)
                            if epoch>n_min_epochs and checks_without_progress > max_checks_without_progress:
                                print("No progress - early stopping!")
                                break

                            # - - - early stopping - heuristics (too slow progress at half life)
                            if (epoch>n_min_epochs and epoch > 2* bestEpo):
                                print("No progress at half life  - early stopping!")
                                break

                    coord.request_stop()
                    coord.join(threads)

                    ConfMatVal = cm(T_val,  getResVal)
                    ConfMatTest= cm(T_test, getResTest)

                    diagonalsV = [ConfMatVal[i][i]  for i in range(n_classes)]
                    diagonalsT = [ConfMatTest[i][i] for i in range(n_classes)]

                    sum_row_Val  = [sum(ConfMatVal[i][:])  for i in range(n_classes)]
                    sum_row_Test = [sum(ConfMatTest[i][:]) for i in range(n_classes)]

                    stand_diagV = 100.0*np.std(np.vstack(diagonalsV) / np.vstack(sum_row_Val))
                    stand_diagT = 100.0*np.std(np.vstack(diagonalsT) / np.vstack(sum_row_Test))

                    data2save = np.array(
                        [[n_layers, n_neurons, learning_rate, dropout, act_fcn, epoch+1,\
                          my_kernel, stand_diagV, stand_diagT,  acc_valid, bestACC, bestEpo+1, acc_test]])

                    statsFile = pd.read_csv(filename, sep=';', decimal=',')
                    headers = statsFile.columns

                    statsArray = statsFile.as_matrix()
                    v_ref = float(np.max(statsArray[:,10])) # save confusion matrix only for the best score on valid set
                    if(bestACC > v_ref):
                        df = pd.DataFrame(data=ConfMatVal)
                        df.to_csv('CMv' + filename, sep=';', decimal=',', index=False, header=False)
                        df = pd.DataFrame(data=ConfMatTest)
                        df.to_csv('CMt' + filename, sep=';', decimal=',', index=False, header=False)

                    temporaryDFwith1Record = pd.DataFrame(data2save, columns=headers)
                    statsFile = statsFile.append(temporaryDFwith1Record)
                    statsFile.to_csv(filename, sep=';', decimal=',', index=False)

if __name__ == '__main__':
    my_option   = int(  sys.argv[1])
    my_gpu_memo = float(sys.argv[2])/100.
    my_hml      = int(  sys.argv[3])
    myRNNtrials =  RNNclassifier()
    myRNNtrials.start(my_option, my_gpu_memo, my_hml)
