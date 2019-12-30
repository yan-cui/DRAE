import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import numpy as np
import scipy.io as sio
import h5py
import os
import time
import math
import sys

print(len(sys.argv))
print([sys.argv[i] for i in range(len(sys.argv))])
if len(sys.argv) < 15:
    print('Usage: python rnn_train.py DATASET[Q1/Q3] TASK MAXSUB STARTSUB HIDDENSIZE NLAYERS FEATURESIZ CELL[LSTM/GRU/BASIC] EPOCH LR LRDEPOCH LRDRATIO SHUFFLE[0/1] MARK')
    exit()

# model parameters
DATASET = sys.argv[1]
TASK = sys.argv[2]
MAXSUB = int(sys.argv[3])
SUBIDX = int(sys.argv[4])
HIDDENSIZE = int(sys.argv[5])
NLAYERS = int(sys.argv[6])
FEATURESIZE = int(sys.argv[7])
RNNCELL = sys.argv[8] # LSTM/GRU/BASIC
MAXEPOCH = int(sys.argv[9])
learning_rate = float(sys.argv[10])
LRDEPOCH = int(sys.argv[11])
LRDRATIO = int(sys.argv[12])
SHUFFLE = int(sys.argv[13])
MARK = sys.argv[14]
dropout_pkeep = 1.0
if DATASET == 'Q1':
    SIG_SIZE = 223945 # For HCP68 Q1 data, this size is fixed
elif DATASET == 'Q3':
    if TASK == 'MOTOR':
        SIG_SIZE = 244341 # For HCP68 tfMRI Q3 MOTOR data, this size is fixed
    elif TASK == "SOCIAL":
        SIG_SIZE = 244329 # For HCP68 tfMRI Q3 SOCIAL data, this size is fixed
BATCHSIZE = 1 # For now batch_size is fixed to 1, and there're some realated hard code yet, be careful!

STITYPE = 'taskdesign' # 'init_D' for another option
NORM = True # normalization

FASTVALID = False # take init_D as brain signals for fast loading to check code
TRAINING = False
TEST= True
PLAYCKPT = '' # if TRAINING is not set, PLAYCKPT must be set for test
PLAYCKPT = 'DRAE-Q1-MOTOR-68sub1-68-NTS-L32x3-F16-lr0.004-e40-lrd15r4-2720'
EXTRAMARK = '' # extra mark for test

SAVECKPT = True # save a checkpoint
SAVESMM = True # save summaries
if FASTVALID:
    SAVECKPT = False # no saving when doing fast valid
    SAVESMM = False
    SIG_SIZE = 1000

# task related parameters
if TASK == 'MOTOR':
    SEQ_LEN = 284
    STISIZE = 6
elif TASK == 'WM':
    SEQ_LEN = 405
    STISIZE = 4
elif TASK == 'EMOTION':
    SEQ_LEN = 176
    STISIZE = 2
elif TASK == 'LANGUAGE':
    SEQ_LEN = 316
    STISIZE = 2
elif TASK == 'GAMBLING':
    SEQ_LEN = 253
    STISIZE = 2
elif TASK == 'RELATIONAL':
    SEQ_LEN = 232
    STISIZE = 2
elif TASK == 'SOCIAL':
    SEQ_LEN = 274
    STISIZE = 2
else:
    print('!!!!!! Error: invalid TASK:', TASK)
    exit()
TRAINLEN = SEQ_LEN

# data path
DATA_BASE = '/home/cubic/cuiyan/data/HCP/'
STI_DIR = 'HCP_Init_D/'
STI_PREFIX = 'InitD_'+TASK+'_stimulus_hrf_Dsize_1000_sub'
DESIGN_DIR = 'HCP_Taskdesign_allcontrast/'
DESIGN_PREFIX = TASK + '_taskdesign_'
SIG_DIR = 'Whole_b_signals_std_uniform/'+TASK+'/'
if DATASET == 'Q1':
    SIG_DIR_H5 = 'Whole_b_signals_std_uniform_h5norm/'+TASK+'/'
elif DATASET == 'Q3':
    SIG_DIR_H5 = 'Q3_Whole_b_signals_std_uniform_h5norm/'+TASK+'/'
SIG_SUFFIX = '_'+TASK+'_whole_b_signals'

def gen_exp_mark(dataset, task, sbj_idx, max_sub, sti_type, norm, rnn_cell, hidden_size, nlayers, lrate, maxepoch, lrdepoch, lrdratio, pkeep, mark, shuffle):
    exp_mark = 'DRAE'
    exp_mark = exp_mark + '-' + dataset + '-'
    exp_mark = exp_mark + task + '-'
    if max_sub > 1:
        exp_mark = exp_mark + str(max_sub) + 'sub' + str(sbj_idx) + '-' + str(sbj_idx+max_sub-1) + '-'
    else:
        exp_mark = exp_mark + 'sub' + str(sbj_idx) + '-'
    if norm:
        exp_mark = exp_mark + 'N'
    if sti_type == 'init_D':
        exp_mark = exp_mark + 'I'
    elif sti_type == 'taskdesign':
        exp_mark = exp_mark + 'T'
    if shuffle == True:
        exp_mark = exp_mark + "S"
    if rnn_cell == 'LSTM':
        exp_mark = exp_mark + '-L' + str(hidden_size) + 'x' + str(nlayers)
    elif rnn_cell == 'GRU':
        exp_mark = exp_mark + '-G' + str(hidden_size) + 'x' + str(nlayers)
    elif rnn_cell == 'BASIC':
        exp_mark = exp_mark + '-B' + str(hidden_size) + 'x' + str(nlayers)
    exp_mark = exp_mark + '-F' + str(FEATURESIZE)
    exp_mark = exp_mark + '-lr' + str(lrate) + '-e' + str(maxepoch) + '-lrd' + str(lrdepoch) + 'r' + str(lrdratio)
    if mark != '':
        exp_mark = exp_mark + '-' + mark
    if pkeep != 1.0:
        exp_mark = exp_mark + '-pk' + str(pkeep)
    #exp_mark = exp_mark + '-'
    return exp_mark
    
def normalization(x):
    u = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    m, n = x.shape
    x0 = np.zeros([m,n])
    for i in range(n):
        x0[:,i] = (x[:,i] - u[i]) / sigma[i]
    return x0

def list_training_data_h5(sbj_idx, max_sub):
    flist = os.listdir(DATA_BASE + SIG_DIR_H5)
    flist.sort(key = lambda x:int(x[:-(20+len(TASK))]))
    print('Total subjects count:', len(flist), 'requested:', max_sub)

    flist = flist[sbj_idx - 1 : sbj_idx - 1 + max_sub]
    #for i in range(len(flist)):
    #    print('[' + str(i+1) +  ']', flist[i])
    return flist

def load_training_data_h5(flist, idx, seq_len, sti_size, sig_size, fast_valid = False, sti = 'init_D', norm = True):
    stimulus = np.zeros([1, seq_len, sti_size])
    signals = np.zeros([1, seq_len, sig_size])

    i = flist[idx][0:len(flist[idx])-20-len(TASK)]
    #print('['+ str(idx+1) +'] Loading training data subject ' + i + '...')
    if sti == 'init_D':
        x = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + i + '.txt')
    elif sti == 'taskdesign':
        #x = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + i + '.mat')['Normalized_'+TASK+'_taskdesign']
        x = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + '1.mat')['Normalized_'+TASK+'_taskdesign']
    else:
        x = np.zeros([seq_len, sti_size])
        print('Error: invalid stimulus type: ' + sti)
    x = x[:,0:sti_size]
    stimulus[0,:,:] = x
    if fast_valid:
        y_ = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + i + '.txt')
    else:
        f = h5py.File(DATA_BASE + SIG_DIR_H5 + i + SIG_SUFFIX + '.h5', 'r')
        y_ = f['data'][:]
        f.close()
    signals[0,:,:] = y_
    return stimulus, signals

def load_training_data(sbj_idx, max_sub, seq_len, sti_size, sig_size, fast_valid = False, sti = 'init_D', norm = True):
    stimulus = np.zeros([max_sub, seq_len, sti_size])
    signals = np.zeros([max_sub, seq_len, sig_size])

    flist = os.listdir(DATA_BASE + SIG_DIR)
    flist.sort(key = lambda x:int(x[:-(21+len(TASK))]))
    #print(flist)
    print('Total subjects count:', len(flist), 'requested:', max_sub)

    for idx in range (max_sub):
        i = flist[idx + sbj_idx - 1][0:len(flist[idx + sbj_idx - 1])-21-len(TASK)]
        print('['+ str(idx+1) +'] Loading training data subject ' + i + '...')
        if sti == 'init_D':
            x = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + i + '.txt')
        elif sti == 'taskdesign':
            x = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + i + '.mat')['Normalized_'+TASK+'_taskdesign']
        else:
            x = np.zeros([seq_len, sti_size])
            print('Error: invalid stimulus type: ' + sti)
        x = x[:,0:sti_size]
        stimulus[idx,:,:] = x
        if fast_valid:
            y_ = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + i + '.txt')
        else:
            y_ = np.loadtxt(DATA_BASE + SIG_DIR + i + SIG_SUFFIX + '.txt')
        if norm:
            y_ = normalization(y_)
        signals[idx,:,:] = y_
    return stimulus, signals

def load_play_data(test_idx, seq_len, sti_size, sti_idx, sti = 'init_D'):
    i = test_idx
    test_stimulus = np.zeros([1, seq_len, sti_size])

    #print('Loading test stimulus subject ' + str(i) + '...')
    if sti == 'init_D':
        xin = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + str(i) + '.txt')
    elif sti == 'taskdesign':
        xin = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + str(i) + '.mat')['Normalized_'+TASK+'_taskdesign']
    else:
        xin = np.zeros([seq_len, sti_size])
        print('Error: invalid stimulus type: ' + sti)

    #x = np.zeros([seq_len, sti_size])
    x = -np.ones([seq_len, sti_size])
    if sti_idx > 0 and sti_idx <=sti_size:
        x[:,sti_idx-1:sti_idx] = xin[:,sti_idx-1:sti_idx]
    test_stimulus[0,:,:] = x
    return test_stimulus

def load_full_play_data(test_idx, seq_len, sti_size, sti = 'init_D'):
    i = test_idx
    test_stimulus = np.zeros([1, seq_len, sti_size])

    #print('Loading test stimulus subject ' + str(i) + '...')
    if sti == 'init_D':
        xin = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + str(i) + '.txt')
    elif sti == 'taskdesign':
        xin = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + str(i) + '.mat')['Normalized_'+TASK+'_taskdesign']
    else:
        xin = np.zeros([seq_len, sti_size])
        print('Error: invalid stimulus type: ' + sti)

    x = -np.ones([seq_len, sti_size])
    x[:,0:sti_size] = xin[:,0:sti_size]
    test_stimulus[0,:,:] = x
    return test_stimulus

def gen_test_stimulus(sti_size, seq_len, sti_idx, design_range = 0, sti = 'taskdesign'):
    i = 1
    test_stimulus = np.zeros([1, seq_len, sti_size])

    if sti == 'init_D':
        xin = np.loadtxt(DATA_BASE + STI_DIR + STI_PREFIX + str(i) + '.txt')
    elif sti == 'taskdesign':
        xin = sio.loadmat(DATA_BASE + DESIGN_DIR + DESIGN_PREFIX + str(i) + '.mat')['Normalized_'+TASK+'_taskdesign']
    else:
        xin = np.zeros([seq_len, sti_size])
        print('Error: invalid stimulus type: ' + sti)

    if design_range == -1:
        x = -np.ones([seq_len, sti_size])
    elif design_range == 0:
        x = np.zeros([seq_len, sti_size])
        xin = (xin+1)/2; # resize stimulus from [-1,1] to [0,1]

    if sti_idx > 0:
        x[:,sti_idx-1:sti_idx] = xin[:, 1:2] # 2nd stimulus, fixed
    test_stimulus[0,:,:] = x
    return test_stimulus

# Genarate experiment mark
exp_mark = gen_exp_mark(DATASET, TASK, SUBIDX, MAXSUB, STITYPE, NORM, RNNCELL, HIDDENSIZE, NLAYERS, 
        learning_rate, MAXEPOCH, LRDEPOCH, LRDRATIO, dropout_pkeep, MARK, SHUFFLE)
print('======> ', exp_mark)

# List training subjects
flist = list_training_data_h5(SUBIDX, MAXSUB)

lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs & variables
X = tf.placeholder(tf.float32, [BATCHSIZE, None, SIG_SIZE], name='X')
Xt = tf.placeholder(tf.float32, [None, SEQ_LEN, FEATURESIZE], name='Xtest')
We = tf.Variable(tf.truncated_normal([SIG_SIZE, HIDDENSIZE]), name="W_enc")
#We = tf.Variable(tf.ones([SIG_SIZE, HIDDENSIZE]), name="W_enc")
be = tf.Variable(tf.zeros([HIDDENSIZE]), name="b_enc")
#Wd = tf.Variable(tf.truncated_normal([HIDDENSIZE, SIG_SIZE]), name="W_dec")
Wd = tf.Variable(tf.ones([HIDDENSIZE, SIG_SIZE]), name="W_dec")
bd = tf.Variable(tf.zeros([SIG_SIZE]), name="b_dec")

Xflat = tf.reshape(X, [-1, SIG_SIZE])    # [ BATCHSIZE x SEQ_LEN, SIG_SIZE]
Xfc = tf.matmul(Xflat, We) + be          # [ BATCHSIZE x SEQ_LEN, HIDDENSIZE]   
Xfc_ = tf.reshape(Xfc, [BATCHSIZE, SEQ_LEN, HIDDENSIZE]) # [ BATCHSIZE, SEQ_LEN, HIDDENSIZE ]

# RNN model
with tf.variable_scope('rnn_enc', reuse=None):
    if RNNCELL == 'LSTM':
        cells_enc = [rnn.BasicLSTMCell(FEATURESIZE, forget_bias=1.0, state_is_tuple=True) for _ in range(NLAYERS)]
    elif RNNCELL == 'GRU':
        cells_enc = [rnn.GRUCell(FEATURESIZE) for _ in range(NLAYERS)]
    elif RNNCELL == 'BASIC':
        cells_enc = [rnn.BasicRNNCell(FEATURESIZE) for _ in range(NLAYERS)]
    dropcells_enc = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells_enc]
    multicell_enc = rnn.MultiRNNCell(dropcells_enc, state_is_tuple=True)
    zerostate_enc = multicell_enc.zero_state(BATCHSIZE, dtype=tf.float32)
    Yr, Hr = tf.nn.dynamic_rnn(multicell_enc, tf.reverse_sequence(Xfc_, [SEQ_LEN], batch_dim=0, seq_dim=1), dtype=tf.float32, initial_state=zerostate_enc)
    # Yr: [ BATCHSIZE, SEQ_LEN, FEATURESIZE]
    # Hr:  [ BATCHSIZE, FEATURESIZE*NLAYERS ] # this is the last state in the sequence
    Hr = tf.identity(Hr, name='Hr')  # just to give it a name
    cellweights_enc = multicell_enc.weights
    print(cellweights_enc)

Xr = tf.reverse_sequence(Yr, [SEQ_LEN], batch_dim=0, seq_dim=1)
Yrflat = tf.reshape(Xr, [-1, FEATURESIZE])    # [ BATCHSIZE x SEQ_LEN, FEATURESIZE]

# RNN model dec
with tf.variable_scope('rnn_dec', reuse=None):
    if RNNCELL == 'LSTM':
        cells_dec = [rnn.BasicLSTMCell(HIDDENSIZE, forget_bias=1.0, state_is_tuple=True) for _ in range(NLAYERS)]
    elif RNNCELL == 'GRU':
        cells_dec = [rnn.GRUCell(HIDDENSIZE) for _ in range(NLAYERS)]
    elif RNNCELL == 'BASIC':
        cells_dec = [rnn.BasicRNNCell(HIDDENSIZE) for _ in range(NLAYERS)]
    dropcells_dec = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells_dec]
    multicell_dec = rnn.MultiRNNCell(dropcells_dec, state_is_tuple=True)
    zerostate_dec = multicell_dec.zero_state(BATCHSIZE, dtype=tf.float32)
    Yrr, Hrr = tf.nn.dynamic_rnn(multicell_dec, Xr, dtype=tf.float32, initial_state=zerostate_dec)
    Yrrt, Hrrt = tf.nn.dynamic_rnn(multicell_dec, Xt, dtype=tf.float32, initial_state=zerostate_dec)
    # Yrr: [ BATCHSIZE, SEQ_LEN, HIDDENSIZE]
    # Hrr:  [ BATCHSIZE, HIDDENSIZE*NLAYERS ] # this is the last state in the sequence
    Hrr = tf.identity(Hrr, name='Hrr')  # just to give it a name
    cellweights_dec = multicell_dec.weights
    print(cellweights_dec)

# Linear layer to outputs
Yrrflat = tf.reshape(Yrr, [-1, HIDDENSIZE])     # [ BATCHSIZE x SEQ_LEN, HIDDENSIZE]
Yflat = tf.matmul(Yrrflat, Wd) + bd             # [ BATCHSIZE x SEQ_LEN, SIG_SIZE]
Yrrtflat = tf.reshape(Yrrt, [-1, HIDDENSIZE])     # [ BATCHSIZE x SEQ_LEN, HIDDENSIZE]
Ytflat = tf.matmul(Yrrtflat, Wd) + bd             # [ BATCHSIZE x SEQ_LEN, SIG_SIZE]

# loss & optimizer
loss = tf.reduce_mean(tf.pow(Yflat - Xflat, 2))
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQ_LEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# stats for display
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
loss_summary = tf.summary.scalar("batch_loss", batchloss)
lr_summary = tf.summary.scalar('learning_rate', lr)
We_hist = tf.summary.histogram("We", We)
be_hist = tf.summary.histogram("be", be)
Wd_hist = tf.summary.histogram("Wd", Wd)
bd_hist = tf.summary.histogram("bd", bd)
summaries = tf.summary.merge([loss_summary, lr_summary, We_hist, be_hist, Wd_hist, bd_hist])

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1)

# init
init = tf.global_variables_initializer()
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.75
#sess = tf.Session(config=config)
sess = tf.Session()

if TRAINING:
    # Init Tensorboard stuff.
    timestamp = str(math.trunc(time.time()))
    if SAVESMM:
        summary_writer = tf.summary.FileWriter("log/" + exp_mark + "-" + timestamp, sess.graph)

    # start training
    sess.run(init)

    # training loop
    step = 0
    t_tick = time.time()
    sublist = np.arange(1, MAXSUB + 1)
    for epoch in range(1, MAXEPOCH + 1):
        if SHUFFLE:
            np.random.shuffle(sublist)
        print('======> [epoch', epoch, '],subidx list:', sublist)
        epoch_loss = 0;
        for sub in sublist:
            step += 1
            # training on whole batch
            t_start = time.time()
            stimulus, signals = load_training_data_h5(flist, sub-1, SEQ_LEN, STISIZE, SIG_SIZE, FASTVALID, STITYPE)
            t_load = time.time()
            feed_dict = {X: signals[:,0:TRAINLEN,:], lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
            _, l, bl, smm = sess.run([train_step, seqloss, batchloss, summaries], feed_dict=feed_dict)
            epoch_loss = epoch_loss + bl
            t_run = time.time()
            print(str(step) + ' [epoch ' +  str(epoch) + '] Batchloss ' + str(bl) + ', Timecost ' +
                    str(round(time.time()-t_tick, 3)) + '(' + str(round(t_load-t_start, 3)) + '/' +
                    str(round(t_run-t_load, 3)) + ')')
            t_tick = time.time()
            # save training data for Tensorboard
            if SAVESMM:
                summary_writer.add_summary(smm, step)
            # save a checkpoint
            if step % 100 == 1 and SAVECKPT: # mod(step)==1 to forbid overwrite run over ckpt from auto save ckpt
                saver.save(sess, 'checkpoints/rnn_train_' + exp_mark, global_step=step)
        epoch_loss = epoch_loss / MAXSUB
        summary = tf.Summary()
        summary.value.add(tag='epoch_loss', simple_value = epoch_loss)
        summary_writer.add_summary(summary, epoch)
        # lr decay
        if epoch % LRDEPOCH == 0:
            learning_rate = learning_rate/LRDRATIO
            print("learning rate decay: ", learning_rate)
    # End of training, save checkpoint
    if SAVECKPT:
        saver.save(sess, 'checkpoints/rnn_train_' + exp_mark, global_step=step)

# test
if TEST:
    if not TRAINING:
        saver = tf.train.import_meta_graph('./checkpoints/rnn_train_' + PLAYCKPT + '.meta')
        saver.restore(sess, './checkpoints/rnn_train_' + PLAYCKPT)
    else:
        PLAYCKPT = exp_mark + '-' + str(step)

    if not os.path.exists("test"):
        os.mkdir("test")
    save_path = './test/' + PLAYCKPT + EXTRAMARK
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    Weo, beo, Wdo, bdo = sess.run([We, be, Wd, bd])
    print('======> Saving W and b...')
    print('We', Weo.shape, ' be', beo.shape, 'Wd', Wdo.shape, ' bd', bdo.shape)
    np.savetxt(save_path + '/We.txt', Weo, fmt='%.9e')
    np.savetxt(save_path + '/be.txt', beo, fmt='%.9e')
    np.savetxt(save_path + '/Wd.txt', Wdo, fmt='%.9e')
    np.savetxt(save_path + '/bd.txt', bdo, fmt='%.9e')

    print('======> Test 0 based input')
    for index in range(0, FEATURESIZE + 1):
        print('  ======> Feature', index)
        stimulus = gen_test_stimulus(FEATURESIZE, SEQ_LEN, index, 0, STITYPE)
        feed_dict = {Xt: stimulus[:,:,:], pkeep: 1.0, batchsize: BATCHSIZE}
        yt, st = sess.run([Yrrtflat, Ytflat], feed_dict=feed_dict)
        #yt = np.array(yt)
        #yt = yt[0,:,:]
        np.savetxt(save_path + '/Yt' + str(index) + '.txt', yt, fmt='%.9e')
        np.savetxt(save_path + '/St' + str(index) + '.txt', st, fmt='%.3e')

    print('======> Test -1 based input')
    for index in range(0, FEATURESIZE + 1):
        print('  ======> Feature', index)
        stimulus = gen_test_stimulus(FEATURESIZE, SEQ_LEN, index, -1, STITYPE)
        feed_dict = {Xt: stimulus[:,:,:], pkeep: 1.0, batchsize: BATCHSIZE}
        yt = sess.run([Yrrtflat], feed_dict=feed_dict)
        yt = np.array(yt)
        yt = yt[0,:,:]
        np.savetxt(save_path + '/Ytn' + str(index) + '.txt', yt, fmt='%.9e')

    print('======> Saving features and reconstructed signals')
    sublist = np.arange(1, MAXSUB + 1)
    for sub in sublist:
        print('  ======> subject', sub)
        stimulus, signals = load_training_data_h5(flist, sub-1, SEQ_LEN, STISIZE, SIG_SIZE, FASTVALID, STITYPE)
        feed_dict = {X: signals[:,0:TRAINLEN,:], lr: learning_rate, pkeep: 1.0, batchsize: BATCHSIZE}
        fo, Sro = sess.run([Yrflat, Yflat], feed_dict=feed_dict)
        #np.savetxt(save_path + '/F_sub' + str(sub) + '.txt', fo, fmt='%.9e')
        #np.savetxt(save_path + '/Sr_sub' + str(sub) + '.txt', Sro, fmt='%.3e')
