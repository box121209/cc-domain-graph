"""
Example usage:

python ./python/train_rnn.py smallstring.gz -arch 16,16
python ./python/train_rnn.py mediumstring.gz -arch 16 -niters 1
python ./python/train_rnn.py big_domain_string_1.gz -arch 16 -niters 1

"""

import sys, time, os, shutil
import numpy as np
import gzip
import math
import time

#####################################################################
# read input parameters

# defaults:
outfile = ""
niters = 10
nhidden = [16]
unroll = 20
step = 3
dropout = 0.1
batch_size = 128
verbose = 1

# fixed:
INSIZE  = 8
OUTSIZE = 256

# command line:
try:
    infile = sys.argv[1];           del sys.argv[1]
except:
    print "Usage: python ", sys.argv[0], "infile [options]"
    print "Options are:"
    print "        -o outfile "
    print "        -unroll [20]"
    print "        -step [3]"
    print "        -dropout [0.1]"
    print "        -niters [10]"
    print "        -arch (hidden layer sizes) [16]"
    sys.exit(1)

while len(sys.argv) > 1:
    option = sys.argv[1];               del sys.argv[1]
    if option == '-o':
        outfile = sys.argv[1];          del sys.argv[1]
    elif   option == '-unroll':
        unroll = int(sys.argv[1]);      del sys.argv[1]
    elif option == '-step':
        step = int(sys.argv[1]);        del sys.argv[1]
    elif option == '-dropout':
        dropout = float(sys.argv[1]);   del sys.argv[1]
    elif option == '-niters':
        niters = int(sys.argv[1]);      del sys.argv[1]
    elif option == '-arch':
        nhidden = [int(x) for x in sys.argv[1].split(',')]
        del sys.argv[1]
    else:
        print sys.argv[0],': invalid option', option
        sys.exit(1)


#####################################################################
# set output file

if outfile == "":
    outfile = "model_from_%s_arch_%d" % (infile, INSIZE)
for i in nhidden: 
    outfile += "_%d" % i
outfile += "_unroll_%d_step_%d_dropout_%g.npy" % (unroll, step, dropout)
    

#####################################################################
# preliminary definitions

def bn(x):
    """
    Binary representation of 0-255 as 8-long int vector.
    """
    str = bin(x)[2:]
    while len(str) < 8:
	str = '0' + str
    return [int(i) for i in list(str)]

def hx(i):
    """
    Normalised 2-char hex representation of 0-255
    """
    a = hex(i)[2:]
    if len(a)<2: a = ''.join(['0',a])
    return a

binabet = [bn(x) for x in range(256)]
hexabet = [hx(x) for x in range(256)]
byte_idx = dict((c, i) for i,c in enumerate(hexabet))
nlayers = len(nhidden)


#####################################################################
# check hardware

from tensorflow.python.client import device_lib

print("Checking hardware ...")
print(device_lib.list_local_devices())

#####################################################################
# build the model: stacked LSTM

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, Recurrent

model = Sequential()
if nlayers > 1:
    for i in range(nlayers-1):
        model.add(LSTM(nhidden[i],
                       return_sequences=True,
                       input_shape=(unroll, INSIZE)))
        model.add(Dropout(dropout))
    model.add(LSTM(nhidden[nlayers-1],
                   return_sequences=False))
else:
    model.add(LSTM(nhidden[0],
                   return_sequences=False,
                   input_shape=(unroll, INSIZE)))
model.add(Dropout(dropout))
model.add(Dense(OUTSIZE))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Model architecture:")
print(model.summary())

# load current weights
try:
    model.set_weights(np.load("./models/%s" %outfile))
    print("Initialising with weights found at ./models/%s" % outfile)
except:
    print("Can't find existing model, initialising random weights")


#####################################################################
# read training data

print("Reading data file ./sdata/%s ..." % infile)
t0 = time.clock()
with gzip.open("./sdata/%s" % infile, 'rb') as f:
    content = f.read()

bytes = [b.encode('hex') for b in content]
n_train = len(bytes)
t1 = time.clock()
print("Read %d bytes in %g seconds" % (len(bytes), (t1 - t0)))


#####################################################################
# prepare and run training

print("Cutting into byte sequences of size %d ..." % unroll)
t0 = time.clock()
sentences = []
next_byte = []
for i in range(0, n_train - unroll, step):
    sentences.append(bytes[i: i + unroll])
    next_byte.append(bytes[i + unroll])
del bytes # release memory
t1 = time.clock()
print("Written %d sequences in %g seconds" % (len(sentences), (t1 - t0)))

# convert to feature vector + next character:
print("Formatting arrays ...")
t0 = time.clock()
X = np.zeros((len(sentences), unroll, INSIZE), dtype=np.bool)
y = np.zeros((len(sentences), OUTSIZE), dtype=np.bool)

if INSIZE == 256:
    for i, sentence in enumerate(sentences):
        for t,b in enumerate(sentence):
            X[i, t, byte_idx[b]] = 1
        y[i, byte_idx[next_byte[i]]] = 1

elif INSIZE == 8:
    for i, sentence in enumerate(sentences):
        for t,b in enumerate(sentence):
            X[i, t, :] = binabet[byte_idx[b]]
        y[i, byte_idx[next_byte[i]]] = 1
del sentences # release memory
t1 = time.clock()
print("Arrays written in %g seconds" % (t1 - t0))

print("Fitting model ...")
t0 = time.clock()
model.fit(X, y, batch_size=batch_size, epochs=niters, verbose=verbose)
t1 = time.clock()
print("Done in %g seconds" % (t1 - t0))

#####################################################################
# report

np.save("./models/%s" % outfile, model.get_weights())
print("Model written to ./models/%s" % outfile)

#####################################################################
