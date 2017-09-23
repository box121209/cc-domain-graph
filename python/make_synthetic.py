"""
Example usage:

python ./python/make_synthetic.py model_from_big_domain_string_1.gz_arch_8_16_unroll_20_step_3_dropout_0.1.npy

"""

import sys, time, os, shutil
import numpy as np
import gzip
import math
import time

#####################################################################
# read input parameters

# defaults:
temp = 1.0
quote_length = 1000
unroll = 20

# command line:
try:
    infile = "./models/%s" % sys.argv[1];   del sys.argv[1]
except:
    print "Usage: python ", sys.argv[0], "infile [options]"
    print "Options are:"
    print "        -arch (input + hidden layer sizes) [8,16]"
    sys.exit(1)

while len(sys.argv) > 1:
    option = sys.argv[1];               del sys.argv[1]
    if option == '-arch':
        arch = [int(x) for x in sys.argv[1].split(',')]
        del sys.argv[1]
    else:
        print sys.argv[0],': invalid option', option
        sys.exit(1)
    
#####################################################################
# read model file

try:
    model_wts = np.load(infile)
except:
    print("Can't find model file")
    
arch = [model_wts[2*(i+1)].shape[0] for i in range(len(model_wts)/2)]

INSIZE = model_wts[0].shape[0]
nhidden = [a/4 for a in arch[:-1]]
OUTSIZE = 256

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
    model.add(LSTM(nhidden[nlayers-1],
                   return_sequences=False))
else:
    model.add(LSTM(nhidden[0],
                   return_sequences=False,
                   input_shape=(unroll, INSIZE)))
model.add(Dense(OUTSIZE))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# load current weights
model.set_weights(model_wts)


#####################################################################
# generate bytes

init_quote = '\n' * unroll

generated = [b.encode('hex') for b in init_quote]
sys.stdout.write(''.join([unichr(int(h, 16)) for h in generated]))
for i in range(quote_length):
    x = np.zeros((1, unroll, 256))
    for t,b in enumerate(generated):
        x[0, t, byte_idx[b]] = 1.0

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, temperature=temp)
    next_byte = hexabet[next_index]
    generated = generated[1:] + [next_byte]

    sys.stdout.write(unichr(int(next_byte, 16)))
    sys.stdout.flush()

#####################################################################
