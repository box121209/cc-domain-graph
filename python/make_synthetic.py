# -*- coding: utf-8 -*-
"""
Example usage:

FILE=model_from_big_domain_string_1.gz_arch_8_16_unroll_20_step_3_dropout_0.1_iter_17.npy

python ./python/make_synthetic.py $FILE -n 2000 -init 'github.com'

"""

import sys, os
import numpy as np
from binascii import unhexlify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppresses Tensorflow messages

#####################################################################
# read input parameters

# defaults:
temp = 1.0
quote_length = 1000
unroll = 20
init = '\n' * unroll
modelpath = "./models"

# command line:
try:
    infile = sys.argv[1];   del sys.argv[1]
except:
    print "Usage: python ", sys.argv[0], "infile [options]"
    print "Options are:"
    print "        -n length of output [1000]"
    print "        -init        [newlines]"
    print "        -temp        [1.0]"
    print "        -modelpath        [./models]"
    sys.exit(1)

while len(sys.argv) > 1:
    option = sys.argv[1];               del sys.argv[1]
    if option == '-n':
        quote_length = int(sys.argv[1]); del sys.argv[1]
    elif option == '-init':
        init = sys.argv[1];             del sys.argv[1]
    elif option == '-temp':
        temp = float(sys.argv[1]);           del sys.argv[1]
    elif option == '-modelpath':
        modelpath = sys.argv[1];             del sys.argv[1]
    else:
        print sys.argv[0],': invalid option', option
        sys.exit(1)


#####################################################################
# read model file, parse architecture

infile = "%s/%s" % (modelpath, infile)
   
try:
    model_wts = np.load(infile)
except:
    print("Can't find model file")

arch = [model_wts[i].shape[0]\
       for i in range(len(model_wts))\
       if len(model_wts[i].shape) == 2] 

INSIZE = arch[0]
nh = len(arch)/2
nhidden = [arch[1+2*i] for i in range(nh)]
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

def sample(a, temperature=1.0):
    """
    Samples an index from a probability array;
    higher temperature raises the entropy and vice versa    
    """
    a = a**(1/temperature)
    dist = a / np.sum(a)
    choices = range(len(a)) 
    return np.random.choice(choices, p=dist)

#####################################################################
# build the model: stacked LSTM

from keras.models import Sequential
from keras.layers.core import Dense, Activation
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


def stepping(window):
    x = np.zeros((1, unroll, INSIZE))
    if INSIZE == 8:
        for t,b in enumerate(window):
            x[0, t, :] = binabet[b]
    elif INSIZE == 256:
        for t,b in enumerate(window):
            x[0, t, b] = 1.0
    return model.predict(x, verbose=0)[0]

output = [int(b.encode('hex'), 16) for b in init]

while len(output) < unroll:
    output = [int('\n'.encode('hex'), 16)] + output

window = output[:unroll]
idx = unroll

while idx < len(output):
    preds = stepping(window)
    next_byte = output[idx]
    window = window[1:] + [next_byte]
    idx += 1

for i in range(quote_length):
    preds = stepping(window)
    next_byte = sample(preds, temperature=temp)
    window = window[1:] + [next_byte]
    output += [next_byte]

print(bytearray(output))


#####################################################################