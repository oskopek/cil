import tensorflow as tf
import  sys

CHAR_SIZE = 128
LAYERS_CHAR = 2
LAYERS_WORD = 2
WORD_SIZE = 4*LAYERS_CHAR*CHAR_SIZE

MAKE_TEST = False

save_path="../data_out/M/128_2_2/model.ckpt"
dropout_keep_prob = 0.65
learning_rate = 0.0005
training_steps = 100000
batch_size = 50
display_step = 10


dict = eval(open("../data_in/dict.json").read())
table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(list(dict.keys()), list(dict.values()),
    value_dtype=tf.int32), -1)

DICT_SIZE = len(dict) #-1

# Split sentences into characters
# based on https://stackoverflow.com/questions/48675780/tensorflow-split-string-then-split-the-result
def split_line(line):
    # Split the line into words
    line = tf.expand_dims(line, axis=0)
    line = tf.string_split(line, delimiter=' ')

    # Loop over the resulting words, split them into characters, and stack them back together
    def body(index, words):
        next_word = tf.sparse_slice(line, start=tf.to_int64(index), size=[1, 1]).values
        next_word = tf.string_split(next_word, delimiter='')
        words = tf.sparse_concat(axis=0, sp_inputs=[words, next_word], expand_nonconcat_dim=True)
        return index+[0, 1], words
    def condition(index, words):
        return tf.less(index[1], tf.size(line))

    i0 = tf.constant([0,1])
    first_word = tf.string_split(tf.sparse_slice(line, [0,0], [1, 1]).values, delimiter='')
    _, line = tf.while_loop(condition, body, loop_vars=[i0, first_word], back_prop=False)

    # Convert to dense
    return tf.sparse_tensor_to_dense(line, default_value=' ')


#Open datasets
pos = tf.data.TextLineDataset("../data_in/t_pos.shuf.txt")
pos_lbl = tf.data.TextLineDataset("../data_in/t_pos_labels.txt")
neg = tf.data.TextLineDataset("../data_in/t_neg.shuf.txt")
neg_lbl = tf.data.TextLineDataset("../data_in/t_neg_labels.txt")

valid = tf.data.TextLineDataset("../data_in/valid.txt")
valid_lbl = tf.data.TextLineDataset("../data_in/valid_labels.txt")

test = tf.data.TextLineDataset("../data_in/test.txt")

pos=pos.map(split_line) #Split sentences into padded characters
pos = pos.map(lambda x:table.lookup(x)) #Map characters to IDs
pos_lbl = pos_lbl.map(lambda x:tf.string_to_number(x,out_type=tf.int32)) #Make int out of string labels
pos_lbl = pos_lbl.map(lambda x:tf.one_hot(x,2)) # Encode labels as one hot, i.e. [0,1] or [1,0]
pos = pos.zip((pos, pos_lbl)) #zip sentences with labels

neg=neg.map(split_line)
neg = neg.map(lambda x:table.lookup(x))
neg_lbl = neg_lbl.map(lambda x:tf.string_to_number(x,out_type=tf.int32))
neg_lbl = neg_lbl.map(lambda x:tf.one_hot(x,2))
neg= neg.zip((neg,neg_lbl))

pos=pos.shuffle(10000)
neg=neg.shuffle(10000)

valid=valid.map(split_line)
valid = valid.map(lambda x:table.lookup(x))
valid_lbl = valid_lbl.map(lambda x:tf.string_to_number(x,out_type=tf.int32))
valid_lbl = valid_lbl.map(lambda x:tf.one_hot(x,2))
valid = valid.zip((valid, valid_lbl))
#valid = valid.repeat(1)

test= test.map(split_line)
test = test.map(lambda x:table.lookup(x))
test = test.repeat(1)


#character embeddings
embed = tf.get_variable("embedding", shape=[DICT_SIZE,CHAR_SIZE])

#Cell definitions
fwcell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=CHAR_SIZE, state_is_tuple=True) for _ in range(LAYERS_CHAR)])
bwcell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=CHAR_SIZE, state_is_tuple=True) for _ in range(LAYERS_CHAR)])

fwcell2 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=WORD_SIZE, state_is_tuple=True) for _ in range(LAYERS_WORD)])
bwcell2 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=WORD_SIZE, state_is_tuple=True) for _ in range(LAYERS_WORD)])

#Dropout
fwcell = tf.nn.rnn_cell.DropoutWrapper(fwcell,dropout_keep_prob)
bwcell = tf.nn.rnn_cell.DropoutWrapper(bwcell,dropout_keep_prob)
fwcell2 = tf.nn.rnn_cell.DropoutWrapper(fwcell2,dropout_keep_prob)
bwcell2 = tf.nn.rnn_cell.DropoutWrapper(bwcell2,dropout_keep_prob)

#Classifier
classW = tf.Variable(tf.random_normal([4*LAYERS_WORD*WORD_SIZE, 2]))
classB = tf.Variable(tf.random_normal([2]))

#Predict sentiment for one sentence, returns logits
def predict(X):
    embedded = tf.nn.embedding_lookup(embed,X)
    lengths = tf.cast(tf.count_nonzero(embedded, [2, 1]) / CHAR_SIZE,dtype=tf.int32)
    lengths = tf.reshape(lengths, [tf.shape(lengths)[0]])

    outputs, states  = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fwcell,
        cell_bw=bwcell,
        dtype=tf.float32,
        sequence_length=lengths,
        inputs=embedded, scope="char"
    #    , initial_state_fw=fwcell.zero_state(tf.shape(lengths)[0],tf.float32), initial_state_bw=bwcell.zero_state(tf.shape(lengths)[0],tf.float32)
    )

    statefw, statebw = states

    X2 = tf.concat([tf.reshape(statefw,[tf.shape(lengths)[0],CHAR_SIZE*2*LAYERS_CHAR]),
                    tf.reshape(statebw,[tf.shape(lengths)[0],CHAR_SIZE*2*LAYERS_CHAR])],1)

    sentlen = tf.shape(lengths)[0]

    outputs2, states2 = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fwcell2,
        cell_bw=bwcell2,
        dtype=tf.float32,
        sequence_length=tf.expand_dims(tf.convert_to_tensor(sentlen),0),
        inputs=tf.expand_dims(X2,0), scope="word"
        #,initial_state_fw=fwcell2.zero_state(1,tf.float32), initial_state_bw=bwcell2.zero_state(1,tf.float32)
    )

    statefw2, statebw2 = states2

    X3 = tf.expand_dims(tf.concat([tf.reshape(statefw2,[-1]),tf.reshape(statebw2,[-1])],0),0)
    #Run result through linear classifier
    return tf.matmul(X3, classW) + classB

#Run batch of predictions. uses TF while loop and TensorArray
def do_batch(batch_size,  it_pos, it_neg):
    def body(counter, maxcounter,logits,Y):
        s1, l1 = it_pos.get_next()
        s2, l2 = it_neg.get_next()
        logits1 = predict(s1)
        logits2 = predict(s2)

        logits = logits.write(counter, logits1)
        Y = Y.write(counter, tf.expand_dims(l1,0))
        counter = tf.add(counter,one)

        logits = logits.write(counter, logits2)
        Y = Y.write(counter, tf.expand_dims(l2, 0))
        counter = tf.add(counter, one)

        return counter, maxcounter,logits,Y
    def condition(counter, maxcounter,logits,Y):
        return tf.less(counter, maxcounter)

    counter =tf.convert_to_tensor(0, dtype=tf.int32)
    one = tf.convert_to_tensor(1, dtype=tf.int32)
    maxcounter = tf.convert_to_tensor(batch_size, dtype=tf.int32)
    logits = tf.TensorArray(tf.float32, batch_size)
    Y = tf.TensorArray(tf.float32, batch_size)
    _ ,_ , logits, Y  = tf.while_loop(condition, body, loop_vars=[counter, maxcounter,logits,Y], back_prop=True)
    return logits.concat(), Y.concat()

#create iterators over positive and negative dataset
it_pos = pos.make_initializable_iterator()
it_neg = neg.make_initializable_iterator()

logits, Y = do_batch(batch_size,it_pos,it_neg)
prediction = tf.nn.softmax(logits)

#Unused legacy function
def eval_one(va,vb):
    pp = tf.nn.softmax(predict(va))
    return (tf.equal(tf.argmax(pp, 1), tf.argmax(vb, 0)))


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#run one validation batch
def valid_one(vit):
    ll, yy = do_batch(batch_size,vit,vit)
    pp = tf.nn.softmax(ll)
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pp, 1), tf.argmax(yy, 1)), tf.float32))

#run one test sentence
def test_one(test_sent):
    ll = predict(test_sent)
    pp = tf.nn.softmax(ll)
    #return pp
    return tf.argmax(pp,1)


config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
saver = tf.train.Saver()
with tf.Session(config=config) as sess:

    table.init.run()

    sess.run(it_pos.initializer)
    sess.run(it_neg.initializer)

    try:
        saver.restore(sess, save_path)
    except:
        print("No saved model")

    if MAKE_TEST:
        test_it = test.make_initializable_iterator()
        sess.run(test_it.initializer)
        while True:
            try:
                tsent = test_it.get_next()
                res = sess.run(test_one(tsent))
                #print(res)
                if res[0] == 0:
                    print(-1)
                else:
                    print(1)
            except tf.errors.OutOfRangeError:
                break
    else:
        for step in range(1, training_steps + 1):
            sess.run(train_op)
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc= sess.run([loss_op, accuracy])
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
            if step % (display_step*20) == 0:
                #Validate and save
                print("Validating")
                acc = 0.0
                vit = valid.make_initializable_iterator()
                sess.run(vit.initializer)
                batches_count = 30 # one validation round is this amount of batches. So sloww....
                for ii in range(0,batches_count):
                    acc = acc + sess.run(valid_one(vit))
                    print(".", end='')
                    sys.stdout.flush()
                print("Accuracy "+"{:.3f}".format(acc/batches_count))
                print("Saving...")
                saver.save(sess, save_path)
                print("Saved!")
