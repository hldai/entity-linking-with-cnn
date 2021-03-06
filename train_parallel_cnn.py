#! /usr/bin/python
__author__ = 'dhl'

import sys

import numpy as np

import theano
import theano.tensor as T

import data_load
from sentence_cnn import SentenceCNN
from theano_cnn import HiddenLayer, relu, sgd_updates_adadelta


# TODO remove these global variables
def_filter_hs = [2, 3]
sentence_pad_len = def_filter_hs[-1] - 1


max_num_entity_words = 50
entity_pad_len = 1
entity_rep_len = max_num_entity_words + 2 * entity_pad_len
entity_hs = [1]
num_entity_rep_feature_maps = 300


def to_theano_shared(vals):
    return theano.shared(value=np.asarray(vals,
                                          dtype=theano.config.floatX),
                         borrow=True)


def get_entity_context_similarities(unit_mc, cnn_output_for_entities, batch_size, num_candidates):
    entity_reps = cnn_output_for_entities.reshape((batch_size, num_candidates,
                                                   cnn_output_for_entities.shape[1]))
    unit_entity_reps = entity_reps / T.sqrt(T.maximum(
        T.sum(T.sqr(entity_reps), 2), 0.0001)).dimshuffle(0, 1, 'x')
    return (unit_mc.dimshuffle(0, 'x', 1) * unit_entity_reps).sum(axis=2)


def get_training_variables_with_entity_side_cnn(batch_entity_vecs, shared_words, word_vec_len, batch_size,
                                                hidden_out_len, unit_mc, num_train_candidates,
                                                num_val_candidates, gold_labels, rng):
    entity_vecs_reshaped = batch_entity_vecs.reshape((batch_entity_vecs.shape[0] * batch_entity_vecs.shape[1],
                                                      batch_entity_vecs.shape[2]))

    sentence_cnn1_train = SentenceCNN(entity_vecs_reshaped, shared_words, entity_rep_len, word_vec_len, entity_hs,
                                      num_entity_rep_feature_maps,
                                      batch_size * num_train_candidates, hidden_out_len, rng)
    entity_reps_train = sentence_cnn1_train.output
    similarities_train = get_entity_context_similarities(unit_mc, entity_reps_train, batch_size, num_train_candidates)
    loss = T.maximum(0, 1 - similarities_train[:, 0] + similarities_train[:, 1]).sum()

    sentence_cnn1_val = SentenceCNN(entity_vecs_reshaped, shared_words, entity_rep_len, word_vec_len, entity_hs,
                                    num_entity_rep_feature_maps,
                                    batch_size * num_val_candidates,
                                    hidden_out_len, rng,
                                    hidden_W=sentence_cnn1_train.hiddenW,
                                    hidden_b=sentence_cnn1_train.hiddenb,
                                    conv_Ws=sentence_cnn1_train.convWs,
                                    conv_bs=sentence_cnn1_train.convbs)
    entity_reps_val = sentence_cnn1_val.output

    similarities_val = get_entity_context_similarities(unit_mc, entity_reps_val, batch_size, num_val_candidates)
    correct_rate = T.mean(T.eq(gold_labels, T.argmax(similarities_val, axis=1)))

    params = sentence_cnn1_train.params

    return loss, correct_rate, params


def get_training_variables_no_entity_side_cnn(batch_entity_vecs, entity_vec_len, hidden_out_len, unit_mc,
                                              gold_labels, rng):
    matcher1 = HiddenLayer(rng, batch_entity_vecs, entity_vec_len, hidden_out_len, relu)
    entity_reps = matcher1.output
    unit_entity_reps = entity_reps / T.sqrt(T.maximum(
        T.sum(T.sqr(entity_reps), 2), 0.0001)).dimshuffle(0, 1, 'x')

    similarities = (unit_mc.dimshuffle(0, 'x', 1) * unit_entity_reps).sum(axis=2)
    loss = T.maximum(0, 1 - similarities[:, 0] + similarities[:, 1]).sum()
    correct_rate = T.mean(T.eq(gold_labels, T.argmax(similarities, axis=1)))
    params = matcher1.params
    return loss, correct_rate, params


def train_cnn_for_el(train_data_file_name,
                     val_data_file_name,
                     num_val_candidates,
                     test_data_file_name,
                     num_test_candidates,
                     sentence_len, word_vec_len,
                     all_words,  # first row of all_words should be a non-existing word
                     wid_idx_dict,
                     entity_vecs,
                     entity_side_cnn=False,
                     gold_as_first_candidate=False,
                     skip_width_loading=40,  # skip width while loading samples
                     n_epochs=25,
                     batch_size=50,
                     filter_hs=def_filter_hs,
                     num_feature_maps=100,
                     lr_decay=0.9,
                     sqr_norm_lim=9,
                     hidden_out_len=50,
                     training_part_size=50000,
                     num_train_candidates=2):
    full_sentence_len = sentence_len + 2 * sentence_pad_len
    rng = np.random.RandomState(3435)

    print 'making entity_vecs...', len(entity_vecs)
    if entity_side_cnn:
        shared_entity_vecs = theano.shared(value=np.asarray(entity_vecs, dtype="int32"),
                                           name='entity_vecs', borrow=True)
    else:
        shared_entity_vecs = theano.shared(value=np.asarray(entity_vecs, dtype=theano.config.floatX),
                                           name='entity_vecs', borrow=True)

    print 'making shared_words...', len(all_words)
    shared_words = theano.shared(value=np.asarray(all_words, dtype=theano.config.floatX),
                                 name='shared_words', borrow=True)
    print 'done'

    val_contexts, val_indices = data_load.load_samples_full(val_data_file_name, wid_idx_dict, sentence_len,
                                                            sentence_pad_len,
                                                            skip_width=skip_width_loading,
                                                            num_candidates=num_val_candidates)
    num_val_batches = len(val_contexts) / batch_size
    print num_val_batches, 'validation batches'
    print len(val_indices[0]), 'candidates per mention'
    val_contexts = T.cast(to_theano_shared(val_contexts), 'int32')
    val_indices = T.cast(to_theano_shared(val_indices), 'int32')

    test_contexts, test_indices = data_load.load_samples_full(test_data_file_name, wid_idx_dict, sentence_len,
                                                              sentence_pad_len,
                                                              skip_width=skip_width_loading,
                                                              num_candidates=num_test_candidates)
    num_test_batches = len(test_contexts) / batch_size
    print num_test_batches, 'test batches'
    print len(test_indices[0]), 'candidates per mention'
    test_contexts = T.cast(to_theano_shared(test_contexts), 'int32')
    test_indices = T.cast(to_theano_shared(test_indices), 'int32')

    if gold_as_first_candidate:
        gold_labels = theano.shared(value=np.zeros(batch_size,
                                                   dtype='int32'),
                                    borrow=True)
    else:
        gold_labels = theano.shared(value=np.ones(batch_size,
                                                  dtype='int32'),
                                    borrow=True)

    x = T.imatrix('x')
    entities = T.imatrix('entities')

    sentence_cnn0 = SentenceCNN(x, shared_words, full_sentence_len, word_vec_len, filter_hs, num_feature_maps,
                                batch_size,
                                hidden_out_len, rng)
    mc = sentence_cnn0.output  # mention contexts
    unit_mc = mc / T.sqrt(T.maximum(T.sum(T.sqr(mc), 1), 0.0001)).dimshuffle(0, 'x')

    batch_entity_vecs = shared_entity_vecs[entities]

    if entity_side_cnn:
        loss, correct_rate, entity_side_params = get_training_variables_with_entity_side_cnn(batch_entity_vecs,
                                                                                             shared_words,
                                                                                             word_vec_len, batch_size,
                                                                                             hidden_out_len, unit_mc,
                                                                                             num_train_candidates,
                                                                                             num_val_candidates,
                                                                                             gold_labels, rng)
    else:
        loss, correct_rate, entity_side_params = get_training_variables_no_entity_side_cnn(batch_entity_vecs,
                                                                                           len(entity_vecs[0]),
                                                                                           hidden_out_len,
                                                                                           unit_mc,
                                                                                           gold_labels, rng)
    # params = matcher0.params + entity_side_params
    # for conv_layer in conv_layers:
    #     params += conv_layer.params

    # params = sentence_cnn0.params + matcher1.params
    # params = sentence_cnn0.params + sentence_cnn1_train.params

    params = sentence_cnn0.params + entity_side_params
    grad_updates = sgd_updates_adadelta(params, loss, lr_decay, 1e-6, sqr_norm_lim)

    index = T.lscalar()

    val_model = theano.function(
        [index],
        correct_rate,
        givens={x: val_contexts[index * batch_size: (index + 1) * batch_size],
                entities: val_indices[index * batch_size: (index + 1) * batch_size]}
    )

    test_model = theano.function(
        [index],
        correct_rate,
        givens={x: test_contexts[index * batch_size: (index + 1) * batch_size],
                entities: test_indices[index * batch_size: (index + 1) * batch_size]}
    )

    train_contexts = theano.shared(
        value=np.zeros((3, 2)),
        borrow=True)
    int_train_contexts = T.cast(train_contexts, 'int32')
    train_indices = theano.shared(
        value=np.zeros((3, 2)),
        borrow=True)
    int_train_indices = T.cast(train_indices, 'int32')
    train_model = theano.function(
        [index],
        loss,
        updates=grad_updates,
        givens={x: int_train_contexts[index * batch_size: (index + 1) * batch_size],
                entities: int_train_indices[index * batch_size: (index + 1) * batch_size]}
    )

    fdebug = theano.function(
        [index],
        batch_entity_vecs,
        givens={entities: int_train_indices[index * batch_size: (index + 1) * batch_size]}
    )
    # print fdebug(0)

    val_perfs = [val_model(i) for i in xrange(num_val_batches)]
    print('init val perf %f' % np.mean(val_perfs))

    epoch = 0
    max_val_perf = 0
    test_perf = 0
    print 'training ...'
    while epoch < n_epochs:
        f_train = open(train_data_file_name, 'rb')
        epoch += 1

        train_part_cnt = 0

        cur_train_contexts, cur_train_indices = data_load.load_training_samples(f_train,
                                                                                training_part_size,
                                                                                wid_idx_dict,
                                                                                sentence_len,
                                                                                sentence_pad_len)

        while not len(cur_train_contexts) == 0 and train_part_cnt < 100:
            train_contexts.set_value(cur_train_contexts, borrow=True)
            train_indices.set_value(cur_train_indices, borrow=True)

            train_part_cnt += 1
            num_train_batches = len(cur_train_contexts) / batch_size
            # print 'num_train_batches', num_train_batches
            mean_loss = 0
            for minibatch_index in xrange(num_train_batches):
                cur_loss = train_model(minibatch_index)
                mean_loss += cur_loss
                # print minibatch_index, cur_loss
            print 'loss:', mean_loss / num_train_batches
            # print fdebug(0)

            val_perfs = [val_model(i) for i in xrange(num_val_batches)]
            val_perf = np.mean(val_perfs)
            print('epoch %i, training part %i, val perf %f(%f), test perf %f'
                  % (epoch, train_part_cnt, val_perf, max_val_perf, test_perf))

            if val_perf > max_val_perf:
                max_val_perf = val_perf
                test_perfs = [test_model(i) for i in xrange(num_test_batches)]
                test_perf = np.mean(test_perfs)
                print('\tepoch %i, training part %i, test_perf %f'
                      % (epoch, train_part_cnt, test_perf))

            cur_train_contexts, cur_train_indices = data_load.load_training_samples(f_train,
                                                                                    training_part_size,
                                                                                    wid_idx_dict,
                                                                                    sentence_len,
                                                                                    sentence_pad_len)
        f_train.close()


def load_params(param_file_name):
    f = open(param_file_name, 'rb')
    params = dict()
    line = f.readline()
    while line:
        vals = line.decode('utf8').strip()
        if not vals == '':
            vals = vals.split(' ')
            params[vals[0]] = vals[1]
        line = f.readline()
    f.close()
    return params


def main():
    if len(sys.argv) < 2:
        print 'need params file'

    params = load_params(sys.argv[1])

    entity_side_cnn = params['entity_side_cnn'] == '1'
    word_vec_file_name = params['word_vec_file']

    if entity_side_cnn:
        entity_rep_file_name = params['entity_rep_indices_file']
    else:
        entity_rep_file_name = params['entity_rep_vec_file']

    train_data_file_name = params['train_data_file']
    val_data_file_name = params['val_data_file']
    test_data_file_name = params['test_data_file']

    training_part_size = int(params['training_part_size'])
    sentence_len = int(params['context_sentence_len'])

    _, word_vecs = data_load.load_word_vectors(word_vec_file_name)
    word_vec_len = len(word_vecs[0])

    if entity_side_cnn:
        print 'entity use cnn'
        global entity_rep_len
        wid_idx_dict, entity_vecs, entity_rep_len = data_load.load_index_vec_of_entities_fixed_len(
            entity_rep_file_name)
    else:
        wid_idx_dict, entity_vecs = data_load.load_entities(
            entity_rep_file_name,
            False)

    # wid_idx_dict, entity_vecs = data_load.load_entities_indices(
    #     entity_rep_file_name, max_num_entity_words, entity_pad_len)

    num_val_candidates = 30
    num_test_candidates = 30
    skipwidth_loading = 0
    train_cnn_for_el(train_data_file_name,
                     val_data_file_name,
                     num_val_candidates,
                     test_data_file_name,
                     num_test_candidates,
                     sentence_len, word_vec_len,
                     word_vecs,
                     wid_idx_dict,
                     entity_vecs,
                     entity_side_cnn=entity_side_cnn,
                     gold_as_first_candidate=False,
                     skip_width_loading=skipwidth_loading,
                     n_epochs=1,
                     training_part_size=training_part_size)


if __name__ == '__main__':
    main()
