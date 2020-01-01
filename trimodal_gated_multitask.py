import numpy as np, json
import pickle, sys, argparse
import keras
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers import *
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
global seed
seed = 1337
np.random.seed(seed)
import gc
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
#=============================================================
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#=============================================================
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))
#=============================================================
def calc_valid_result(result, valid_label, valid_mask, print_detailed_results=False):

    true_label=[]
    predicted_label=[]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if valid_mask[i,j]==1:
                true_label.append(np.argmax(valid_label[i,j] ))
                predicted_label.append(np.argmax(result[i,j] ))

    if print_detailed_results:
        print ("Confusion Matrix :")
        print (confusion_matrix(true_label, predicted_label))
        print ("Classification Report :")
        print (classification_report(true_label, predicted_label))

    return accuracy_score(true_label, predicted_label), precision_recall_fscore_support(true_label, predicted_label, average='weighted')

def attention(att_type, x, y):

    if att_type == 'simple':
        m_dash = dot([x, y], axes=[2,2])
        m = Activation('softmax')(m_dash)
        h_dash = dot([m, y], axes=[2,1])
        return multiply([h_dash, x])

    elif att_type == 'gated':
        alpha_dash = dot([y, x], axes=[2,2])
        alpha = Activation('softmax')(alpha_dash)
        x_hat = Permute((2, 1))(dot([x, alpha], axes=[1,2]))
        return multiply([y, x_hat])

    else:
        print ('Attention type must be either simple or gated.')

def emotionClass(testLabel):
    trueLabel     = []
    for i in range(testLabel.shape[0]):
        maxLen       = []
        for j in range(testLabel.shape[1]):
            temp = np.zeros((1,7),dtype=int)[0]
            pos  = np.nonzero(testLabel[i][j])[0]
            temp[pos] = 1
            maxLen.append(temp)
        trueLabel.append(maxLen)
    trueLabel = np.array(trueLabel)
    return trueLabel

def seventhClass(inputLabel, mask):
    updatedLabel = np.zeros((inputLabel.shape[0],inputLabel.shape[1],7), dtype ='float')
    for i in range(inputLabel.shape[0]):
        for j in range(list(mask[i]).count(1)):
            suM = np.sum(inputLabel[i][j])
            if suM == 0:
                updatedLabel[i][j][6] = 1
            else:
                updatedLabel[i][j][0:6] = inputLabel[i][j]
                updatedLabel[i,j,np.nonzero(updatedLabel[i][j])[0]]=1
    return updatedLabel

def featuresExtraction():
    global train_text, train_audio, train_video, senti_train_label, emo_train_label, train_mask
    global valid_text, valid_audio, valid_video, senti_valid_label, emo_valid_label, valid_mask
    global test_text, test_audio, test_video, senti_test_label, emo_test_label, test_mask
    global max_segment_len

    text          = np.load('MOSEI/text.npz',mmap_mode='r')
    audio         = np.load('MOSEI/audio.npz',mmap_mode='r')
    video         = np.load('MOSEI/video.npz',mmap_mode='r')

    train_text    = text['train_data']
    train_audio   = audio['train_data']
    train_video   = video['train_data']

    valid_text    = text['valid_data']
    valid_audio   = audio['valid_data']
    valid_video   = video['valid_data']

    test_text     = text['test_data']
    test_audio    = audio['test_data']
    test_video    = video['test_data']


    senti_train_label   = video['trainSentiLabel']
    senti_valid_label   = video['validSentiLabel']
    senti_test_label    = video['testSentiLabel']

    senti_train_label   = to_categorical(senti_train_label >= 0)
    senti_valid_label   = to_categorical(senti_valid_label >= 0)
    senti_test_label    = to_categorical(senti_test_label >= 0)

    emo_train_label   = video['trainEmoLabel']
    emo_valid_label   = video['validEmoLabel']
    emo_test_label    = video['testEmoLabel']

    train_length  = video['train_length']
    valid_length  = video['valid_length']
    test_length   = video['test_length']

    max_segment_len = train_text.shape[1]

    train_mask = np.zeros((train_video.shape[0], train_video.shape[1]), dtype='float')
    valid_mask = np.zeros((valid_video.shape[0], valid_video.shape[1]), dtype='float')
    test_mask  = np.zeros((test_video.shape[0], test_video.shape[1]), dtype='float')

    for i in range(len(train_length)):
        train_mask[i,:train_length[i]]=1.0

    for i in range(len(valid_length)):
        valid_mask[i,:valid_length[i]]=1.0

    for i in range(len(test_length)):
        test_mask[i,:test_length[i]]=1.0

    #====================== Add 7th class =========================================
    trainL = seventhClass(emo_train_label, train_mask)
    validL = seventhClass(emo_valid_label, valid_mask)
    testL  = seventhClass(emo_test_label, test_mask)

    #=================== Add multilabel class =====================================
    emo_train_label = emotionClass(trainL)
    emo_valid_label = emotionClass(validL)
    emo_test_label  = emotionClass(testL)

#=================================================================================

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    #y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    #y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    #y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def calc_valid_result_emotion(result, test_label, test_mask):

    true_label=[]
    predicted_label=[]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i,j]==1:
                true_label.append(test_label[i,j])
                predicted_label.append(result[i,j])
    true_label      = np.array(true_label)
    predicted_label = np.array(predicted_label)

    return true_label, predicted_label

def calc_valid_result_sentiment(result, test_label, test_mask):

    true_label=[]
    predicted_label=[]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i,j]==1:
                true_label.append(np.argmax(test_label[i,j] ))
                predicted_label.append(np.argmax(result[i,j] ))
    return true_label, predicted_label

def weighted_accuracy(y_true, y_pred):
    TP, TN, FN, FP, N, P = 0, 0, 0, 0, 0, 0
    for i,j in zip(y_true,y_pred):
        if i == 1 and i == j:
            TP += 1
        elif i == 0 and i == j:
           TN += 1

        if i == 1 and i != j:
            FN += 1
        elif i == 0 and i != j:
           FP += 1

        if i == 1:
            P += 1
        else:
           N += 1

    w_acc = (1.0 * TP * (N / (1.0 * P)) + TN) / (2.0 * N)

    return w_acc, TP, TN, FP, FN, P, N
#=================================================================================
class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        #super().__init__()
        self.test_data = test_data

    def on_train_begin(self, logs={}):
        self.Precision_senti    = []
        self.Recall_senti       = []
        self.Fscore_senti       = []
        self.Accuracy_senti     = []
        self.Weighted_acc_senti = []
        self.Precision_emo      = []
        self.Recall_emo         = []
        self.Fscore_emo         = []
        self.Accuracy_emo       = []
        self.Weighted_acc_emo   = []


    def on_epoch_end(self, epoch, logs={}):
        x_data    = self.test_data[0]
        y_actual  = self.test_data[1]

        #=============================== classification for sentiment ============================

        y_prediction = self.model.predict(x_data)

        true_label_senti, predicted_label_senti = calc_valid_result_sentiment(y_prediction[0], y_actual[0], test_mask)
        w_acc, TP, TN, FP, FN, P, N = weighted_accuracy(true_label_senti, predicted_label_senti)

        #=============================== classification for Emotion ============================
        th=[0.10,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.30,0.35,0.40,0.50]
        for t in range(len(th)):
            print (th[t])
            emotion = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'No Class']
            y_prediction = self.model.predict(x_data)

            y_prediction[1][y_prediction[1] >= th[t]]  = 1
            y_prediction[1][y_prediction[1] <  th[t]]  = 0

            true_label_emo, predicted_label_emo = calc_valid_result_emotion(y_prediction[1], y_actual[1], test_mask)

            Acc_emo     = []
            F1Score_emo = []
            wAcc_emo    = []
            for i in range(7):
                Acc_emo.append(accuracy_score(true_label_emo[:,i], predicted_label_emo[:,i]))
                F1Score_emo.append(precision_recall_fscore_support(true_label_emo[:,i], predicted_label_emo[:,i], average='weighted')[2])
                wAcc_emo.append(weighted_accuracy(true_label_emo[:,i], predicted_label_emo[:,i])[0])

                w_acc, TP, TN, FP, FN, P, N = weighted_accuracy(true_label_emo[:,i], predicted_label_emo[:,i])
                open('results/emotion_'+modality+'.txt', 'a').write('Threshold: ' + str(th[t]) +'\t'+
                                                  str(epoch)+'\t'+
                                                  str(attn_type)+'\t'+
                                                  str(accuracy_score(true_label_emo[:,i], predicted_label_emo[:,i]))+'\t' +
                                                  str(precision_recall_fscore_support(true_label_emo[:,i], predicted_label_emo[:,i], average='weighted')[2])+'\t'+
                                                  str(w_acc) + '\t('+
                                                  str(TP) + ',' + str(TN) + ',' + str(FP) + ',' + str(FN) + ',' + str(P) + ',' + str(N) + ')\t'+
                                                  str(emotion[i])+'\n')

            open('results/emotion_'+modality+'.txt', 'a').write('Threshold: ' + str(th[t]) +'\t'+
                                              str(epoch)+'\t'+
                                              str(attn_type)+'\t'+
                                              str(np.average(Acc_emo)) + '\t'+
                                              str(np.average(F1Score_emo))+ '\t'+
                                              str(np.average(wAcc_emo)) + '\t()'+'\taverage\n')

def CCMA_model(modality, attn_type='mmmu', drops=[0.7, 0.5, 0.5], r_units=300, td_units=100, in_gmu=100, att_gmu=100, in_GMU=True, attn_GMU=True):


    runs = 3

    for i in range(runs):

        drop0  = drops[0]
        drop1  = drops[1]
        r_drop = drops[2]

        ################################### model architecture #################################

        in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
        in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
        in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))

        ########### masking layer ############

        masked_text  = Masking(mask_value=0)(in_text)
        masked_audio = Masking(mask_value=0)(in_audio)
        masked_video = Masking(mask_value=0)(in_video)

        if in_GMU:
            concatAll  = concatenate([masked_text, masked_audio, masked_video])
            concatAll1 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
            concatAll2 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
            concatAll3 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)

            h_text     = TimeDistributed(Dense(in_gmu, activation='tanh'))(masked_text)
            h_audio    = TimeDistributed(Dense(in_gmu, activation='tanh'))(masked_audio)
            h_video    = TimeDistributed(Dense(in_gmu, activation='tanh'))(masked_video)

            o_text     = multiply([h_text, concatAll1])
            o_audio    = multiply([h_audio, concatAll2])
            o_video    = multiply([h_video, concatAll3])

            rnn_text   = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(o_text)
            rnn_audio  = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(o_audio)
            rnn_video  = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(o_video)

        else:
            ########### recurrent layer ############
            rnn_text   = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(masked_text)
            rnn_audio  = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(masked_audio)
            rnn_video  = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat')(masked_video)

        inter_text  = Dropout(drop0)(rnn_text)
        inter_audio = Dropout(drop0)(rnn_audio)
        inter_video = Dropout(drop0)(rnn_video)

        td_text  = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(inter_text))
        td_audio = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(inter_audio))
        td_video = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(inter_video))

        ########### attention layer ############

        ## cross modal cross utterance attention ##
        if attn_type == 'mmmu':

            va_att = attention('simple', td_video, td_audio)
            vt_att = attention('simple', td_video, td_text)
            av_att = attention('simple', td_audio, td_video)
            at_att = attention('simple', td_audio, td_text)
            tv_att = attention('simple', td_text, td_video)
            ta_att = attention('simple', td_text, td_audio)

            if attn_GMU:

                concatAll = concatenate([va_att, vt_att, av_att, at_att, tv_att, ta_att])

                concatAlls1 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlls2 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlls3 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlls4 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlls5 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlls6 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)

                hs_va = TimeDistributed(Dense(att_gmu, activation='tanh'))(va_att)
                hs_vt = TimeDistributed(Dense(att_gmu, activation='tanh'))(vt_att)
                hs_av = TimeDistributed(Dense(att_gmu, activation='tanh'))(av_att)
                hs_at = TimeDistributed(Dense(att_gmu, activation='tanh'))(at_att)
                hs_tv = TimeDistributed(Dense(att_gmu, activation='tanh'))(tv_att)
                hs_ta = TimeDistributed(Dense(att_gmu, activation='tanh'))(ta_att)

                os_va  = multiply([hs_va, concatAlls1])
                os_vt  = multiply([hs_vt, concatAlls2])
                os_av  = multiply([hs_av, concatAlls3])
                os_at  = multiply([hs_at, concatAlls4])
                os_tv  = multiply([hs_tv, concatAlls5])
                os_ta  = multiply([hs_ta, concatAlls6])

                merged_s = concatenate([os_va, os_vt, os_av, os_at, os_tv, os_ta, td_video, td_audio, td_text])

                concatAlle1 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlle2 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlle3 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlle4 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlle5 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)
                concatAlle6 = TimeDistributed(Dense(1, activation='sigmoid'))(concatAll)

                he_va = TimeDistributed(Dense(att_gmu, activation='tanh'))(va_att)
                he_vt = TimeDistributed(Dense(att_gmu, activation='tanh'))(vt_att)
                he_av = TimeDistributed(Dense(att_gmu, activation='tanh'))(av_att)
                he_at = TimeDistributed(Dense(att_gmu, activation='tanh'))(at_att)
                he_tv = TimeDistributed(Dense(att_gmu, activation='tanh'))(tv_att)
                he_ta = TimeDistributed(Dense(att_gmu, activation='tanh'))(ta_att)

                oe_va  = multiply([he_va, concatAlle1])
                oe_vt  = multiply([he_vt, concatAlle2])
                oe_av  = multiply([he_av, concatAlle3])
                oe_at  = multiply([he_at, concatAlle4])
                oe_tv  = multiply([he_tv, concatAlle5])
                oe_ta  = multiply([he_ta, concatAlle6])

                merged_e = concatenate([oe_va, oe_vt, oe_av, oe_at, oe_tv, oe_ta, td_video, td_audio, td_text])
                output_sentiment = TimeDistributed(Dense(2, activation='softmax'), name='output_sentiment')(merged_s)
                output_emotion   = TimeDistributed(Dense(7, activation='sigmoid'), name='output_emotion')(merged_e)

            else:
                merged = concatenate([va_att, vt_att, av_att, at_att, tv_att, ta_att, td_video, td_audio, td_text])
                output_sentiment = TimeDistributed(Dense(2, activation='softmax'), name='output_sentiment')(merged)
                output_emotion   = TimeDistributed(Dense(7, activation='sigmoid'), name='output_emotion')(merged)

        ########### output layer ############

        model = Model([in_text, in_audio, in_video], [output_sentiment, output_emotion])
        model.compile(optimizer='adam', loss={'output_sentiment':'categorical_crossentropy', 'output_emotion':'binary_crossentropy'}, sample_weight_mode='temporal', metrics = {'output_sentiment': ['accuracy', f1, precision, recall],'output_emotion': 'accuracy'})

        ###################### model training #######################
        np.random.seed(i)

        metrics_callback = MetricsCallback(test_data=([test_text, test_audio, test_video], [senti_test_label, emo_test_label]))

        a1 = '_drops_' + str(drops[0]) +'_'+ str(drops[1]) +'_'+ str(drops[2]) + '_'
        a2 = 'td_units_' + str(td_units) +'_'
        a3 = 'r_units_' + str(r_units) +'_'
        a4 = 'in_GMU_' + str(in_GMU) + '_' + str(in_gmu) + '_'
        a5 = 'attn_GMU_' + str(attn_GMU) + '_' + str(att_gmu) + '_'

        path1  = 'weights/sentiment_'+ modality + '_' + attn_type + a1 + a2 + a3 + a4 + a5 + 'run_' + str(i) + '.hdf5'
        path2  = 'weights/emotion_'+ modality + '_' + attn_type + a1 + a2 + a3 + a4 + a5 + 'run_' + str(i) + '.hdf5'

        check1 = EarlyStopping(monitor='val_output_sentiment_loss', patience=30)
        check2 = EarlyStopping(monitor='val_output_emotion_loss', patience=30)
        check3 = ModelCheckpoint(path1, monitor='val_output_sentiment_acc', verbose=1, save_best_only=True, mode='max')
        check4 = ModelCheckpoint(path2, monitor='val_output_emotion_acc', verbose=1, save_best_only=True, mode='max')

        history = model.fit([train_text, train_audio, train_video], [senti_train_label, emo_train_label],
                            epochs=100,
                            batch_size=16,
                            sample_weight=[train_mask, train_mask],
                            shuffle=True,
                            callbacks=[metrics_callback,check3,check4],
                            validation_data=([test_text, test_audio, test_video], [senti_test_label,emo_test_label], [test_mask,test_mask]),
                            verbose=1)

        acc   = max(history.history['val_output_sentiment_acc'])

        model.load_weights(path1)
        result = model.predict([test_text, test_audio, test_video])
        np.ndarray.dump(result[0],open('sentiment_for_errorAnalysis'+ modality + '_' + attn_type + a1 + a2 + a3 + a4 + a5 + 'run_' + str(i) + '.np', 'wb'))

        model.load_weights(path2)
        result = model.predict([test_text, test_audio, test_video])
        np.ndarray.dump(result[1],open('emotion_for_errorAnalysis'+ modality + '_' + attn_type + a1 + a2 + a3 + a4 + a5 + 'run_' + str(i) + '.np', 'wb'))

        best_accuracy, f1Score = calc_valid_result(result[0],senti_test_label,test_mask)
        open('results/sentiment_'+modality+'.txt', 'a').write(modality +
                                          ', drops: ' + str(drops[0]) +'_'+ str(drops[1]) +'_'+ str(drops[2]) +
                                          ', r_units: ' + str(r_units) +
                                          ', td_units: ' + str(td_units) +
                                          ', in_GMU: ' + str(in_GMU) + '_' + str(in_gmu) +
                                          ', attn_GMU: ' + str(attn_GMU) + '_' + str(att_gmu) +
                                          ', accuracy: ' + str(acc) +'\t'+
                                          ', f1_Score: ' + str(f1Score[2]) + '\n'*2)


        ################### release gpu memory ##################
        K.clear_session()
        del model
        del history
        gc.collect()


featuresExtraction()
for drop in [0.3]:
    for rdrop in [0.3]:
        for r_units in [300]:
            for td_units in [100]:
                for in_gmu in [80]:
                    for att_gmu in [10]:
                        for in_GMU in [True]:
                            for attn_GMU in [True]:
                    	        modality  = 'trimodal'+ '_' + str(in_GMU) + ':' + str(in_gmu) + '_' + str(attn_GMU) + ':' + str(att_gmu)
                    	        attn_type = 'mmmu'
                                CCMA_model(modality, attn_type=attn_type, drops=[drop, drop, rdrop], r_units=r_units, td_units=td_units, in_gmu=in_gmu, att_gmu=att_gmu, in_GMU=in_GMU, attn_GMU=attn_GMU)
