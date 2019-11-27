"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""

import os
import numpy as np
import tensorflow as tf
import buffer_a3c
import buffer_load_trace
import math
import buffer_env
import random

S_INFO = 3
S_LEN = 1
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
ACTION_SPACE = ['a+b~c-','a-b~c+','a~b+c-','a~b-c+','a+b-c~','a-b+c~']  # Kbps
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
GRADIENT_BATCH_SIZE = 16
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = None
NN_MODEL = './buffer_models/nn_model_buffer_60.ckpt'
TOTAL_BUFFER = 600
gamma_ARE = 0.9
DATA_FOLDER = './dataset_test_sim'
LOOK_FOLDER = './dataset'

# def compute_reward(ARE_list):
#     reward = 0
#     for ii in len(ARE_list):
#         reward += ARE_list[ii] * math.pow(gamma_ARE, ii)
#     return reward

def load_test_data(test_folder):
    test_distribute = []
    test_ARE = []
    with open(test_folder, 'rb') as f:
        for line in f:
            parse = line.split()
            test_distribute.append([float(parse[0]), float(parse[1]), TOTAL_BUFFER - float(parse[0]) - float(parse[1])])
            test_ARE.append(np.float32(parse[2]))

    return test_distribute, test_ARE

def main():

    np.random.seed(RANDOM_SEED)

    assert len(ACTION_SPACE) == A_DIM

    test_distribute, test_ARE = load_test_data(test_folder=DATA_FOLDER)

    look_heavy_size, look_buffer_size, look_ARE = buffer_load_trace.load_trace(train_folder=LOOK_FOLDER)
    look_light_size = [600 - t[0] - t[1] for t in zip(look_heavy_size, look_buffer_size)]
    look_buffer_combine = [list(z) for z in zip(look_heavy_size, look_buffer_size, look_light_size)]
    #random.shuffle(look_buffer_combine)

    look_net = buffer_env.Environment(look_buffer_combine, look_ARE, len(look_buffer_combine), RANDOM_SEED)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    with tf.Session() as sess, open(LOG_FILE, 'wb') as log_file:

        actor = buffer_a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = buffer_a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = buffer_a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        for ii in range(len(test_distribute)):

            buffer_distrib, are = test_distribute[ii], test_ARE[ii]
            total_enfore = 0
            ARE_list = []
            action_vec = np.zeros(A_DIM)
            action_index = 1
            action_vec[action_index] = 1

            s_batch = [np.zeros((S_INFO, S_LEN))]
            a_batch = [action_vec]
            r_batch = []

            entropy_record = []
            print('Test data: ', test_distribute[ii])
            print('Initial ARE: ', test_ARE[ii])

            while total_enfore <= 50 and are != 0:

                ARE_list.append(are)

                if len(ARE_list) == 1:
                    reward = ARE_list[-1]
                else:
                    if ARE_list[-2] >= ARE_list[-1]:
                        reward = ARE_list[-2]/ARE_list[-1]
                    else:
                        reward = - ARE_list[-1]/ARE_list[-2]

                r_batch.append(reward)

                # retrieve previous state
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                state[0, 0] = buffer_distrib[0]
                state[1, 0] = buffer_distrib[1]
                state[2, 0] = buffer_distrib[2]

                action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN))) # length 6, output after softmax
                action_cumsum = np.cumsum(action_prob) # Action list is the same as original format
                action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                entropy_record.append(buffer_a3c.compute_entropy(action_prob[0]))

                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[action] = 1
                a_batch.append(action_vec)

                total_enfore = total_enfore + 1
                print(buffer_distrib, are)

                buffer_distrib, are, end_of_dataset = look_net.get_next_size(ACTION_SPACE[action], pre_select=buffer_distrib, train_flag=False)

if __name__ == '__main__':
    main()