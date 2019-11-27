"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""
#-*-coding:utf-8-*-

import os
import numpy as np
import tensorflow as tf
import buffer_a3c
import buffer_load_trace
import math
import buffer_env


S_INFO = 3
S_LEN = 1
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 20
ACTION_SPACE = ['a+b~c-','a-b~c+','a~b+c-','a~b-c+','a+b-c~','a-b+c~']  # Kbps
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
GRADIENT_BATCH_SIZE = 16
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = None
# gamma_ARE = 0.9

# def compute_reward(ARE_list):
#     reward = 0
#     for ii in len(ARE_list):
#         reward += ARE_list[ii] * math.pow(gamma_ARE, ii)
#     return reward

def main():

    np.random.seed(RANDOM_SEED)

    assert len(ACTION_SPACE) == A_DIM

    heavy_size, buffer_size, ARE = buffer_load_trace.load_trace() # reading files
    light_size = [600 - t[0] - t[1] for t in zip(heavy_size, buffer_size)]
    # buffer_combine is a two-dimentional list, each row corresponds a [heavy, buffer, light]--the length of dataset
    buffer_combine = [list(z) for z in zip(heavy_size, buffer_size, light_size)] # combine three buffer distribute to buffer_combine

    env_net = buffer_env.Environment(buffer_combine, ARE, len(buffer_combine), RANDOM_SEED) # building the enviroment of the reinforcement learning

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

        epoch = 0

        action_vec = np.zeros(A_DIM)
        action_index = 1
        action_vec[action_index] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        actor_gradient_batch = []
        critic_gradient_batch = []
        ARE_list = []

        buffer_distrib, are, end_of_dataset = env_net.get_initial_size()

        while True:

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

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(buffer_a3c.compute_entropy(action_prob[0]))

            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_dataset:  # do training once

                actor_gradient, critic_gradient, td_batch = \
                    buffer_a3c.compute_gradients(s_batch=np.stack(s_batch[1:], axis=0),  # ignore the first chuck
                                          a_batch=np.vstack(a_batch[1:]),  # since we don't have the
                                          r_batch=np.vstack(r_batch[1:]),  # control over it
                                          terminal=end_of_dataset, actor=actor, critic=critic)
                td_loss = np.mean(td_batch)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                print "===="
                print "Epoch", epoch
                print "TD_loss", td_loss, "Avg_reward", np.mean(r_batch), "Avg_entropy", np.mean(entropy_record)
                print "===="

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: td_loss,
                    summary_vars[1]: np.mean(r_batch),
                    summary_vars[2]: np.mean(entropy_record)
                })

                writer.add_summary(summary_str, epoch)
                writer.flush()

                entropy_record = []

                if len(actor_gradient_batch) >= GRADIENT_BATCH_SIZE:

                    assert len(actor_gradient_batch) == len(critic_gradient_batch)
                    # assembled_actor_gradient = actor_gradient_batch[0]
                    # assembled_critic_gradient = critic_gradient_batch[0]
                    # assert len(actor_gradient_batch) == len(critic_gradient_batch)
                    # for i in xrange(len(actor_gradient_batch) - 1):
                    #     for j in xrange(len(actor_gradient)):
                    #         assembled_actor_gradient[j] += actor_gradient_batch[i][j]
                    #         assembled_critic_gradient[j] += critic_gradient_batch[i][j]
                    # actor.apply_gradients(assembled_actor_gradient)
                    # critic.apply_gradients(assembled_critic_gradient)

                    for i in xrange(len(actor_gradient_batch)):
                        actor.apply_gradients(actor_gradient_batch[i])
                        critic.apply_gradients(critic_gradient_batch[i])

                    actor_gradient_batch = []
                    critic_gradient_batch = []

                    epoch += 1
                    if epoch % MODEL_SAVE_INTERVAL == 0:
                        # Save the neural net parameters to disk.
                        save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_buffer_" +
                                               str(epoch) + ".ckpt")
                        print("Model saved in file: %s" % save_path)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

            if end_of_dataset:
                break

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[action] = 1
                a_batch.append(action_vec)

            buffer_distrib, are, end_of_dataset = env_net.get_next_size(ACTION_SPACE[action])

if __name__ == '__main__':
    main()
