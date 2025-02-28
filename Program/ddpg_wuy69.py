import gym
from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from tensorflow.python.keras.engine.training import collect_trainable_weights

import json



from ReplayBuffer import ReplayBuffer


from ActorNetwork_wuy69 import ActorNetwork
from CriticNetwork_wuy69 import CriticNetwork
from OU import OU
import timeit

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 8  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 1000
    episode_count = 1000
    max_steps = 1000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    actor = ActorNetwork(state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    
    
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    
    

    # Load model if exists
    print("Now we load the weight")
    try:
        actor.model = tf.keras.models.load_model(r"C:\Users\Administrator\Desktop\vandy1st\Reinforcement-Learning\homework\Project\DDPG-Keras-Torcs-master\actormodel.h5")
        critic.model = tf.keras.models.load_model(r"C:\Users\Administrator\Desktop\vandy1st\Reinforcement-Learning\homework\Project\DDPG-Keras-Torcs-master\criticmodel.h5")
        actor.target_model = tf.keras.models.load_model(r"C:\Users\Administrator\Desktop\vandy1st\Reinforcement-Learning\homework\Project\DDPG-Keras-Torcs-master\actormodel.h5")
        critic.target_model = tf.keras.models.load_model(r"C:\Users\Administrator\Desktop\vandy1st\Reinforcement-Learning\homework\Project\DDPG-Keras-Torcs-master\criticmodel.h5")
        print("Weight load successfully")
    except Exception as e:
        print("Cannot find the weight, Error:", e)
    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        ob = env.reset()
        angle=ob[0]
        track=ob[1]
        trackPos=ob[2]
        speedX=ob[3]
        speedY=ob[4]
        speedZ=ob[5]
        wheelSpinVel=ob[6]
        rpm=ob[7]
        

        s_t = np.hstack((angle, track, trackPos, speedX, speedY, speedZ, wheelSpinVel/100.0, rpm))
     
        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])
            angle=ob[0]
            track=ob[1]
            trackPos=ob[2]
            speedX=ob[3]
            speedY=ob[4]
            speedZ=ob[5]
            wheelSpinVel=ob[6]
            rpm=ob[7]

            s_t1 = np.hstack((angle, track, trackPos, speedX, speedY, speedZ, wheelSpinVel/100.0, rpm))
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                action_grads = critic.get_action_gradients(states, a_for_grad)
                actor.train(states, action_grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        if np.mod(i, 3) == 0:  # Save model every 3 episodes
            if train_indicator:
                print("Now we save model")
                actor.model.save('actormodel.h5')
                critic.model.save('criticmodel.h5')

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
