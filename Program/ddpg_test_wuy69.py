from gym_torcs import TorcsEnv
import numpy as np
import tensorflow as tf
from ActorNetwork_wuy69 import ActorNetwork
from CriticNetwork_wuy69 import CriticNetwork

def playGame(train_indicator=0):    # 1 means Train, 0 means simply Run
    TAU = 0.001     # Target Network HyperParameters
    LRA = 0.0001    # Learning rate for Actor

    action_dim = 3  # Steering/Acceleration/Brake
    state_dim = 8  # of sen
    vision = False
    episode_count = 1
    max_steps = 1000 # 100000

    actor = ActorNetwork(state_dim, action_dim, 1, TAU, LRA)
    critic = CriticNetwork(state_dim, action_dim, 1, TAU, LRA)

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    # Now load the weight
    print("Now we load Actor model's weights")
    try:
        actor.model.load_weights(r"C:\Users\Administrator\Desktop\vandy1st\Reinforcement-Learning\homework\Project\DDPG-Keras-Torcs-master\actormodel.h5")
        critic.model.load_weights(r'C:\Users\Administrator\Desktop\vandy1st\Reinforcement-Learning\homework\Project\DDPG-Keras-Torcs-master\criticmodel.h5')
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):
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
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            ob, r_t, done, info = env.step(a_t_original[0])
            angle=ob[0]
            track=ob[1]
            trackPos=ob[2]
            speedX=ob[3]
            speedY=ob[4]
            speedZ=ob[5]
            wheelSpinVel=ob[6]
            rpm=ob[7]
            s_t1 = np.hstack((angle, track, trackPos, speedX, speedY, speedZ, wheelSpinVel/100.0, rpm))

            total_reward += r_t
            s_t = s_t1

            if done:
                break

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame(train_indicator=0)
