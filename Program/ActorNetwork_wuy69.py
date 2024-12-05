import tensorflow as tf
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        # 创建模型
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)

        # 优化器
        self.optimizer = Adam(learning_rate=LEARNING_RATE)

    def create_actor_network(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1, activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4))(h1)
        Acceleration = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4))(h1)
        Brake = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4))(h1)
        V = Concatenate()([Steering, Acceleration, Brake])
        model = Model(inputs=S, outputs=V)
        return model, model.trainable_weights, S

    def train(self, states, gradients):
        """训练演员模型"""
        with tf.GradientTape() as tape:
            # 预测当前状态的动作
            predictions = self.model(states, training=True)
        
        # 计算用于更新模型的梯度
        actor_gradients = tape.gradient(predictions, self.model.trainable_variables, -gradients)
        
        # 应用梯度
        self.optimizer.apply_gradients(zip(actor_gradients, self.model.trainable_variables))
        
    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1, activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4))(h1)
        Acceleration = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4))(h1)
        Brake = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4))(h1)
        concat_layer=Concatenate()
        V = concat_layer([Steering,Acceleration,Brake])          
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S

