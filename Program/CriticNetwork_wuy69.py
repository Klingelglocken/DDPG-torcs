import tensorflow as tf
from keras.layers import Dense, Input, Add
from keras.models import Model
from keras.optimizers import Adam

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(object):
    def __init__(self, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        # 创建模型
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        
        # 优化器
        self.optimizer = Adam(learning_rate=self.LEARNING_RATE)

    def create_critic_network(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        
        add_layer = Add()
        h2 = add_layer([h1, a1])

        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim, activation='linear')(h3)
        model = Model(inputs=[S, A], outputs=V)
        return model, A, S
    
    def get_action_gradients(self, states, actions):
        """计算 Q 值相对于动作的梯度"""
        # 确保 actions 是一个 TensorFlow 张量
        actions = tf.convert_to_tensor(actions)

        with tf.GradientTape() as tape:
            # 记录动作输入以获取梯度
            tape.watch(actions)
            # 预测当前状态和动作的 Q 值
            Q_values = self.model([states, actions], training=True)
        # 计算 Q 值相对于动作的梯度
        return tape.gradient(Q_values, actions)

    def train(self, states, actions, y):
        with tf.GradientTape() as tape:
            predictions = self.model([states, actions], training=True)
            loss = tf.keras.losses.MSE(y, predictions)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)

        # 替换 merge 调用
        add_layer = Add()
        h2 = add_layer([h1, a1])

        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim, activation='linear')(h3)

        # 更新 Model 构造函数的参数
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(learning_rate=self.LEARNING_RATE)

        model.compile(loss='mse', optimizer=adam)
        return model, A, S
