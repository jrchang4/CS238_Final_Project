'''
    File name: rlcard.examples.gin_rummy_dqn.py
    Author: William Hale
    Date created: 2/12/2020

    An example of learning a Deep-Q Agent on GinRummy
'''

import tensorflow as tf
import os

import rlcard

from rlcard import models
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from tensorboardX import SummaryWriter
from rlcard.games.gin_rummy.utils.move import DealHandMove
from rlcard.games.gin_rummy.player import GinRummyPlayer
import datetime
import pytz

# Make environment
env = rlcard.make('gin-rummy')
eval_env = rlcard.make('gin-rummy')

#config = {'seed':0}
env.game.settings.print_settings()

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
evaluate_num = 100  # mahjong_dqn has 1000
episode_num = 10000 # mahjong_dqn has 100000

# The initial memory size
memory_init_size = 10000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/gin_rummy_dqn_result/'

# Set a global seed
#set_global_seed(0)

utc_now = pytz.utc.localize(datetime.datetime.utcnow())
pst_now = utc_now.astimezone(pytz.timezone("America/Los_Angeles"))    
current_time = pst_now.strftime("%Y-%m-%d-%H-%M-%S")
log_dir = '../tensorboard_logs/' + current_time
print(current_time)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

with tf.Session() as sess:
    # Set agents
    global_step = tf.Variable(0, name='global_step', trainable=False)
    agent = DQNAgent(sess,
                     scope='dqn',
                     action_num=env.action_num,
                     replay_memory_size=20000,
                     replay_memory_init_size=memory_init_size,
                     update_target_estimator_every=250,
                     train_every=train_every,
                     state_shape=env.state_shape,
                     learning_rate=0.5,
                     mlp_layers=[512*8, 512*8, 512*4, 512*4, 512*2, 512*2, 512, 512])

    # 512*16,512*16, 512*8, 512*8, 512*4, 512*4, 512*2, 512*2, 512, 512

    random_agent = RandomAgent(action_num=eval_env.action_num)
    novice_agent = models.load("gin-rummy-novice-rule").agents[0]

    sess.run(tf.global_variables_initializer())

    env.set_agents([agent, random_agent])
    eval_env.set_agents([agent, random_agent])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    for episode in range(episode_num):

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        """

        move_sheet = env.game.round.move_sheet
        move_sheet_count = len(move_sheet)
        move_sheet_count = len(move_sheet)
        for i in range(move_sheet_count):
            move = move_sheet[i]
            print("{}".format(move))
            if i == 0 and isinstance(move, DealHandMove):
                player_dealing_id = move.player_dealing.player_id
                leading_player_id = GinRummyPlayer.opponent_id_of(player_dealing_id)
                shuffle_deck = move.shuffled_deck
                leading_player_hand_text = [str(card) for card in shuffle_deck[-11:]]
                dealing_player_hand_text = [str(card) for card in shuffle_deck[-21:-11]]
                stock_pile_text = [str(card) for card in shuffle_deck[:31]]
                short_name_of_player_dealing = GinRummyPlayer.short_name_of(player_id=player_dealing_id)
                short_name_of_player_leading = GinRummyPlayer.short_name_of(player_id=leading_player_id)
                print("player_dealing is {}; leading_player is {}.".format(short_name_of_player_dealing,
                                                                       short_name_of_player_leading))
                print("leading player hand: {}".format(leading_player_hand_text))
                print("dealing player hand: {}".format(dealing_player_hand_text))
                print("stock_pile: {}".format(stock_pile_text))
        """
   

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent.feed(ts)

        # extra logging
        if episode % evaluate_every == 0:
            reward = 0
            reward2 = 0
            for eval_episode in range(evaluate_num):
                _, payoffs = eval_env.run(is_training=False)
                reward += payoffs[0]
                reward2 += payoffs[1]
            logger.log("\n\n########## Evaluation {} ##########".format(episode))
            reward_text = "{}".format(float(reward)/evaluate_num)
            reward2_text = "{}".format(float(reward2)/evaluate_num)
            info = "Timestep: {} Average reward is {}, reward2 is {}".format(env.timestep, reward_text, reward2_text)
            logger.log(info)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            player1, player2 = tournament(eval_env, evaluate_num)
            logger.log_performance(env.timestep, player1 - player2)
            writer.add_scalar('Reward', player1 - player2, env.timestep)

    writer.close()

    # Close files in the logger
    logger.close_files()


    # Plot the learning curve
    logger.plot('DQN')

    # Save model
    save_dir = 'models/gin_rummy_dqn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
