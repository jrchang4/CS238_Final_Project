import rlcard
from rlcard import models

from rlcard.utils import set_global_seed
from rlcard.games.gin_rummy.player import GinRummyPlayer
from rlcard.agents import RandomAgent
from rlcard.agents.mcts_agent import MCTSAgent
from rlcard.games.gin_rummy.utils.move import DealHandMove
from rlcard.utils import utils as ut
from rlcard.utils import Logger


# Make environment
env = rlcard.make('gin-rummy', config={'seed': 0, 'allow_step_back': True})
episode_num = 1
evaluate_num = 1
evaluate_every = 100
log_dir = './experiments/gin_rummy_mcts_result/'
env.game.settings.print_settings()

# Set a global seed
set_global_seed(0)

# Set up agents
agent = MCTSAgent(kmax=1000, c=.6)
random_agent = RandomAgent(action_num=env.action_num)
env.set_agents([agent, random_agent])

# Init a Logger to plot the learning curve
logger = Logger(log_dir)

for episode in range(episode_num):

    # Generate data from the environment
    #trajectories, _ = env.run(is_training=False)

    # extra logging
    if episode % evaluate_every == 0:
        reward = 0
        reward2 = 0
        for eval_episode in range(evaluate_num):
            trajectories, payoffs = env.run(is_training=False)
            reward += payoffs[0]
            reward2 += payoffs[1]
        logger.log("\n\n########## Evaluation {} ##########".format(episode))
        reward_text = "{}".format(float(reward)/evaluate_num)
        reward2_text = "{}".format(float(reward2)/evaluate_num)
        info = "Timestep: {} Average reward is {}, reward2 is {}".format(env.timestep, reward_text, reward2_text)
        logger.log(info)

        # print move sheet
        print("\n========== Move Sheet ==========")
        move_sheet = env.game.round.move_sheet
        move_sheet_count = len(move_sheet)
        for i in range(move_sheet_count):
            move = move_sheet[i]
            print("{}".format(move))
            if i == 0 and isinstance(move, DealHandMove):
                player_dealing_id = move.player_dealing.player_id
                leading_player_id = GinRummyPlayer.opponent_id_of(player_dealing_id)
                shuffle_deck = move.shuffled_deck
                #leading_player_hand_text = [str(card) for card in shuffle_deck[-11:]]
                leading_player_hand_text = [str(card) for card in shuffle_deck[-(ut.NUM_CARDS+1):]]
                #dealing_player_hand_text = [str(card) for card in shuffle_deck[-21:-11]]
                dealing_player_hand_text = [str(card) for card in shuffle_deck[-(ut.NUM_CARDS*2+1):-(ut.NUM_CARDS+1)]]
                #stock_pile_text = [str(card) for card in shuffle_deck[:31]]
                stock_pile_text = [str(card) for card in shuffle_deck[:52-(ut.NUM_CARDS*2+1)]]
                short_name_of_player_dealing = GinRummyPlayer.short_name_of(player_id=player_dealing_id)
                short_name_of_player_leading = GinRummyPlayer.short_name_of(player_id=leading_player_id)
                print("player_dealing is {}; leading_player is {}.".format(short_name_of_player_dealing,
                                                                           short_name_of_player_leading))
                print("leading player hand: {}".format(leading_player_hand_text))
                print("dealing player hand: {}".format(dealing_player_hand_text))
                print("stock_pile: {}".format(stock_pile_text))

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(env, evaluate_num)[0])

    # # Print out the trajectories
    # print('\nEpisode {}'.format(episode))
    # for ts in trajectories[0]:
    #     print('State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.format(ts[0], ts[1], ts[2], ts[3], ts[4]))

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('DQN')