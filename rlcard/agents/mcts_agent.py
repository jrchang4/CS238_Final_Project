import time, math
import numpy as np
from rlcard.games.gin_rummy.utils.action_event import ActionEvent
from rlcard.games.gin_rummy.utils.gin_rummy_error import GinRummyProgramError
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent as novice

class MCTS(object):
    def __init__(self):
        self.d = 10

    # def TR(s, a):
    #     state = {}
    #     reward = 0
    #     if a == 0: #return state prime w/ 100% confidence
    #         state['legal_actions'] = np.array([0 for i in range(52)])
    #     return state, reward

class MCTSAgent(object):
    ''' A human agent for Gin Rummy. It can be used to play against trained models.
    '''

    def __init__(self, c, kmax=500):
        ''' Initialize the human agent

        Args:
            action_num (int): the size of the output action space
        '''
        self.use_raw = True
        self.kmax = kmax
        self.c = c

    def step(self, state): #TODO: DONT NEED THIS ?
        print("###BAD### MCTS STEP CALLED")
        action = np.random.choice(np.arange(len(state['legal_actions'])), p=state['legal_actions'])
        return action

    def eval_step(self, state, player_id, sim_env):
        ''' Predict the action given the current state for evaluation.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted by the MCTS agent
        '''
        print("MCTS STATE:", state)
        for k in self.kmax:
            simulate(MCTS, state, player_id, deep_copy(sim_env))

        return np.argmax(MCTS.Q) #return action that maximizes Q

    def simulate(MCTS, s, sim_env, d=10):
        if d <= 0:
            return 0

        gamma, N, Q, c = MCTS.gamma, MCTS.N, MCTS.Q, MCTS.c
        A = s['legal_actions']

        if not N.has_key((s, A[0])):
            for a in A:
                N[(s,a)] = 0
                Q[(s,a)] = 0
            return rollout(MCTS, s, d, sim_env)

        a = explore(MCTS, s)
        sp, next_player_id = sim_env.step(action, sim_env.agents[player_id].use_raw)
        #sp, r = TR[(s, a)] #TODO: HOW DO WE GENERATE THE NEXT STATE RANDOMLY FROM TR
        r = 0
        if sim_env.game.is_over():
            r == sim_env.get_payoffs()[player_id] #TODO: IS THIS CORRECT
        q = r + gamma*simulate(MCTS, sp, sim_env d - 1) #TODO: when do we make fresh copies??? 
        N[(s, a)] += 1
        Q[(s, a)] += (q - Q[(s, a)]) / N[(s, a)]

        return 1

    def rollout(MCTS, s, d, sim_env):
        if d <= 0:
            return 0

        a = novice.step(s) #TODO: IS OUR ROLLOUT POLICY JUST THE NOVICE POLICY?
        #sp, r = MCTS.TR[(s, a)] #TODO: HOW DO WE GENERATE THE NEXT STATE RANDOMLY FROM TR
        sp, next_player_id = sim_env.step(action, sim_env.agents[player_id].use_raw)
        r = 0
        if sim_env.game.is_over():
            r == sim_env.get_payoffs()[player_id] #TODO: IS THIS CORRECT
        return r + MCTS.gamma * rollout(MCTS, sp, d - 1)

    def explore(MCTS, s):
        A, N, Q, c = MCTS.A, MCTS.N, MCTS.Q, MCTS.c
        Ns = sum([N[(s, a)] for a in A])
        UCBs = [Q[(s, a)] + (c * bonus(N[(s, a)], Ns)) for a in A]
        return argmax(UCBs)

    def bonus(Nsa, ns):
        return float("inf") if Nsa == 0 else sqrt(log(Ns)/ Nsa)