import time, math
import numpy as np
from rlcard.games.gin_rummy.utils.action_event import ActionEvent
from rlcard.games.gin_rummy.utils.gin_rummy_error import GinRummyProgramError
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent as novice
from copy import copy

class TS:
    def __init__(self):
        self.d = 10
        self.gamma = .999
        self.N = {}
        self.Q = {}

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
        self.use_raw = False
        self.kmax = kmax
        self.c = c

    def step(self, state): #TODO: DONT NEED THIS ?
        print("###BAD### MCTS STEP CALLED")
        action = np.random.choice(np.arange(len(state['legal_actions'])), p=state['legal_actions'])
        return action

    def rollout(self, MCTS, s, player_id, d, sim_env):
        if d <= 0:
            return 0

        print("ROLLOUT PLAYER ID: ", str(player_id))

        a = novice.step(s) #TODO: IS OUR ROLLOUT POLICY JUST THE NOVICE POLICY?
        print("ROLLOUT ACTION: %s" % a)
        #sp, r = MCTS.TR[(s, a)] #TODO: HOW DO WE GENERATE THE NEXT STATE RANDOMLY FROM TR
        sp, next_player_id = sim_env.step(a, False)
        print("ROLLOUT NEXT PLAYER ID: %s" % next_player_id)
        r = 0
        if sim_env.game.is_over():
            r == sim_env.get_payoffs()[player_id] #TODO: IS THIS CORRECT

        result = r + MCTS.gamma * self.rollout(MCTS, sp, next_player_id, d - 1, sim_env)
        print("ROLLOUT RESULT: ", str(result))
        return result

    def bonus(self, Nsa, ns):
        return float("inf") if Nsa == 0 else sqrt(log(Ns)/ Nsa)

    def hashable_state(self, state):
        result = "".join(map(str, state['obs'][0]))
        result += "".join(map(str, state['obs'][1]))
        return result

    def explore(self, MCTS, s):
        A = s['legal_actions']
        print("LEGAL ACTIONS: ", A)
        print("HAND: ", s['obs'][0])
        N, Q = MCTS.N, MCTS.Q
        hashable_state = self.hashable_state(s)
        Ns = sum([N[(hashable_state, a)] for a in A])
        UCBs = {Q[(hashable_state, a)] + (self.c * self.bonus(N[(hashable_state, a)], Ns)) : a for a in A}
        return UCBs[max(UCBs.keys())]

    def simulate(self, MCTS, s, player_id, sim_env, d): #TODO: SHOULD WE ROLLOUT UNTIL GAME OVER?
        if d <= 0:
            return 

        print("SIMULATE PLAYER ID: ", str(player_id))

        gamma, N, Q = MCTS.gamma, MCTS.N, MCTS.Q
        A = s['legal_actions']

        hashable_state = self.hashable_state(s)
        if not (hashable_state, A[0]) in N:
            for a in A:
                N[(hashable_state,a)] = 0
                Q[(hashable_state,a)] = 0
            return self.rollout(MCTS, s, player_id, d, sim_env)

        a = self.explore(MCTS, s)
        print("SIMULATE ACTION: ", str(a))
        sp, next_player_id = sim_env.step(a, False)
        print("NEXT PLAYER_ID: %s" % next_player_id)
        #sp, r = TR[(s, a)] #TODO: HOW DO WE GENERATE THE NEXT STATE RANDOMLY FROM TR
        r = 0
        if sim_env.game.is_over():
            r == sim_env.get_payoffs()[player_id] #TODO: IS THIS CORRECT
        q = r + gamma*self.simulate(MCTS, sp, next_player_id, sim_env, d - 1) #TODO: when do we make fresh copies??? 
        N[(hashable_state, a)] += 1
        Q[(hashable_state, a)] += (q - Q[(hashable_state, a)]) / N[(hashable_state, a)]

        return q
 
    def eval_step(self, arg):
        ''' Predict the action given the current state for evaluation.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted by the MCTS agent
        '''
        state, player_id, sim_env = arg[0], arg[1], arg[2]
        MCTS = TS()
        for k in range(self.kmax):
            self.simulate(MCTS, state, player_id, copy(sim_env), 15)

        argmax = None
        maxQ = -100
        for a in state['legal_actions']:
            possible = MCTS.Q[(self.hashable_state(s), a)]
            if possible > maxQ:
                maxQ = possible
                argmax = a
        return argmax #return action that maximizes Q