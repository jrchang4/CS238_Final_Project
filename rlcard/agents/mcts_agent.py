import time, math
import numpy as np
from rlcard.games.gin_rummy.utils.action_event import *
from rlcard.games.gin_rummy.utils.gin_rummy_error import GinRummyProgramError
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent as novice
from rlcard.games.gin_rummy.utils import utils, melding
from rlcard.games.gin_rummy import judge

class TS:
    def __init__(self):
        self.d = 10
        self.gamma = .999
        self.N = {}
        self.Q = {}

    def TR(self, s, a, player_id, won_id):
        state = {}
        state['legal_actions'] = []
        state['obs'] = s['obs']
        reward = 0
        hand = utils.decode_cards(s['obs'][0])
        print("TR HAND: ", [str(card) for card in hand])
        if a == 0: #score player 0
            if won_id == 0:
                reward = 1
            else:
                deadwood_values = [utils.get_deadwood_value(card) for card in hand] #ONLY IF WE DIDNT WIN
                reward = -1 * sum(deadwood_values) / 100
            state['obs'] = np.ndarray(shape=(2,52), dtype=int)
            state['legal_actions'] = [1]
        elif player_id == 0 and (a == 2 or a == 3): #draw/pick up
            legal_actions = set()
            meld_clusters = melding.get_meld_clusters(hand=hand)
            for meld_cluster in meld_clusters:
                meld_cards = [card for meld_pile in meld_cluster for card in meld_pile]
                hand_deadwood = [card for card in hand if card not in meld_cards]  # hand has 11 cards
                if len(hand_deadwood) <= 1:
                    legal_actions.add(5)
            discard_actions = [DiscardAction(card=card) for card in hand]
            discard_action_ids = [action_event.action_id for action_event in discard_actions]
            legal_actions.update(discard_action_ids)
            state['legal_actions'] = list(legal_actions)

            if a == 2:
                deck = [1 if card == 0 else 0 for card in s['obs'][0]]
                deck = [deck[i] - s['obs'][1][i] for i in range(52)]
                occurrence = np.random.randint(48)
                index = [i for i, card in enumerate(deck) if card == 1][occurrence]
                state['obs'][0][index] = 1
            else:
                index = np.where(s['obs'][1] == 1)
                state['obs'][0][index] = 1 #pickup from top of discard
                state['obs'][1][index] = 0 #wipe old discard
                deck = [1 if card == 0 else 0 for card in state['obs'][0]] #possible cards that could be in discard
                occurrence = np.random.randint(48)
                index = [i for i, card in enumerate(deck) if card == 1][occurrence]
                state['obs'][1][index] = 1
        elif a == 4:
            state['legal_actions'] = [0]
        elif player_id == 0 and a == 5: #don't care about discard pile
            state['legal_actions'] = [0]
            _, gin_cards = judge.get_going_out_cards(hand, 10)
            card_id = utils.get_card_id(gin_cards[0])
            state['obs'][0][card_id] = 0
            old_index = np.where(s['obs'][1] == 1)
            state['obs'][1][old_index] = 0
            state['obs'][1][card_id] = 1
        elif a in range(6, 58):
            if player_id == 0:
                state['legal_actions'] = [2, 3] #we dont know how many cards are left in the deck :(
                card_id = a - 6
                state['obs'][0][card_id] = 0
                old_index = np.where(s['obs'][1] == 1)
                state['obs'][1][old_index] = 0
                state['obs'][1][card_id] = 1
            else:
                card_id = a - 6
                old_index = np.where(s['obs'][1] == 1)
                state['obs'][1][old_index] = 0
                state['obs'][1][card_id] = 1
        elif player_id == 1:
            print("TR PLAYER 1 ACTION: ", str(a))
        else:
            raise GinRummyProgramError("TR KNOCKING. BAD.")
        return state, reward

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

    def get_next_player(self, a, player_id):
        if a == 0 or a == 1:
            next_player_id = 1
        elif a == 2 or a == 3:
            return player_id
        elif a == 4 or a == 5:
            return 0
        elif a in range(6, 58):
            return (player_id + 1) % 2
        else:
            raise GinRummyProgramError("GET NEXT PLAYER KNOCKING. BAD.")

    def rollout(self, MCTS, s, player_id, d, winner):
        if d <= 0:
            return 0

        print("ROLLOUT PLAYER ID: ", str(player_id))
        print("ROLLOUT STATE: ", s)

        if player_id == 1:
            deck = [1 if card == 0 else 0 for card in s['obs'][0]]
            deck = [deck[i] - s['obs'][1][i] for i in range(52)]
            index = np.where(s['obs'][1] == 1)
            s['obs'][1][index] = 0 #wipe old discard
            occurrence = np.random.randint(48)
            index = [i for i, card in enumerate(deck) if card == 1][occurrence]
            s['obs'][1][index] = 1
        else:
            a = novice.step(s) #TODO: IS OUR ROLLOUT POLICY JUST THE NOVICE POLICY?
        if a == 5:
            print("GIN!! WINNER = %s" % player_id)
            winner = player_id
        print("ROLLOUT ACTION: %s" % a)
        sp, r = MCTS.TR(s, a, player_id, winner) #TODO: HOW DO WE GENERATE THE NEXT STATE RANDOMLY FROM TR
        print("ROLLOUT NEXT STATE: ", sp)
        next_player_id = self.get_next_player(a, player_id)
        print("ROLLOUT NEXT PLAYER ID: %s" % next_player_id)
        if a == 1: #scored the last player, we out
            d = 1
        result = r + MCTS.gamma * self.rollout(MCTS, sp, next_player_id, d - 1, winner)
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

    def simulate(self, MCTS, s, player_id, d, winner): #TODO: SHOULD WE ROLLOUT UNTIL GAME OVER?
        if d <= 0:
            return 

        print("SIMULATE PLAYER ID: ", str(player_id))
        print("SIMULATE STATE: ", s)

        gamma, N, Q = MCTS.gamma, MCTS.N, MCTS.Q
        
        A = s['legal_actions']

        hashable_state = self.hashable_state(s)
        if not (hashable_state, A[0]) in N:
            for a in A:
                N[(hashable_state,a)] = 0
                Q[(hashable_state,a)] = 0
            return self.rollout(MCTS, s, player_id, d, None)

        a = self.explore(MCTS, s)
        if isinstance(a, GinAction):
            winner = player_id
        print("SIMULATE ACTION: ", str(a))
        sp, r = MCTS.TR(s, a, player_id, None) #TODO: HOW DO WE GENERATE THE NEXT STATE RANDOMLY FROM TR
        next_player_id = self.get_next_player(a, player_id)
        print("NEXT PLAYER_ID: %s" % next_player_id)
        q = r + gamma*self.simulate(MCTS, sp, next_player_id, d - 1, winner) #TODO: when do we make fresh copies??? 
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
        state, player_id = arg[0], arg[1]
        self.state = state
        MCTS = TS()
        for k in range(self.kmax):
            self.simulate(MCTS, state, player_id, 90, None)
        argmax = None
        maxQ = -100
        for a in state['legal_actions']:
            possible = MCTS.Q[(self.hashable_state(s), a)]
            if possible > maxQ:
                maxQ = possible
                argmax = a
        return argmax #return action that maximizes Q