# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey

from sys import argv

from collections import Counter

import pickle

import os

import sys, signal
def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class Learner(object):
    '''
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # init Q
        # -- dictionary class that handles zero initialization and indexing
        # -- for us so that we do not run into unforseen errors
        self.Q = Counter()

        # init discount rate
        self.discount = DISCOUNT
        # init learning rate
        self.eta = ETA
        # init greedy param
        self.epsilon = EPSILON

        # init gravity (at 1, choose 4 otherwise)
        self.gravity = 1
        # second chance for gravity
        # -- turn it on when we fail to get a good reading of gravity
        # -- on the first go
        self.bad_gravity = False

        # init epoch
        self.epoch = 0
        # new_game has started
        self.new_game = True


    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # reset game
        self.new_game = True
        self.gravity = 1
        self.bad_gravity = False
        self.epoch += 1


    # --------------------------BEGIN HELPER FUNCTIONS--------------------------------------
    # --------------------------------------------------------------------------------------
    # get discrete states
    def getDiscreteState(self, state, dsize=20):
        # store discrete state information
        discrete_state = {}

        # distance from the monkey to the next tree
        discrete_state['dist_tree'] = int(state['tree']['dist']/dsize)

        # distnace from top of monkey to top of tree gap
        discrete_state['mky_to_top'] = int((state['tree']['top'] - state['monkey']['top'])/dsize)

        # distance from bottom of monkey to bottom of tree gap
        discrete_state['mky_to_bot'] = int((state['monkey']['bot'] - state['tree']['bot'])/dsize)

        # gravity
        discrete_state['gravity'] = self.gravity

        return(discrete_state)

    # get Q value from dictionary
    def getQValue(self, state, action):
        Q_value = self.Q[(tuple(state.values()), action)]
        return(Q_value)

    # get action from state via epsilon-greedy policy
    def getAction(self, state):
        # choose random action with episilon probability
        if npr.rand() < self.epsilon:
            action = npr.choice([0, 1])
        # choose the max action
        else:
            # else choose optimal policy
            actions = [self.getQValue(state,i) for i in [0,1]]
            action = np.argmax(actions)

        return(action)

    # --------------------------------------------------------------------------------------
    # -----------------------------END HELPER FUNCTIONS-------------------------------------

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

    # -------------------------BEGIN ADDED ACTION CODE-------------------------------------
    # -------------------------------------------------------------------------------------
        # define gravity
        # -- need to have a past action to obtain past velocity
        # AND
        # -- need indicator of a new game to be on, telling us to find gravity
        # OR
        # -- we got a bad reading of gravity the first time
        # ^ to be fully implemented
        if (self.last_state is not None) and (self.new_game== True):

            # gravity = acceleration = change in velocity
            grav = int(self.last_state['monkey']['vel'] - state['monkey']['vel'])

            # check if its in set (1,4)
            # -- if not, define it to be one
            if grav not in set([1,4]):
                grav = 1
                self.bad_gravity = True

            # updates
            self.gravity = grav
            self.new_game = False


        # first action: do nothing
        # -- the fall will let us calculate gravity and immediately doing nothing should not
        # -- be a problem to recover from
        if self.last_state == None:
            current_action = 0
        # else proceed with Q Learning
        else:
            # grab discrete states
            last_state = self.getDiscreteState(self.last_state)
            current_state = self.getDiscreteState(state)

            # last Q value
            last_Qval = self.getQValue(last_state, self.last_action)

            # grab current action
            current_action = self.getAction(current_state)
            # current Q value
            current_Qval =  self.getQValue(current_state, current_action)

            # update Q
            gradient = last_Qval - (self.last_reward + (self.discount*current_Qval))
            update = last_Qval - (self.eta*gradient)
            self.Q[(tuple(last_state.values()), self.last_action)] = update

        # move forwward
        # -- place current, non-discrete into last state
        # -- place current action into last action
        self.last_state = state
        self.last_action = current_action

        # return action
        return(self.last_action)
    # -------------------------END ADDED ACTION CODE---------------------------------------
    # -------------------------------------------------------------------------------------

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

def run_games(file, learner, hist, iters = 100, t_len = 10):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    f = open(file+'.txt', 'w')
    f.write('Training History\n')

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # save modulo
        if learner.epoch <= 200:
            mod = 50
        else:
            mod = 10

        if learner.epoch%mod == 0:
            curr_best = '\nEpoch {} - Current Best Score: {}\n'.format(learner.epoch, np.max(hist))
            print(curr_best)
            f.write(curr_best)

            with open(file+'.pickle', 'wb') as outputfile:
                pickle.dump(learner.Q, outputfile)

            np.save(file,np.array(hist))

        if DECREASING_EPSILON:
            learner.epsilon *= 0.99

        # Reset the state of the learner.
        learner.reset()

    best_score = 'Best Score: {}'.format(np.max(hist))
    print(best_score)
    f.write(best_score+'\n')

    avg_score = 'Average Score: {}'.format(np.mean(hist))
    print(avg_score)
    f.write(avg_score+'\n')

    f.close()

    pg.quit()
    return


if __name__ == '__main__':

    # check command line arguments
    if len(argv[1:]) != 5:
        error_string = 'Check inputs.'
        error_string += ' ' + 'First argument should be the discount rate.'
        error_string += ' ' + 'Second argument should be the learning rate.'
        error_string += ' ' + 'Third argument should be the explorative parameter.'
        error_string += ' ' + 'Fourth argument should be the mumber of epochs to run.'
        error_string += ' ' + 'Fifth argument should be the adaptive explorative parameter.'

        exit(error_string)

    try:
        DISCOUNT = float(argv[1])
        ETA = float(argv[2])
        EPSILON = float(argv[3])
        EPOCHS = int(argv[4])
        DECREASING_EPSILON = bool(int(argv[5]))
    except ValueError:
        exit('Please provide numerical command line arguments.')

    file = 'SARSA/'

    if not os.path.exists('SARSA'):
        os.makedirs('SARSA')

    file += 'Discount'+'_'+str(DISCOUNT)+'_'
    file += 'Eta'+'_'+str(ETA)+'_'
    file += 'Epsilon'+'_'+str(EPSILON)

    if DECREASING_EPSILON:
        file += '_DecreasingEpsilon'

    # Select agent
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(file, agent, hist, EPOCHS, 1)

    # Save history.
    np.save(file,np.array(hist))
