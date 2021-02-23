# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
from util import PriorityQueue
import math

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        mdp = self.mdp
        print("ITERATIONs: ", self.iterations)
        for i in range(0, self.iterations, 1):
            #print(i)
            counter = util.Counter()

            for states in mdp.getStates():

                if mdp.isTerminal(states):
                    counter[states] = 0

                elif not mdp.isTerminal(states):

                    max_val = -math.inf

                    for action in mdp.getPossibleActions(states):
                        agg = 0
                        for next, probability in mdp.getTransitionStatesAndProbs(states, action):
                            sum = mdp.getReward(states, action, next) + (self.values[next]*self.discount)
                            prob = probability*sum
                            agg += prob

                        max_val = max(max_val, agg)
                        #print("max: ", max_val)
                        counter[states] = max_val

            self.values = counter


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        sum = 0
        mdp = self.mdp
        for nxt, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            Q = probability * (mdp.getReward(state, action, nxt) + self.getValue(nxt) * self.discount)
            #print(Q)
            sum += Q

        #print(sum)
        return sum
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
            return None

        policy = None
        value = -1*math.inf
        mdp = self.mdp
        tuple_vector = []

        for actions in mdp.getPossibleActions(state):
            tempQ = self.computeQValueFromValues(state, actions)
            tuple_vector.append((actions, tempQ))

        res = max(tuple_vector, key = lambda i : i[1])[0]
        res2 = max(tuple_vector, key = lambda i : i[1])[1]



        return res

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        num_of_states = len(self.mdp.getStates())
        states = self.mdp.getStates()

        for i in range(0, self.iterations):
            #counter = util.Counter()
            #indx = (i % num_of_states)
            state = states[i % num_of_states]

            if self.mdp.isTerminal(state):
                self.values[state] = 0

            elif not self.mdp.isTerminal(state):
                max_val = -math.inf
                #values = []
                for action in self.mdp.getPossibleActions(state):
                    #V = probability * (mdp.getReward(curr_state, action, next_state) + (self.values[next_state] * self.discount))
                    q = self.computeQValueFromValues(state, action)
                    max_val = max(max_val, q)
                self.values[state] = max_val

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):

                for action in self.mdp.getPossibleActions(state):
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            if next_state in predecessors:
                                predecessors[next_state].add(state)
                            else:
                                predecessors[next_state] = {state}


        pq = util.PriorityQueue()
        for state in self.mdp.getStates():

            if not self.mdp.isTerminal(state):
                max_v = -math.inf

                for actions in self.mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, actions)
                    max_v = max(max_v, q)

                diff = abs(self.values[state] - max_v)
                pq.push(state, -diff)

        iter = self.iterations
        for i in range(0, iter):

            if pq.isEmpty():
                break

            s = pq.pop()

            if not self.mdp.isTerminal(s):
                max_v = -math.inf

                for actions in self.mdp.getPossibleActions(s):
                    q = self.computeQValueFromValues(s, actions)
                    max_v = max(max_v, q)

                self.values[s] = max_v

            for p in predecessors[s]:

                if not self.mdp.isTerminal(p):
                    max_v = -math.inf

                    for actions in self.mdp.getPossibleActions(p):
                        q = self.computeQValueFromValues(p, actions)
                        max_v = max(max_v, q)

                    diff = abs(self.values[p] - max_v)
                    if diff > self.theta:
                        pq.update(p, -diff)


