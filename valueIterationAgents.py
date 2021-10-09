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

    def runValueIteration(self): #problem1 

        for i in range(self.iterations): 
            updated = self.values.copy() 

            for state in self.mdp.getStates(): 

                if self.mdp.isTerminal(state): 
                    continue

                actions = self.mdp.getPossibleActions(state)
                optimal = max([self.getQValue(state, action) for action in actions])
                updated[state] = optimal

            self.values = updated

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action): #problem1
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        val = 0 

        for s_prime, T in self.mdp.getTransitionStatesAndProbs(state, action): 

            val += T * ( self.mdp.getReward(state, action, s_prime) + self.discount * self.getValue(s_prime) )

        return val 

        util.raiseNotDefined()

    def computeActionFromValues(self, state): #problem1
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        policy = util.Counter()

        for action in self.mdp.getPossibleActions(state): 
            policy[action] = self.getQValue(state, action)

        return policy.argMax() 
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
        states =  self.mdp.getStates()
        numstates = len(states)

        for i in range(self.iterations):  
            state = states[i% len(states)]
            if self.mdp.isTerminal(state): 
                continue
            actions = self.mdp.getPossibleActions(state)
            optimal = max([self.getQValue(state, action) for action in actions])
            self.values[state] = optimal

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
        "*** YOUR CODE HERE *** "
        """
        compute predecessors of all states.
        Initialize an empty priority queue.
        For each non-terminal state s, do:
Find the absolute value of the difference between self.values[state] and max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]) (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
For iteration in 0, 1, 2, ..., self.iterations - 1, do:
If the priority queue is empty, then terminate.
Pop a state s off the priority queue.
Update the value of s (if it is not a terminal state) in self.values.
For each predecessor p of s, do:
Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p (this represents what the value should be); call this number diff. Do NOT update self.values[p] in this step.
If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as long as it does not already exist in the priority queue with equal or lower priority. As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
A couple of important notes on implementation:

When you compute predecessors of a state, make sure to store them in a set, not a list, to avoid duplicates.
Please use util.PriorityQueue in your implementation. The update method in this class will likely be useful; look at its documentation.
        """
        predecessors = util.Counter()
        for parent in self.mdp.getStates():
            children = [self.mdp.getTransitionStatesAndProbs(parent, action)[0] for action in self.mdp.getPossibleActions(parent)]
            for child in children:
                if predecessors[child[0]] == 0:
                    predecessors[child[0]] = {parent}
                else:
                    predecessors[child[0]].add(parent)
        pqueue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state): 
                continue
            diff = abs(self.getValue(state) -  max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]))
            pqueue.push(state, -diff)
        for _ in range(self.iterations):
            if pqueue.isEmpty():
                break
            s = pqueue.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
            for p in predecessors[s]:
                diff = abs(self.getValue(p) -  max([self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)]))
                if diff > self.theta:
                    pqueue.update(p, -diff)

