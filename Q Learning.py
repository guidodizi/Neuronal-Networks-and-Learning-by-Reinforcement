from random import randint
import time
 

EPISODES = 5

class Qlearning:

	def __init__(self):
		self.alpha = 0.1
		self.gamma = 0.9 
 
# states A,B,C,D,E,F

		self.stateA = 0
		self.stateB = 1
		self.stateC = 2
		self.stateD = 3
		self.stateE = 4
		self.stateF = 5
		
		self.statesCount = 6
		self.states =  [self.stateA,self.stateB,self.stateC,self.stateD,self.stateE,self.stateF] 
	 
		# Q(s,a)= Q(s,a) + alpha * (R(s,a) + gamma * Max(next state, all actions) - Q(s,a))
	 
		self.R = [[0 for x in range(self.statesCount)] for y in range(self.statesCount)]  # reward lookup
		self.Q = [[0 for x in range(self.statesCount)] for y in range(self.statesCount)]  # Q learning
	 
		self.actionsFromA = [self.stateB, self.stateD]
		self.actionsFromB = [self.stateA, self.stateC, self.stateE]
		self.actionsFromC = [self.stateC]
		self.actionsFromD = [self.stateA, self.stateE]
		self.actionsFromE = [self.stateB,self.stateD, self.stateF]
		self.actionsFromF = [self.stateC, self.stateE]
		self.actions = [self.actionsFromA, self.actionsFromB, self.actionsFromC,
				self.actionsFromD, self.actionsFromE, self.actionsFromF]
	 
		self.stateNames = ["A", "B", "C", "D", "E", "F"]
		self.R[self.stateB][self.stateC] = 100 # from b to c
		self.R[self.stateF][self.stateC] = 100

	def run(self): 
	# 1. Set parameter , and environment reward matrix R 
	# 2. Initialize matrix Q as zero matrix 
	# 3. For each episode: Select random initial state 
	#	Do while not reach goal state o 
	#		Select one among all possible actions for the current state o 
	#		Using this possible action, consider to go to the next state o 
	#		Get maximum Q value of this next state based on all possible actions o 
	#		Compute o Set the next state as the current state
		for i in range(0,EPISODES):
			#Select random initial state
			state = randint(0,self.statesCount-1)		
			while state != self.stateC:
			
				#Obtengo las acciones del estado actual
				actionsFromState = self.actions[state]
					
				#Selecciono una de las acciones posibles en forma aleatoria				
				index = randint(0,len(actionsFromState)-1)			
				action = actionsFromState[index]
				
				nextState = action
				
				#Using this possible action, consider to go to the next state
				q = self.Q[state][action]
				maxQ = self.maxQ(nextState)
				r = self.R[state][action]

				print "estoy en estado: " + str(state)
				print "accion: " + str(action)
				print "next state: " + str(nextState)
				print "q : " + str(q) + " maxQ: " + str(maxQ) + " r: " + str(r)
				#value = q + self.alpha * (r + self.gamma * maxQ - q)
				value = r + self.gamma*
				print "value: " + str(value)
				self.Q[state][action] =  value
	 
				#Set the next state as the current state
				state = nextState
					
	def maxQ (self,s):
		actionsFromState = self.actions[s]
		maxValue = -100000
		for i in range(0,len(actionsFromState)):
			nextState = actionsFromState[i]
			value = self.Q[s][nextState]
			
			if value > maxValue:
				maxValue = value
		return maxValue
		
	def policy (self,state):
		actionsFromState = self.actions[state]
		maxValue = -10000
		policiGoToState = state
		for i in range (0,len(actionsFromState)):
			nextState = actionsFromState[i]
			value = self.Q[state][nextState]
			
			if (value > maxValue):
				maxValue = value
				policiGoToState = nextState
		return policiGoToState
		
	def printResult(self):
		print "Print result"
		for i in range(0,len(self.Q)):
			stateN = ""
			for j in range(0,len(self.Q[i])):
				stateN += str(self.Q[i][j]) + " "
			print "out from " + self.stateNames[i] + ": " + stateN
	
	# policy is maxQ(states)
	def showPolicy(self):
		print("\nshowPolicy")
		for i in range(0,len(self.states)):
			sfrom = self.states[i]
			sto =  self.policy(sfrom)
			print("from "+ self.stateNames[sfrom]+" goto "+ self.stateNames[sto])
                 


##MAIN
BEGIN = int(round(time.time() * 1000)) 
obj = Qlearning() 
obj.run()
obj.printResult()
obj.showPolicy()

END = int(round(time.time() * 1000))
print("Time: " + str((END - BEGIN) / 1000.0) + " sec.")