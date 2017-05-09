from random import randint
import time
 

EPISODES = 30

class Qlearning:

	def __init__(self,part):
		
		if part == 0:
			self.gamma = 0.9 
		elif part == 1:
			self.gamma = 0.4
		
		self.states = []
		for i in range(0,36):
			self.states.append(i)

		self.statesCount = 36
	 
		self.R = [[0 for x in range(self.statesCount)] for y in range(self.statesCount)]  # reward lookup
		if part == 0:
			self.Q = [[0 for x in range(self.statesCount)] for y in range(self.statesCount)]  # Q learning
		elif part == 1:
			self.Q = [[randint(-50,50) for x in range(self.statesCount)] for y in range(self.statesCount)]  # Q learning

		self.actions = []
		for i in range(0,36):
			actionsFrom = []
			if i % 6 != 5: # si no esta en la columna de mas a la derecha agrego el de su der
				actionsFrom.append(i+1) 
			if i % 6 != 0: # si no esta en la columna de mas a la izq agrego el de su izq
				actionsFrom.append(i-1) 
			if i > 5: # si no esta en la fila de mas arriba agrego el de arriba
				actionsFrom.append(i-6) 
			if i < 30: # si no esta en la fila de mas abajo agrego el de abajo
				actionsFrom.append(i+6)	

			self.actions.append(actionsFrom)


		self.R[6][7] = -20
		self.R[7][13] = 60
		self.R[8][14] = 60
		self.R[9][8] = -20
		self.R[25][19] = 20
		self.R[26][20] = 20

		self.statesGoal = [13,14,19,20]
		
		self.epsilon = float(1) / (EPISODES + (EPISODES*.4))

	def run(self): 
		for i in range(0,EPISODES):
			#print "EPISODE: " + str(i)
			stack = []

			#Select random initial state different to the goal state
			while True:
				state = randint(0,self.statesCount-1)	
				keepSearching = state in self.statesGoal
				if keepSearching:
					continue 			
				else:
					break
			
			while not keepSearching:		
				#Obtengo las acciones del estado actual
				actionsFromState = self.actions[state]	

				#For each state decide if its exploration or explotation
				rand = float(randint(0,99))/ 100			
				if (rand > self.epsilon):
					#EXPLORACION: Selecciono una de las acciones posibles en forma aleatoria				
					index = randint(0,len(actionsFromState)-1)			
					action = actionsFromState[index]
				else:
					#EXPLOTACION: Selecciono la mejor accion posible									
					action = self.max(state)
				nextState = action

				stack.append({'state':state,'nextState':nextState,'action':action})				
				
				#Set the next state as the current state
				state = nextState
				keepSearching = state in self.statesGoal
				
			#recorro en orden inverso el stack y actualizo el Q		
			while len(stack) > 0:
				item = stack.pop()

				q = self.Q[item['state']][item['action']]
				maxQ = self.maxQ(item['nextState'])
				r = self.R[item['state']][item['action']]
				
				value = r + self.gamma*(maxQ)
				if q < value:
					self.Q[item['state']][item['action']] = value
				
			self.epsilon += float(1) / (EPISODES + (EPISODES*.4))							

	def max (self,s):
		actionsFromState = self.actions[s]
		maxValue = -100000
		maxStates = []
		for i in range(0,len(actionsFromState)):
			nextState = actionsFromState[i]
			value = self.Q[s][nextState]			
			if value > maxValue:
				maxStates = []
				maxValue = value
				maxStates.append(nextState)
			elif value == maxValue:
				maxStates.append(nextState)

		if len(maxStates) == 1:
			return maxStates[0]
		else:
			rand = randint(0,len(maxStates)-1)
			return maxStates[rand]

	def maxQ (self,s):
		actionsFromState = self.actions[s]
		maxValue = -100000
		for i in range(0,len(actionsFromState)):
			nextState = actionsFromState[i]
			value = self.Q[s][nextState]
			
			if value > maxValue:
				maxValue = value
		return int(maxValue)
		
	def policy (self,state):
		actionsFromState = self.actions[state]
		maxValue = -10000
		policyGoToState = state
		for i in range (0,len(actionsFromState)):
			nextState = actionsFromState[i]
			value = self.Q[state][nextState]			
			if (value > maxValue):
				maxValue = value
				policyGoToState = nextState
		return policyGoToState
		
	def printResult(self):
		print "Print result"
		for i in range(0,len(self.Q)):
			stateN = ""
			for j in range(0,len(self.Q[i])):
				stateN += str(self.Q[i][j]) + " "
			print "out from " + str(self.states[i]) + ": " + stateN
	
	def showPolicy(self):
		print("\nshowPolicy")
		for i in range(0,len(self.states)):
			sfrom = self.states[i]
			sto =  self.policy(sfrom)
			print("from "+ str(self.states[sfrom]) + " goto "+ str(self.states[sto]) + " value " + str(self.Q[sfrom][sto]))
                
##MAIN

while True:
	part = input("Choose '0' for part B or '1' for part C:  ")
	if part == 0 or part == 1:
		break
	else:
		print "\nSorry, invalid input\n"
		continue

qLearner = Qlearning(part) 
qLearner.run()
qLearner.printResult()
qLearner.showPolicy()