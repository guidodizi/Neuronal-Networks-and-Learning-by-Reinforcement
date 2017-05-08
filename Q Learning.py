from random import randint
import time
 

EPISODES = 1

class Qlearning:

	def __init__(self):
		self.gamma = 0.9 

		self.states = []
		for i in range(0,36):
			self.states.append(i)

		self.statesCount = 36
	 
		self.R = [[0 for x in range(self.statesCount)] for y in range(self.statesCount)]  # reward lookup
		self.Q = [[0 for x in range(self.statesCount)] for y in range(self.statesCount)]  # Q learning
		#self.Q = [[randint(-50,50) for x in range(self.statesCount)] for y in range(self.statesCount)]  # Q learning

		self.actions = []
		for i in range(0,36):
			actionsFrom = []
			if i % 5 != 0: # si no esta en la columna de mas a la derecha agrego el de su der
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
		
		self.epsilon = float(1) / EPISODES

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
			stack = []
			#Select random initial state
			state = randint(0,self.statesCount-1)		
			while state not in self.statesGoal:
			
				#Obtengo las acciones del estado actual
				actionsFromState = self.actions[state]	
				rand = float(randint(0,9))/ 10			
						
				if (rand > self.epsilon):
					#EXPLORACION: Selecciono una de las acciones posibles en forma aleatoria				
					index = randint(0,len(actionsFromState)-1)			
					action = actionsFromState[index]				
				else:
					#EXPLOTACION: Selecciono la mejor accion posible									
					action = self.max(state)

				nextState = action
				#Using this possible action, consider to go to the next state
				q = self.Q[state][action]
				maxQ = self.maxQ(nextState)
				r = self.R[state][action]


				#value = q + self.alpha * (r + self.gamma * maxQ - q)
				value = r + self.gamma*(maxQ)
				
				print "estoy en estado: " + str(state)
				print "accion: " + str(action)
				print "next state: " + str(nextState)
				print "q : " + str(q) + " maxQ: " + str(maxQ) + " r: " + str(r)
				print "value: " + str(value)
				print "----------"

				stack.append({'state':state,'value':value,'action':action})
				#self.Q[state][action] =  value
				#Set the next state as the current state
				state = nextState

			#recorro en orden inverso el stack y actualizo el Q		
			while len(stack) > 0:
				item = stack.pop()		
				self.Q[item['state']][item['action']] = item['value']
			self.epsilon += float(1) / EPISODES

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
			print "out from " + str(self.states[i]) + ": " + stateN
	
	# policy is maxQ(states)
	def showPolicy(self):
		print("\nshowPolicy")
		for i in range(0,len(self.states)):
			sfrom = self.states[i]
			sto =  self.policy(sfrom)
			print("from "+ str(self.states[sfrom]) + " goto "+ str(self.states[sto]))
                 


##MAIN
obj = Qlearning() 
obj.run()
obj.printResult()
obj.showPolicy()