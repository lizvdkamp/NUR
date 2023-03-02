import numpy as np
import matplotlib.pyplot as plt
import timeit

#Question 1

def Poisson(l, k):
	"""A function that returns the Poisson probability distribution for integer k and positive mean l."""
	#Converting the inputs to 32 bit to ensure low memory usage
	k = k.astype(np.int32)
	l = l.astype(np.float32)
	P = np.zeros(len(k))
	
	for i in range(len(k)):
	#Looping over the values so that we can use np.math.factorial
	#But first we want to make sure that the factorial will not overflow
		ki = k[i]
		li = l[i]
		kisum = 0
		if ki > 16:
		#If k is large, we want to go to log space to prevent overflow
		#Using ln(k!) = sum(ln(j)) with j from 1 to k
			for j in range(1,ki+1):
				kisum += np.log(j)
			#Calculating the ln of the Poisson distribution
			lnP = np.float32((ki*np.log(li)-li)-(kisum))
			P[i] = np.float32(np.exp(lnP))
		else:
			P[i] = np.float32((li**ki * np.exp(-li))/np.math.factorial(ki))
	
	return P

	
ks = np.array([0,10,21,40,200],dtype=np.int32)
ls = np.array([1,5,3,2.6,101],dtype=np.float32)

Poisson = Poisson(ls, ks)

# Save a text file
np.savetxt('Poissonoutput.txt',np.transpose([ls,ks,Poisson]))


