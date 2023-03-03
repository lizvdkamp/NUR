import numpy as np
import matplotlib.pyplot as plt
import timeit
import sys
import os

#Question 2

#I have created these functions in collaboration with my sister Evelyn van der Kamp (s2138085) during the tutorial classes

#Functions to swap, scale and add rows to each other
def SwapRow(M,i,j):
	"""Takes a matrix M and swaps rows i and j"""
	#Making a copy to make sure that it doesn't overwrite M
	B = np.copy(M).astype('float64')
	#Swaps row i and j
	save = B[i,:].copy()
	B[i,:]= B[j,:]
	B[j,:] = save

	return B

def SwapRowVec(x,i,j):
	"""Takes a vector/array x and swaps rows i and j"""
	#Making a copy to make sure that it doesn't overwrite x
	B = np.copy(x).astype('float64')
	#Swaps row i and j of the vector
	save = B[i].copy()
	B[i]= B[j]
	B[j] = save

	return B

def ScaleRow(M,i,scale):
	"""Takes a matrix M and scales row i with scale = scale"""
	#Making a copy to make sure that it doesn't overwrite M
	B = np.copy(M).astype('float64')
	#Scales row i
	B[i,:] *= scale

	return B

def AddRow(M,i,j,scale):
	"""Takes a matrix M and adds row i to row j scale times"""
	#Making a copy to make sure that it doesn't overwrite M
	B = np.copy(M).astype('float64')
	#Adds row i to row j scale times
	B[j,:] += B[i,:]*scale

	return B

#A function that uses Crouts algorithm to do LU decomposition
def Crout(A,b):
	"""Takes a Matrix A and vector b and applies Crouts algorithm to do LU decomposition on the Matrix A and using the L and U matrices to calculate a solution to the equation Ax = b."""
	rows = A.shape[0]
	columns = A.shape[1]

	#saving the parity just in case
	parity = 1

	#Making a copy of the matrix and the vector to make sure that it doesn't overwrite them
	LU = np.copy(A).astype('float64')
	b_new = np.copy(b).astype('float64')

	#Making an index array to keep track of swapped indices
	indx = np.arange(0,rows,1, dtype=int)
	#print(indx)

	#First I check if the matrix A is singular by looking if there is an all-zero column
	for j in range(columns):
		if np.all(LU[:,j]==0):
			print("The matrix is singular. Stopping.")
			#Returning a zero solution
			return np.zeros(rows)
        
	#Putting the highest values in a given column on a diagonal by looking through all values below the diagonal and checking if there is a higher value before swapping
	for k in range(columns):
		#looping over columns and saving the diagonal
		mx = np.abs(LU[k,k])
		piv = k
		#print(mx)
        
		for i in range(k+1,rows):
			#checking if the rows below have higher values to put on the diagonal
			xik = np.abs(LU[i,k])
			#print(xik)
            
			if xik > mx:
				mx = xik
				piv = i
				#print(mx)
                
		#If there is a higher value in the column below the diagonal we swap the relevant rows
		if piv != k:
			#swapping row & swapping index array
			#print("Swap", k, piv)
			LU = SwapRow(LU,k,piv)
			indx = SwapRowVec(indx,k,piv).astype(int)
			parity = -parity
			#print(LU, b_new)
            
		#getting the LU matrix
		diag = LU[k,k]
		for i in range(k+1,rows):
			LUik = LU[i,k] / diag
			LU[i,k] = LUik
			for j in range(k+1,rows):
				LU[i,j] -= LUik * LU[k,j]
	#print(LU)

	#Getting the solution
	x_sol = np.zeros(rows)
	y_sol = np.zeros(rows)
    
	#print(indx)
	#Solving the equation Ux = y with forward substitution
	for n in range(rows):
		ay = 0
		for m in range(0,n):
			ay += LU[n,m]*y_sol[m]
		y_sol[n] = b_new[indx[n]]-ay
		#print(b_new[n],y_sol)
    
	#Solving Ly = b with backsubstitution
	for n in range(rows):
		#Making sure that we loop from N-1 --> 0
		backsub = rows-(n+1)
		ax = 0
		for m in range(backsub,rows):
			ax += LU[backsub,m]*x_sol[m]
		x_sol[backsub] = 1/LU[backsub,backsub] * (y_sol[backsub]-ax)
		#print(x_sol)

	return LU, x_sol, indx

#Obtaining the data
data=np.genfromtxt(os.path.join(sys.path[0],"Vandermonde.txt"),comments='#',dtype=np.float64)
x=data[:,0]
y=data[:,1]
xx=np.linspace(x[0],x[-1],1001) #x values to interpolate at

length = len(x)

#Creating the Vandermonde matrix
VMm = np.zeros((length,length))
for j in range(length):
	VMm[:,j] = x**j

#LU decomposition
LUmat, c_sol, indxs = Crout(VMm, y)

#Printing the solution / Saving the solution
np.savetxt('LU1output.txt',np.transpose([c_sol]))

#Interpolated polynomial solution at 1000 points and only at the points xi
yy = np.zeros(len(xx))
yi = np.zeros(length)
for j in range(length):
	yy += c_sol[j]*xx**j
	yi += c_sol[j]*x**j

#Absolute difference between solution and data
ydif = np.abs(yi-y)

#Plotting
fig,(ax1,ax2)=plt.subplots(2, sharex=True)
ax1.plot(x,y,marker='o',linewidth=0, color='k', label='data')
ax1.plot(xx,yy, label='polynomial')
ax1.set_xlim(-1,101)
ax1.set_ylim(-400,400)
ax1.set_ylabel('$y$')
ax1.legend()
#Absolute difference
ax2.plot(x,ydif, marker='o', linewidth=0, label='absolute difference')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$|y(x)-y_i|$')
ax2.legend()
plt.savefig('LU1plot.png')
plt.close()

#2b

#Nevilles algorithm function
def polyinterp(x, y, x_i):
	"""A function which takes data points x & y and uses Nevilles algorithms to interpolate over points x_i, returns the interpolated y_i which is the same length as x_i"""
	#Bisection
	size = len(x)
	y_i = np.zeros(len(x_i))
	M = size
	for i in range(len(x_i)):
		#Initializing a starting index and stopping index
		start = 0
		stop = int(size/2)
		x_j = x_i[i]
		m = 2	#a variable to store powers of 2
		while start != stop:
			if x_j < x[stop]:
			#If our x is smaller than the data point at index stop, we want to keep start the same index and half our range of searching, aka stop=stop-(range)/2 
				#print("yes", x[stop], m)
				start = start
				stop = int(stop-(stop-start)/2)
				m*=2 #multiply m by two, because we've halved our range
				#print(start, stop)
			else:
			#If our x is larger than the data point at index stop, we want to take the upper half of the array and check where x is in that range, so we set start = stop, and stop = stop+range. We do not want to halve our range yet.
				#print("no", x[stop], m)
				start = stop
				stop = stop+int(size/m)
				#print(start, stop)

			if stop > size-1:
			#If stop gets larger than the size of the array, we want to stop and set our start at size-2 (aka second to last point) and stop the same so that the while loop stops
				stop = size-2
				start = size-2
		#Now we have stop and start such that x_i is between x[start] and x[stop+1]
                
		#print(start,stop)

		#Now depending on the order of the polynomial, we set start and stop such that x[start] until x[stop] are the M points around x_i
		start = start-int((M-1)/2)
		stop = stop+(1-M%2)+int((M-1)/2)
        
		if stop > size-1:
		#If stop is too large we take the M points at the end
			start = size-M
			stop = size-1
		if start <= 0:
		#If start is below 0 we take the M points at the start
			start = 0
			stop = M-1
            
		#print(start, stop)

		#Interpolating with Nevilles algorithm
		y_p = y.copy()
		for k in range(1, M):
			for j in range(M-k):
				#print(k, j, x_j)
				y_p[j] = ((x[j+k] - x_j)*y_p[j] + (x_j - x[j])*y_p[j+1])/(x[j+k]-x[j])
				#print(y_p)
        
		y_i[i] = y_p[0]
        
		#print(x_j, x[start],x[stop+1], y_i[i], y[start], y[stop+1])    
        
	return y_i

#Creating the interpolated arrays, one for the full sample and one for the absolute difference
y_poly = polyinterp(x,y,xx)
y_i = polyinterp(x,y,x)

ydif2 = np.abs(y_i-y)


#Plotting
fig,(ax1,ax2)=plt.subplots(2, sharex=True)
ax1.plot(x,y,marker='o',linewidth=0, color='k', label='data')
ax1.plot(xx,yy, label='polynomial (LU)')
ax1.plot(xx,y_poly, label='polynomial (Neville)')
ax1.set_xlim(-1,101)
ax1.set_ylim(-400,400)
ax1.set_ylabel('$y$')
ax1.legend()
#Absolute difference
ax2.plot(x,ydif2, marker='o', linewidth=0, label='absolute difference Neville polynom')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$|y(x)-y_i|$')
ax2.legend()
plt.savefig('Nevilleplot.png')
plt.close()

#There is a difference between the absolute differences, from 2(a) it was order 0.1 and now it is order 10^-14, the polynomials are different, especially at the start.
#In the solution from LU decomposition the error is higher because in Neville's algorithm we already reduce the error by iterating the different solutions

#2c

#A function that uses Crouts algorithm to do LU decomposition
def LUiteration(A,b,LU,xsol,indx,its):
	"""Takes the LU decomposition of matrix A, the solution xsol for which A*xsol = b, and the indx matrix which has information about which rows of A have been switched, and calculates an improved solution after iterating over the solution its times."""

	rows = A.shape[0]
	columns = A.shape[1]
	#zeroth iteration
	xiter = xsol.copy()
	
	for i in range(its):
		#Calculating delta b with db = A*xsol - b
		delta_b = np.sum(A*xiter, axis=1) - b
		#Now calculating the solution to A*delta_x = delta_b with the LU matrix in the same way as was done in the Crout function
		y_sol = np.zeros(len(xsol))
		x_sol = np.zeros(len(xsol))
		#Solving the equation Ux = y with forward substitution
		for n in range(rows):
			ay = 0
			for m in range(0,n):
				ay += LU[n,m]*y_sol[m]
			y_sol[n] = delta_b[indx[n]]-ay
	    
		#Solving Ly = b with backsubstitution
		for n in range(rows):
			#Making sure that we loop from N-1 --> 0
			backsub = rows-(n+1)
			ax = 0
			for m in range(backsub,rows):
				ax += LU[backsub,m]*x_sol[m]
			x_sol[backsub] = 1/LU[backsub,backsub] * (y_sol[backsub]-ax)
		
		#Now we set the next iteration of x, x((i+1)'th iteration) = x(i) - delta_x(i)
		xiter -= x_sol

	return xiter


#Our new solution after 10 iterations
c_sol10 = LUiteration(VMm,y,LUmat,c_sol,indxs,10)

#Printing the solution
print(c_sol10)

#Interpolated polynomial solution at 1000 points and only at the points xi
yy10 = np.zeros(len(xx))
yi10 = np.zeros(len(x))
for j in range(len(x)):
	yy10 += c_sol10[j]*xx**j
	yi10 += c_sol10[j]*x**j

#Absolute difference between solution and data
ydif10 = np.abs(yi10-y)

#Plotting
fig,(ax1,ax2)=plt.subplots(2, sharex=True)
ax1.plot(x,y,marker='o',linewidth=0, color='k', label='data')
ax1.plot(xx,yy, label='polynomial (LU1)')
ax1.plot(xx,yy10, label='polynomial (LU10)')
ax1.set_xlim(-1,101)
ax1.set_ylim(-400,400)
ax1.set_ylabel('$y$')
ax1.legend()
#Absolute difference
ax2.plot(x,ydif10, marker='o', linewidth=0, label='absolute difference LU10')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$|y(x)-y_i|$')
ax2.legend()
plt.savefig('LU10plot.png')
plt.close()

#The solution after 10 iterations is closer to the Neville algorithm solution, and the error is a lot lower except for the last point


#Question 2d

#Defining a variable for the amount of times to run it
times = 50

#2a
start2a = timeit.default_timer()

# run the LU decomposition function x times
for i in range(times):
	Crout(VMm, y)

att2a = (timeit.default_timer()-start2a)/times
print("Average time taken 2a:", att2a, "s")

#2b
start2b = timeit.default_timer()

# run Nevilles algorithm function x times
for i in range(times):
	polyinterp(x,y,xx)

att2b = (timeit.default_timer()-start2b)/times
print("Average time taken 2b:", att2b, "s")

#2c
start2c = timeit.default_timer()

# run the LU iteration function x times
for i in range(times):
	LUiteration(VMm,y,LUmat,c_sol,indxs,10)

att2c = (timeit.default_timer()-start2c)/times
print("Average time taken 2c:", att2c, "s")

#LU decomposition & iteration is about a 100 times faster than Nevilles algorithm, because Nevilles algorithm already iterates over the solution to get a more accurate result. Doing the LU decomposition and iterating over the solution is faster because you do not have to recalculate the LU matrix, and for Nevilles algorithm we use bisection for every single point we want to interpolate, which adds to the time if we want to interpolate many points with a big data set and create a high order polynomial.

# Save a text file
np.savetxt('Timesoutput.txt',np.transpose([att2a,att2b,att2c]))


