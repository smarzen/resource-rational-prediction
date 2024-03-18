import numpy as np
import pylab as pl
import scipy as sp
import json
import csv
import seaborn as sns

## Based on the hidden Markov model for each process,
## these are the emission probabilities.
## The emission matrix has states as rows and
## symbols as columns.
def emissionProbs(input_type='NoisyPeriodic'):
	if input_type=='NoisyPeriodic':
		q = 0.1
		p = np.zeros([2,2])
		p[0,0] = 1
		p[1,0] = q
		p[1,1] = 1-q
		p = p*0.5
	elif input_type=='EvenProcess':
		q = 0.3
		p = np.zeros([2,2])
		p[0,0] = (1-q)/(1+q)
		p[0,1] = q/(1+q)
		p[1,1] = 1-(1/(1+q))
	else:
		q = 0.3
		p = np.zeros([3,2])
		p[0,0] = 1-q
		p[0,1] = q
		p[1,0] = 1
		p[2,1] = 1
		cl_psigma = np.array([10/3,1,1])
		cl_psigma = cl_psigma/(16/3)
		p[0,:] = p[0,:]*cl_psigma[0]
		p[1,:] = p[1,:]*cl_psigma[1]
		p[2,:] = p[2,:]*cl_psigma[2]
	return p

## The Blahut-Arimoto algorithm for finding the
## PRA curve, which involves choosing a Lagrange multiplier
## beta and calculating the rate and distortion at that beta
## after iterating an equation as shown in the main text.
def PRD(p,beta):
	# uses accuracy rather than pred power
	# d(x,xhat) = p(x=xhat|sigma)
	# calculate p(x|sigma)
	pX = np.sum(p,0)
	pS = np.sum(p,1)
	pXgS = np.dot(np.diag(1/pS),p)
	# The distortion matrix is given by the emission probabilities properly normalized.
	d = pXgS
	# Initialize the lossy compression at the identity.
	pXhatgS0 = np.random.uniform(size=len(pS))
	pXhatgS = np.vstack([pXhatgS0,1-pXhatgS0]).T
	pXhat = np.dot(pXhatgS.T,pS) #np.sum(np.dot(np.diag(pS),pXhatgS),0)
	# Run the Blahut Arimoto equation for 5000 timesteps to ensure convergence.
	for t in range(5000): # fix this
		log_pXhatgS = np.meshgrid(np.log(pXhat),np.ones(len(pS)))[0]+beta*d
		pXhatgS = np.exp(log_pXhatgS)
		Zs = np.sum(pXhatgS,1)
		pXhatgS = np.dot(np.diag(1/Zs),pXhatgS)
		pXhat = np.dot(pXhatgS.T,pS)
	# Calculate the rate and distortion, where the rate is the mutual information as shown in main text
	# and the distortion D is the expected distortion for that lossy compression.
	HXhat = -np.nansum(pXhat*np.log(pXhat))
	R = HXhat+np.dot(pS,np.nansum(pXhatgS*np.log(pXhatgS),1)) 
	D = np.dot(pS,np.sum(pXhatgS*pXgS,1))
	return R, D

# choose a range of betas (Lagrange multipliers) 
# to ensure that the entire curve is seen
betas = np.linspace(0,100,1000)
p = emissionProbs('EvenProcess') # can be chosen to be any process
Rs = []
Ds = []
for i in range(len(betas)): # run through all betas
	# run the Blahut-Arimoto algorithm to calculate the rate
	# and expected distortion at that Lagrange multiplier
	r, d = PRD(p,betas[i])
	Rs.append(r)
	Ds.append(d)