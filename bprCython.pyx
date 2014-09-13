#cython: profile=True 
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import random 
import logging 
import cython
from cython.parallel import parallel, prange
cimport numpy
import numpy
from math import exp, log 
import random
from sandbox.util.CythonUtils cimport inverseChoice
from sandbox.util.MCEvaluator import MCEvaluator

class BPRArgs(object):

    def __init__(self,learning_rate=0.05,
                 bias_regularization=1.0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.0025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors

cdef class BPR(object):
    cdef public unsigned int D, recordStep
    cdef public double bias_regularization, user_regularization, learning_rate
    cdef public double positive_item_regularization, negative_item_regularization
    cdef public bint update_negative_item_factors  
    cdef public unsigned int numAucSamples 

    def __init__(self, D, args):
        """initialise BPR matrix factorization model
        D: number of factors
        """
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors
        
        self.recordStep = 10 
        self.numAucSamples = 5

    def train(self, data, sampler, unsigned int num_iters):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        """
        cdef unsigned int u, i, j, it    
        cdef numpy.ndarray[double, ndim=2, mode="c"] user_factors        
        cdef numpy.ndarray[double, ndim=2, mode="c"] item_factors 
        
        sampler.numAucSamples = 1
        user_factors, item_factors, loss_samples  = self.init(data, sampler)
        logging.debug('initial loss = {0}'.format(self.loss(loss_samples, user_factors, item_factors)))
        logging.debug("Number of loss samples: " + str(len(loss_samples)))      
        
        sampler.numAucSamples = self.numAucSamples
        #train_samples = [t for t in sampler.generate_samples(data)]
        
        for it in xrange(num_iters):

            for u, i, j in sampler.generate_samples(data):
                self.update_factors(user_factors, item_factors, u, i, j)

            if it % self.recordStep == 0:
                logging.debug( 'iteration {0}: loss = {1}'.format(it, self.loss(loss_samples, user_factors, item_factors)))
                #logging.debug("AUC:" + str(MCEvaluator.averageAuc(data, user_factors, item_factors)))
                loss_samples = [t for t in sampler.generate_samples(data)]
            
        return user_factors, item_factors 

    def init(self, data, sampler):
        num_users, num_items = data.shape

        user_factors = numpy.random.random_sample((num_users, self.D))
        item_factors = numpy.random.random_sample((num_items, self.D))

        loss_samples = [t for t in sampler.generate_samples(data)]
        
        return user_factors, item_factors, loss_samples 

    def update_factors(self, numpy.ndarray[double, ndim=2, mode="c"] user_factors, numpy.ndarray[double, ndim=2, mode="c"] item_factors,  unsigned int u, unsigned int i, unsigned int j):
        """apply SGD update"""
        cdef bint update_j = self.update_negative_item_factors
        cdef double x, z, d, expx
        cdef numpy.ndarray[double, ndim=1, mode="c"] du 
        cdef numpy.ndarray[double, ndim=1, mode="c"] di
        cdef numpy.ndarray[double, ndim=1, mode="c"] dj
        cdef numpy.ndarray[double, ndim=1, mode="c"] item_factors_i
        cdef numpy.ndarray[double, ndim=1, mode="c"] item_factors_j
        cdef numpy.ndarray[double, ndim=1, mode="c"] user_factors_u

        item_factors_i = item_factors[i]
        item_factors_j = item_factors[j]
        user_factors_u = user_factors[u]

        x = numpy.dot(user_factors_u, item_factors_i-item_factors_j)

        #Fix for numerical overflow issues 
        try: 
            expx = exp(-x)
            z = expx/(1.0+expx)
        except OverflowError: 
            if x > 0: 
                z = 0 
            elif x < 0: 
                z = 1
                
        #Note all the regularisers are negative according to objective. Also we are adding the gradient 
        #since we wish to maximise the objective
        du = (item_factors_i-item_factors_j)*z - self.user_regularization*user_factors_u
        user_factors[u, :] += self.learning_rate*du
        
        user_factors_u = user_factors[u, :]
        
        di = user_factors_u*z - self.positive_item_regularization*item_factors_i
        item_factors[i,:] += self.learning_rate*di

        dj = -user_factors_u*z - self.negative_item_regularization*item_factors_j
        item_factors[j,:] += self.learning_rate*dj
            
        return du, di, dj 

    def loss(self, loss_samples, numpy.ndarray[double, ndim=2, mode="c"] user_factors, numpy.ndarray[double, ndim=2, mode="c"] item_factors):
        """
        Compute the BPR objective which is sum_uij ln sigma(x_uij) + lambda ||theta||^2
        """   
        cdef unsigned int u, i, j, normalisation
        cdef double x, ranking_loss, complexity  
        
        ranking_loss = 0;
        for u,i,j in loss_samples:
            x = self.predict(user_factors, item_factors, u, i) - self.predict(user_factors, item_factors, u, j)
            normalisation += 1
            try: 
                ranking_loss += log(1.0/(1.0+exp(-x)))
            except OverflowError: 
                #logging.warning("overflow")
                if x > 0: 
                    ranking_loss += log(1.0) 
                elif x < 0: 
                    #Really it should be minus infinity 
                    ranking_loss += -1000
        
        #ranking_loss /= normalisation

        complexity = 0
        for u,i,j in loss_samples:
            complexity += self.user_regularization * (user_factors[u]**2).sum()
            complexity += self.positive_item_regularization * (item_factors[i]**2).sum()
            complexity += self.negative_item_regularization * (item_factors[j]**2).sum()

        return ranking_loss - 0.5*complexity

    def predict(self, numpy.ndarray[double, ndim=2, mode="c"] user_factors, numpy.ndarray[double, ndim=2, mode="c"] item_factors, unsigned int u, unsigned int i):
        #Note: item_basis is not in the origin paper         
        return numpy.dot(user_factors[u], item_factors[i])


class Sampler(object):
    def __init__(self, max_samples=None):
        self.max_samples = max_samples
        self.numAucSamples = 5

class UniformUserUniformItem(Sampler):
    def generate_samples(self, data):
        cdef unsigned int u, m, n 
        cdef unsigned int i, j
        #cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] rowInds 
        m, n = data.shape
            
        #Note that we ideally need O(m * n^2) samples but picking               
        for _ in xrange(data.nnz * self.numAucSamples):
            u = numpy.random.randint(0, m)
            # sample positive item
            rowInds = numpy.array(data.rowInds(u), dtype=numpy.uint32)
            i = numpy.random.choice(rowInds)
            j = inverseChoice(rowInds, n)

            yield u, i, j
       
class AllUserUniformItem(Sampler):
    def generate_samples(self, data):
        cdef unsigned int u, m, n 
        cdef unsigned int i, j, s
        #cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] rowInds 
        m, n = data.shape
        
        userInds = numpy.random.permutation(m)

           
        #Note that we ideally need O(m * n^2) samples but picking               
        for u in userInds:
            rowInds = numpy.array(data.rowInds(u), dtype=numpy.uint32)
            for s in range(self.numAucSamples): 
                i = numpy.random.choice(rowInds)
                j = inverseChoice(rowInds, n)

                yield u, i, j       
       
class AllUserAllItem(Sampler):
    def generate_samples(self, data):
        cdef unsigned int u, m, n 
        cdef unsigned int i, j
        #cdef numpy.ndarray[unsigned int, ndim=1, mode="c"] rowInds 
        m, n = data.shape
            
        for u in range(m): 
            positiveItems = numpy.array(data.rowInds(u), dtype=numpy.uint32)
            negativeItems = numpy.setdiff1d(numpy.arange(n), positiveItems)
            
            for i in positiveItems: 
                for j in negativeItems: 
                    yield u, i, j
            