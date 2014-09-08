import logging 
import sys 
import unittest
import numpy
import sppy 
import numpy.testing as nptst 
from bpr import BPRArgs, BPR, UniformPairWithoutReplacement, Sampler, UniformUserUniformItem 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator

class  bprTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(precision=3, suppress=True)
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        
        self.m = 30 
        self.n = 20 
        k = 5
        u = 0.1 
        w = 1-u
        self.X = SparseUtils.generateSparseBinaryMatrix((self.m, self.n), k, w, csarray=True)
        self.X.prune()

    def testInit(self): 
        args = BPRArgs()   
        k = 5
        
        learner = BPR(k, args)
        
    def testTrain(self): 
        args = BPRArgs()   
        args.learning_rate = 0.1
        k = 5
        
        learner = BPR(k, args)    
        
        maxIterations = 10
        sample_negative_items_empirically = True
        sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)

        learner.train(self.X, sampler, maxIterations)
        print(MCEvaluator.averageAuc(self.X, learner.user_factors, learner.item_factors))
        
        #Let's try regularisation 
        args.user_regularization = 1
        learner.train(self.X, sampler, maxIterations)
        
        #Let's try regularisation 
        args.positive_item_regularization = 1
        learner.train(self.X, sampler, maxIterations)

    def testSampler(self): 
        sampler = Sampler(False)
        
        sampler.init(self.X, max_samples=10)
        users = set([])   
        
        for i in range(1000): 
            u = sampler.sample_user()
            users.add(u)
        
        nptst.assert_array_equal(list(users), numpy.arange(self.m))
        
        #Test items 
        items = set([])   
        
        for i in range(1000): 
            item = sampler.random_item()
            items.add(item)
        
        nptst.assert_array_equal(list(items), numpy.arange(self.n))
        
        #Test negative item sampling 
        items = set([])        
        
        for i in range(1000): 
            item = sampler.sample_negative_item(self.X.rowInds(0))
            self.assertTrue(item not in self.X.rowInds(0))      

        
        #Now test UniformPairWithoutReplacement
        sampler = UniformPairWithoutReplacement(False)
        samples = sampler.generate_samples(self.X, 100)
        
        for i in samples: 
            u, p, q = i 
            #print(u, p, q, self.X[u, p], self.X[u, q])
            self.assertTrue(0 <= u and u < self.m)
            self.assertEquals(self.X[u, p], 1)
            self.assertEquals(self.X[u, q], 0)
            

    def testLoss(self): 
        args = BPRArgs()   
        k = 5
        
        learner = BPR(k, args)   
        learner.init(self.X)
        
        ell = learner.loss()
        #print(ell)
    
    def testupdate_factors(self): 
        args = BPRArgs()   
        k = 5
        
        learner = BPR(k, args)   
        learner.init(self.X)

    def testDerivatives(self): 
        args = BPRArgs()   
        args.user_regularization = 0.0 
        args.negative_item_regularization = 0 
        args.positive_item_regularization = 0 
        args.bias_regularization = 0
        args.learning_rate = 0 
        k = 5
        
        learner = BPR(k, args)   
        learner.init(self.X)
        
        eps = 10**-8
        
        sampler = UniformUserUniformItem(sample_negative_items_empirically=True)
        
        #Get user, pos item and negative item 
        u = 0 
        
        user_factors = learner.user_factors.copy()
        item_factors = learner.item_factors.copy()            
        
        #Compute user derivate via perturbations 
        du1 = numpy.zeros(k)   
        du2 = numpy.zeros(k)         
        
        inds = self.X.rowInds(u)
        indsBar = numpy.setdiff1d(numpy.arange(self.X.shape[1]), inds)
        
        for i in inds: 
            for j in indsBar: 
                for ell in range(k): 
                    learner.user_factors = user_factors.copy()                
                    
                    learner.user_factors[u, ell] = user_factors[u, ell]+eps
                    loss1 = learner.lossExact()
                    learner.user_factors[u, ell] = user_factors[u, ell]-eps
                    loss2 = learner.lossExact() 
                    
                    learner.user_factors[u, ell] = user_factors[u, ell]
                    du1[ell] += (loss1-loss2)/(2*eps)
                    
                du3, di3, dj3 = learner.update_factors(u, i, j)
                du2 += du3    
                
        du1 = du1/numpy.linalg.norm(du1)      
        du2 = du2/numpy.linalg.norm(du2)        
        
        print(du1, du2)
               
        """
        di1 = numpy.zeros(k)  
        di2 = numpy.zeros(k)

        for u in range(self.m): 
            print(u)
            inds = self.X.rowInds(u)
            indsBar = numpy.setdiff1d(numpy.arange(self.X.shape[1]), inds)            
            
            learner.item_factors = item_factors.copy()             
            
            for j in indsBar: 
                for ell in range(k): 
                    learner.item_factors[i, ell] = item_factors[i, ell]+eps
                    loss1 = learner.lossExact()
                    learner.item_factors[i, ell] = item_factors[i, ell]-eps
                    loss2 = learner.lossExact() 
                    
                    learner.item_factors[i, ell] = item_factors[i, ell]
                    di1[ell] += (loss1-loss2)/(2*eps)
                    
                du3, di3, dj3 = learner.update_factors(u, i, j)
                di2 += di3    
                
        di1 = di1/numpy.linalg.norm(di1)      
        di2 = di2/numpy.linalg.norm(di2)   
        
        print(di1, di2)
        """

        


        dj1 = numpy.zeros(k)  
        dj2 = numpy.zeros(k)

        for u in range(self.m): 
            print(u)
            inds = self.X.rowInds(u)
            indsBar = numpy.setdiff1d(numpy.arange(self.X.shape[1]), inds)            
            
            learner.item_factors = item_factors.copy()             
            
            for i in inds: 
                for ell in range(k): 
                    learner.item_factors[i, ell] = item_factors[i, ell]+eps
                    loss1 = learner.lossExact()
                    learner.item_factors[i, ell] = item_factors[i, ell]-eps
                    loss2 = learner.lossExact() 
                    
                    learner.item_factors[i, ell] = item_factors[i, ell]
                    dj1[ell] += (loss1-loss2)/(2*eps)
                    
                du3, di3, dj3 = learner.update_factors(u, i, j)
                dj2 += dj3    
                
        dj1 = dj1/numpy.linalg.norm(dj1)      
        dj2 = dj2/numpy.linalg.norm(dj2)   


        print(dj1, dj2)


if __name__ == '__main__':
    unittest.main()