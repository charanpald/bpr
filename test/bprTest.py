import logging 
import sys 
import unittest
import numpy
import numpy.testing as nptst 
from bpr import BPRArgs, BPR, UniformPairWithoutReplacement, Sampler 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator

class  bprTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(precision=3, suppress=True)
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        
        self.m = 50 
        self.n = 20 
        k = 5
        u = 0.1 
        w = 1-u
        self.X = SparseUtils.generateSparseBinaryMatrix((self.m, self.n), k, w, csarray=False)
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
        
        maxIterations = 100
        sample_negative_items_empirically = True
        sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
        learner.train(self.X, sampler, maxIterations)
        
        #Check the AUC is large
        print(MCEvaluator.averageAuc(self.X, learner.user_factors, learner.item_factors))

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
            item = sampler.sample_negative_item(self.X[0].nonzero()[1])
            self.assertTrue(item not in self.X[0].nonzero()[1])      

        
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

if __name__ == '__main__':
    unittest.main()