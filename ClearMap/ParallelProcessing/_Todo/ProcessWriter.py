# -*- coding: utf-8 -*-
"""
Provides simple formatting tools to print text with parallel process header
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright 2017 by Christoph Kirst, The Rockefeller University, New York City'


import sys
import multiprocessing


class ProcessWriter(object):
    """Class to handle writing from parallel processes
    
    Attributes:
        process (int): the process number
    
    """
    
    def __init__(self, process = None):
        self.process = process;
        
        try:
          self.identity = multiprocessing.current_process()._identity[0];
        except:
          self.identity = multiprocessing.current_process().pid;
        
        try:
          self.identity = ('Worker %3d' % int(self.identity));
        except:
          self.identity = str(multiprocessing.current_process());
    
    
    def writeString(self, text):
        """Generate string with process prefix
        
        Arguments:
            text (str): the text input
            
        Returns:
            str: text with [process prefix
        """
        if self.process is not None:
            pre = ("%s - Process %5s: " % (self.identity, ('%d' % self.process)));
        else: 
            pre = ("%s: " % self.identity);
            
        return pre + str(text).replace('\n', '\n' + pre);
    
    
    def write(self, text):
        """Write string with process prefix to sys.stdout
        
        Arguments:
            text (str): the text input
        """
        
        print(self.writeString(text));
        sys.stdout.flush();
        
    #def print(self, text):
    #    return self.write(text);


if __name__ == "__main__":
    #%%
    import ClearMap.ParallelProcessing.ProcessWriter as pw;
    reload(pw)
    
    def f(x):
       w = pw.ProcessWriter();
       w.write('hello world %d' % x);
       #proc =pw.multiprocessing.current_process();
       #print(proc)
       #print(proc.pid)
       #print(proc.)
     
    
    pw.multiprocessing.process.current_process()._counter = pw.multiprocessing.process.itertools.count(1)
    p = pw.multiprocessing.Pool(3)
    #give each processes a name
    #for i,pp in enumerate(p._pool):
    #  pp.qid = '%d' % i;
    #  pp._identity = (i,);
    
    #pw.multiprocessing.pool.job_counter =  pw.multiprocessing.pool.itertools.count()
    
    #pw.multiprocessing.process._process_counter = pw.multiprocessing.process.itertools.count(1)
    
    
    p.map(f, range(100))