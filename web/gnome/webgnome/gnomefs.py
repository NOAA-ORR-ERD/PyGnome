import multiprocessing 
import os
import time

'''Handles process and file ops on the GNOME backend'''

def gnome_proc(n,t,a):  
    '''Runs in process address space such that n = process name, t = function of method call, a = arguments'''
    p = multiprocessing.Process(name=n,target=t, args=(a,))
    p.start() 
    print 'Running process ', p.pid
    p.join()
    return t(a)

def gnome_proc_timed(n,t,tstar):  
    '''Runs in '''
    p = multiprocessing.Process(name=n,target =t)
    p.start() 
    time.sleep(tstar)
    print 'Running process ', p.pid
    p.join()
    
def gnome_proc_info(pid):  
    p = multiprocessing.Process(name=n,target =t)
    print 'Running process ', p.pid
    p.join()
    
def gnome_proc_end(pid):
    '''End process'''
    os.system('kill ',pid)
    
def gnome_pool(n,f,d):
    '''Finite multiprocessing'''
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Process(pool_size=pool)
    pool.map(f,d)

def gnome_lock(proc):
    '''Proctect data in GNOME'''
    p = multiprocessing.Process(name=n,target =t)
    p.start() 
    print 'Running process ', p.pid
    p.join() 

def gnome_house_keeper(interval,time):
    '''Clean out spill runs'''
    return {}
    
def test():
    return ('t1','t2')
    
    
    
     
    