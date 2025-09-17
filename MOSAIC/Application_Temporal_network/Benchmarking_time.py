import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib
import sys
matplotlib.rcParams.update({'font.size': 10})

from lib_REGIR_IP_model import run_REGIR_simulation
from lib_Spanning_tree import run_Spanning_tree_simulation
from lib_ADM import run_ADM_simulation


def main():
    
    matplotlib.rcParams.update({'font.size': 16})
    
    Nlist = [5,10,20,40,80,160,320,640]
    Tend = 2000
    
    recompute = False
    if recompute:
    
        case = 'Basic'
        use_IP = False
        print('REGIR %s, use_IP = %s' % (case ,use_IP))
        sim_time_basic = []
        for N in Nlist:
            print(N)
            tstart = time.time()
            run_REGIR_simulation(N, Tend, case=case, use_IP=use_IP)
            tend = time.time()
            t_elapsed = tend - tstart
            print(t_elapsed)
            sim_time_basic.append(t_elapsed)
        np.save('Data/sim_time_basic.npy', sim_time_basic)
        plt.scatter(Nlist, sim_time_basic)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
            
        case = 'first_order'
        use_IP = False
        print('REGIR %s, use_IP = %s' % (case ,use_IP))
        sim_time_1order = []
        for N in Nlist:
            print(N)
            tstart = time.time()
            run_REGIR_simulation(N, Tend, case=case, use_IP=use_IP)
            tend = time.time()
            t_elapsed = tend - tstart
            print(t_elapsed)
            sim_time_1order.append(t_elapsed)
        np.save('Data/sim_time_1order.npy', sim_time_1order)
        plt.scatter(Nlist, sim_time_1order)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
            
    
    
        case = 'second_order'
        use_IP = False
        print('REGIR %s, use_IP = %s' % (case ,use_IP))
        sim_time_2order = []
        for N in Nlist:
            print(N)
            tstart = time.time()
            run_REGIR_simulation(N, Tend, case=case, use_IP=use_IP)
            tend = time.time()
            t_elapsed = tend - tstart
            print(t_elapsed)
            sim_time_2order.append(t_elapsed) 
        np.save('Data/sim_time_2order.npy', sim_time_2order)
        plt.scatter(Nlist, sim_time_2order)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()


        case = 'second_order'    
        use_IP = True
        print('REGIR %s, use_IP = %s' % (case ,use_IP))
        sim_time_IP = []
        for N in Nlist:
            print(N)
            tstart = time.time()
            run_REGIR_simulation(N, Tend, case=case, use_IP=use_IP)
            tend = time.time()
            t_elapsed = tend - tstart
            print(t_elapsed)
            sim_time_IP.append(t_elapsed)
        np.save('Data/sim_time_IP.npy', sim_time_IP)
        plt.scatter(Nlist, sim_time_IP)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

        sim_time_spanning = []
        print('Spanning trees')
        for N in Nlist:
            print(N)
            tstart = time.time()
            run_Spanning_tree_simulation(N, Tend)
            tend = time.time()
            t_elapsed = tend - tstart
            print(t_elapsed)
            sim_time_spanning.append(t_elapsed)
        np.save('Data/sim_time_spanning.npy', sim_time_spanning)
        plt.scatter(Nlist, sim_time_spanning)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    else:
        sim_time_basic = np.load('Data/sim_time_basic.npy')
        sim_time_1order = np.load('Data/sim_time_1order.npy')
        sim_time_2order = np.load('Data/sim_time_2order.npy')
        sim_time_IP = np.load('Data/sim_time_IP.npy')
        sim_time_spanning = np.load('Data/sim_time_spanning.npy')
        sim_time_ADM = np.load('Data/sim_time_ADM.npy')



    color_dict = dict()
    color_dict['REGIR_Basic'] = 'blue'
    color_dict['REGIR_first'] = 'orange'
    color_dict['ADM'] = 'green'
    color_dict['tree'] = 'purple'
    
    
    plt.figure(figsize=(4,4))
    plt.scatter(Nlist, sim_time_basic, label = 'REGIR-TN A', color = color_dict['REGIR_Basic'])
    plt.plot(Nlist, sim_time_basic, color = color_dict['REGIR_Basic'], ls = '--')
    plt.scatter(Nlist, sim_time_1order, label = 'REGIR-TN B', color = color_dict['REGIR_first'])
    plt.plot(Nlist, sim_time_1order, color = color_dict['REGIR_first'], ls = '--')
    #plt.scatter(Nlist, sim_time_2order, label = 'REGIR C')
    #plt.scatter(Nlist, sim_time_IP, label = 'REGIR D')
    plt.scatter(Nlist, sim_time_spanning, label = 'Spanning Tree', color = color_dict['tree'])
    plt.plot(Nlist, sim_time_spanning, color = color_dict['tree'], ls = '--')
    plt.scatter(Nlist, sim_time_ADM, label = 'AD modeling', color = color_dict['ADM'])
    plt.plot(Nlist, sim_time_ADM, color = color_dict['ADM'], ls = '--')
    plt.xscale('log')
    plt.yscale('log')
    x = np.array([4.5,1e3])
    y1 = np.array([0.1,(x[-1]/x[0])**2*0.1])*0.35
    y2 = y1/7
    plt.plot(x,y1/2.7, lw = 1, ls='--', color = 'black', alpha = 0.2)
    #plt.plot(x,y1, lw = 1, ls='--', color = 'black')
    #plt.plot(x,y2, lw = 1, ls='--', color = 'black')
    plt.fill_between(x, y1, y2, color = 'black', alpha = 0.05)
    plt.legend(fontsize = 10, loc = 'lower right')
    plt.xlabel('Number of nodes')
    plt.ylabel('Simulation time [s]')
    plt.xlim(xmin = 4.5, xmax = 1000)
    plt.show()
        
    
    
    
    
if __name__ == '__main__':
    main()