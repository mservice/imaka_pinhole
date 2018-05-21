import matplotlib.pyplot as plt
import numpy as np


def plot():
    '''
    makes error budget figure
    '''

    f, (ax3, ax2, ax1) = plt.subplots(3, 1, sharex=True)

    ax1.bar([2.25, 3.28, 2.72, 3, 3.75], [29, 38,714, 1259,  14], [0.25,0.25,0.25,0.25,0.25], [0,0,0,0,60.8], color='blue', edgecolor='black')
    ax1.bar([3.75, 3.75], [47, 13.8], [0.25, 0.25] , [0, 47], color='teal', edgecolor='black')

    ax2.bar([1.5, 3.28, 2.72, 3, 4.5], [29, 38,714, 1259,  14], [0.25,0.25,0.25,0.25,0.25], [0,0,0,0,60.8], color='blue', label='Static', edgecolor='black')
    ax2.bar([3.75, 3.75], [47, 13.8], [0.25, 0.25] , [0, 47], color='teal', label='Time variable', edgecolor='black')

    ax3.bar([1.5, 3.28, 2.72, 3, 4.5], [29, 38,714, 1259,  14], [0.25,0.25,0.25,0.25,0.25], [0,0,0,0,60.8], color='blue', label='Static', edgecolor='black')
    ax3.bar([3.75, 3.75], [47, 13.8], [0.25, 0.25] , [0, 47], color='teal', label='Time variable', edgecolor='black')

    plt.xticks((2.25, 3, 3.75), ['Mask', 'Optical Distortion', 'Enivronment'])
    plt.subplots_adjust(hspace=0.1)

    ax1.text(2.25, 15, r'$\mathcal{O}(>1)$', va='center', ha='center', weight='bold')
    ax1.text(3.28, 15, r'$\mathcal{O}(>4)$', va='center', ha='center', weight='bold')
    ax1.text(2.72, 75, r'$\mathcal{O}(1)$', va='center', ha='center', weight='bold')
    ax1.text(3, 75, r'$\mathcal{O}(2-4)$', va='center', ha='center', weight='bold')
    ax1.text(3.75, 20, r'$\mathcal{O}(1)$', va='center', ha='center', weight='bold')
    ax1.text(3.75, 60, r'$\mathcal{O}(>1)$', va='center', ha='center', weight='bold')
    ax1.set_ylim(0, 150)
    ax2.set_ylim(700,740)
    ax2.set_ylabel('Error (nm)')
    ax3.set_ylim(1250, 1290)
    
    ax3.legend(loc='upper left')
    ax1.set_xlim(2, 4)
    

    ax2.spines['bottom'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax2.xaxis.set_ticks_position('none')
    ax3.xaxis.set_ticks_position('none') 
    #ax2.xaxis.tick_top()
    #ax2.tick_params(labeltop='off')
    #ax3.xaxis.tick_top()
    #ax3.tick_params(labeltop='off')
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    d = .015
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal


    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal


    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
    ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    
    kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
    ax3.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax3.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    

    #add labels?
    
    
