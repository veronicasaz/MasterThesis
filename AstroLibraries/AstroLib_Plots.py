import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class Plot:
    def __init__(self, titleFig, nrows = 1, ncols = 1, *args, **kwargs):
        """          
        titleFig: title of the figure
        
        additional arguments:
        """
       # Number of of the subplots
        self.n = nrows
        self.m = ncols

        # Size of fig
        self.h = kwargs.get('h',10)
        self.v = kwargs.get('v',15) 

        self.titleFig = titleFig

        # Markers
        self.colors = ['black','red', 'blue', 'green','orange','yellow','magenta','cyan']
        self.marker = ['x','o','-','v','^','s','h','+']
        self.line = ['-','--','-.',':'] 

    def createPlot(self, *args, **kwargs):
        self.titleSize = kwargs.get('titleSize',15)
        
        self.h_space = kwargs.get('hspace',0.4)
        self.w_space = kwargs.get('wspace',0.25)
    
        self.top_margin = kwargs.get('top',0.92)
        self.bot_margin = kwargs.get('bottom',0.08)
        self.right_margin = kwargs.get('right',0.95)
        self.left_margin = kwargs.get('left',0.1)

        # Create fig and subplots
        self.fig, self.ax = plt.subplots(self.n, self.m, figsize=(self.h, self.v))    
        self.fig.suptitle(self.titleFig, size = self.titleSize )
        self.fig.subplots_adjust(top=self.top_margin, bottom=self.bot_margin, left=self.left_margin, right=self.right_margin, hspace=self.h_space, wspace=self.w_space)

        self.listSubplot = dict()

    def size(self, *args, **kwargs):
        self.titleSize = kwargs.get('titleSize',15)
        self.axisSize = kwargs.get('axisSize',15)
        self.legendSize = kwargs.get('legendSize',10)
        self.ticksLabel = kwargs.get('ticksLabel',10)

    def addSubplot(self, name, titleFig, order, *args, **kwargs):
        # Type plot
        self.name = name
        self.titleFig = titleFig
        self.listSubplot[order] = plt.subplot(self.n, self.m, order)

    def plot(self, x, y, order, *args, **kwargs):
        # Data
        self.x = x
        self.y = y

        ax = self.listSubplot[order]

        # Plot labels
        self.xlabel = kwargs.get('xlabel', ' ')
        self.ylabel = kwargs.get('ylabel', ' ') 
        self.labels = kwargs.get('labels', [''])

        self.legend = kwargs.get('legend',False)
        self.grid = kwargs.get('grid', False)
        self.gridAlpha = kwargs.get('gridAlpha', 0.5 )

        self.markersSubpl = kwargs.get('markers', self.marker)
        self.colorsSubpl = kwargs.get('colors', self.colors)
        self.lineSubpl = kwargs.get('line', self.line)

        self.type_plot = kwargs.get('type_plot','other')

        if self.type_plot == 'scatter':

            # Cases for more than one list per plot
            if type(x) == 'list':
                if type(y) == 'list':
                    for i in range(len(x)):
                        ax.scatter(self.x[i] , self.y[i], label = self.labels[i], marker = self.markersSubpl[i] , color = self.colorsSubpl[i])
                else:
                    for i in range(len(x)):
                        ax.scatter(self.x[i] , self.y, label = self.labels[i], marker = self.markersSubpl[i] , color = self.colorsSubpl[i])
            else:
                if type(y) == 'list':
                    for i in range(len(y)):
                        ax.scatter(self.x , self.y[i], label = self.labels[i], marker = self.markersSubpl[i] , color = self.colorsSubpl[i])
                else:  
                    ax.scatter(self.x , self.y, label = self.labels, marker = self.markersSubpl[0] , color = self.colorsSubpl[0])
        else:
                ax.plot(x , y ,label = self.labels, linestyle = self.line[0] , color = self.colorsSubpl[0])
        

        ax.set_title(self.titleFig, size = self.titleSize)
        plt.grid(self.grid, alpha = self.gridAlpha)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        if self.legend == 'True':
            plt.legend()


    # def show(self):
    #     plt.show()

    def save(self, name, *args, **kwargs):
        self.dpi = kwargs.get('dpi', 200) 
        self.layoutSave = kwargs.get('layout', 'tight')
        plt.savefig(name, dpi = self.dpi, bbox_inches = self.layoutSave)

    def show(self):
        plt.show()




    

def TransferPlot(r_E, r_M):
    """
    FinalPlot: plot of the trajectory with the impulses
    Inputs: 
        SMatrx_E: matrix containing [x,y,z,vx,vy,vz,m,ind] of the propagation from the Earth
        SMatrx_M: matrix containing [x,y,z,vx,vy,vz,m,ind] of the propagation from Mars
    Outputs:
        -
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r_E[0], r_E[1], r_E[2], color = 'blue') #, s=50, color="dodgerblue",marker="*") # Plot trajectory from the Earth
    ax.scatter(r_M[0], r_M[1], r_M[2], color = 'red') #, s=50, color="orangered",marker="*") # Plot trajectory from Mars

    # ax.plot([SMatrx_M[0,1],SMatrx_E[0,1]],[SMatrx_M[0,2],SMatrx_E[0,2]],color='black') # Line to join the two initial points
    ax.scatter(0,0,0,color = 'orange') #, marker='o', markersize=25, color="orange") # Plot Sun

    # for i in range(len(SMatrx_E[:,0])): #If there is an impulse, plot a circle
    #     if SMatrx_E[i,-1]==1: #There is an impulse
    #         ax.scatter(SMatrx_E[i,1],SMatrx_E[i,2],s=120,color="dodgerblue",marker="^") #plot in a different color where the impulse is applied
    # for i in range(len(SMatrx_E[:,0])): #If there is an impulse, plot a circle            
    #     if SMatrx_M[i,-1]==1: #There is an impulse
    #         ax.scatter(SMatrx_M[i,1],SMatrx_M[i,2],s=120,color="orangered",marker="v")
    
    # ax.scatter(SMatrx_E[0,1],SMatrx_E[0,2],s=200,color="dodgerblue",marker="o",label='Earth') # Plot Earth
    # ax.scatter(SMatrx_M[0,1],SMatrx_M[0,2],s=200,color="orangered",marker="o",label ='Mars') # Plot Mars
    plt.show()



def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
