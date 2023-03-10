# coding=utf-8
from matplotlib import pyplot as plt
from IPython import display
from time import sleep

class DisplayOption:
    def __init__(self, saveToDisk=True, interactive=False):
        self.saveToDisk = saveToDisk
        self.interactive = interactive
    
    def __bool__(self):
        return self.saveToDisk or self.interactive
        

class DisplayManager:
    def __init__(self, displayOptions, **kwargs):
        defaultKwargs = { 'figsize': (20.0, 10.0) }
        kwargs = { **defaultKwargs, **kwargs }
        plt.ion()
        self.figure = plt.figure(**kwargs)
        self.displayOptions = displayOptions
        self.subplots = {}
        self.subplots_line_draw = {}
        self.display = display.display(self.figure, display_id=True)
    
    def figure(self):
        return self.figure
    
    def __bool__(self):
        return self.displayOptions.saveToDisk or self.displayOptions.interactive
    
    def add_subplot(self, *args, **kwargs):
        if args in self.subplots:
            return self.subplots[args]
        ax = self.figure.add_subplot(*args, **kwargs)
        self.subplots[args] = ax
        ax.grid(True, which='both')
        # ax.set_autoscale_on(True)
        # ax.autoscale_view(True, True, True)
        self.subplots_line_draw[ax] = {}
        return ax
    
    def plot(self, ax, lineId, x, y, **kwargs):
        if lineId not in self.subplots_line_draw[ax]:
            line, = ax.plot(x, y, **kwargs)
            self.subplots_line_draw[ax][lineId] = line
            return line
        line = self.subplots_line_draw[ax][lineId]
        line.set_data(x, y)
        if 'marker' in kwargs:
            line.set_marker(kwargs['marker'])
        return line

    def show(self, title):
        if self.displayOptions.interactive:
            ## Enable this if you use vanilla notebook or google collab
            # self.figure.suptitle(title)
            # self.figure.canvas.draw()
            # self.figure.canvas.flush_events()
            # self.display.update(self.figure)

            ## Enable this for VS Code (it's buggy)
            display.clear_output(wait=True)
            self.figure.suptitle(title)
            display.display(self.figure)
        if self.displayOptions.saveToDisk:
            plt.savefig(''.join(['Figures/', title, '.png']))
        
