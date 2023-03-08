# coding=utf-8
from matplotlib import pyplot as plt
from IPython.display import display, clear_output

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
        self.displayOptions = displayOptions
        self.figure = plt.figure(**kwargs)
        self.subplots = {}
    
    def figure(self):
        return self.figure
    
    def __bool__(self):
        return self.displayOptions.saveToDisk or self.displayOptions.interactive
    
    def add_subplot(self, *args, **kwargs):
        if args in self.subplots:
            return self.subplots[args]
        self.subplots[args] = self.figure.add_subplot(*args, **kwargs)
        self.subplots[args].grid(True, which='both')
        return self.subplots[args]

    def show(self, title):
        if self.displayOptions.interactive:
            clear_output(wait=True)
            self.figure.suptitle(title)
            display(self.figure)
        if self.displayOptions.saveToDisk:
            plt.savefig(''.join(['Figures/', title, '.png']))
        
