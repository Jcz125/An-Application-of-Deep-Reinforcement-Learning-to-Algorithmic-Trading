# coding=utf-8
import PIL
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from IPython import display
from time import sleep
import numpy as np

default_fig_size = (18.0, 8.0)

class DisplayOption:
    def __init__(self, saveToDisk=False, notebook=False, interactive=False, recordVideo=False):
        self.saveToDisk = saveToDisk
        self.notebook = notebook
        self.interactive = interactive
        self.recordVideo = recordVideo
    
    def __bool__(self):
        return self.saveToDisk or self.interactive or self.notebook or self.recordVideo
        

class DisplayManager:
    def __init__(self, displayOptions, **kwargs):
        defaultKwargs = { 'figsize': default_fig_size }
        kwargs = { **defaultKwargs, **kwargs }
        plt.ion()
        self.figure = plt.figure(**kwargs)
        self.displayOptions = displayOptions
        self.subplots = {}
        self.subplots_line_draw = {}
        self.display = display.display(self.figure, display_id=True)
        self.frames = []
        self.title = ""

    def __del__(self):
        if self.displayOptions.recordVideo:
            plt.close()
            self.figure = plt.figure(figsize=default_fig_size)
            ani = animation.ArtistAnimation(self.figure, self.frames, interval=50, blit=True, repeat_delay=1000)
            print(plt.rcParams['animation.ffmpeg_path'])
            # FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
            ani.save(f'{self.title}.gif')
            # video = animation.to_html5_video()
            # html = display.HTML(video)
            # display.display(html)
            self.figure.close()
                
    def figure(self):
        return self.figure
    
    def __bool__(self):
        return self.displayOptions.__bool__()
    
    def add_subplot(self, *args, **kwargs):
        ax = None
        if args in self.subplots:
            ax = self.subplots[args]
        else:
            ax = self.figure.add_subplot(*args, **kwargs)
            self.subplots[args] = ax
            self.subplots_line_draw[ax] = {}
        self.subplots[args].cla()
        ax.grid(True, which='both')
        ax.set_autoscale_on(True)
        ax.autoscale_view(True, True, True)
        return ax
    
    def plot(self, ax, lineId, x, y, *args, **kwargs):
        if True or lineId not in self.subplots_line_draw[ax]:
            line, = ax.plot(x, y, *args, **kwargs)
            self.subplots_line_draw[ax][lineId] = line
            return line
        line = self.subplots_line_draw[ax][lineId]
        line.set_data(x, y)
        if 'marker' in kwargs:
            line.set_marker(kwargs['marker'])
        return line

    def show(self, title):
        self.title = title
        if self.displayOptions.recordVideo:
            self.figure.canvas.draw()
            imageFramebuffer = PIL.Image.frombytes('RGB', self.figure.canvas.get_width_height(), self.figure.canvas.tostring_rgb())
            self.frames.append([plt.imshow(imageFramebuffer, cmap=cm.Greys_r, animated=True)])
        if self.displayOptions.interactive:
            ## Enable this if you use vanilla notebook or google collab
            # self.figure.suptitle(title)
            # self.figure.canvas.draw()
            # self.figure.canvas.flush_events()
            # self.display.update(self.figure)
            ## Enable this for VS Code (it's buggy)
            display.clear_output(wait=True)
            self.figure.suptitle(title)
            self.figure.canvas.draw()
            display.display(self.figure)
        elif self.displayOptions.notebook:
            self.figure.suptitle(title)
            self.figure.canvas.draw()
            display.display(self.figure)
            plt.close()
        if self.displayOptions.saveToDisk:
            plt.savefig(''.join(['Figures/', title, '.png']))
        
