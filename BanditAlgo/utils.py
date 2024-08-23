import time
import numpy as np
import torch as t
from torch import nn
import collections
import inspect
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        #Save function arguments into class attributes
        frame = inspect.currentframe().f_back #find the callers frame -> fetches variables from class that calls save_hyperparameters
        _, _, _, local_vars = inspect.getargvalues(frame) #get values from callers frame
        self.hparams = {k:v for k,v in local_vars.items() if k not in set(ignore+["self"]) and not k.startswith('_')} #cannot overwrite inheret functions starting with __
    for k, v in self.hparams.items():
        setattr(self, k, v) #if object has __dict__() method then new       variable can be given to class

class ProgressBoard(HyperParameters):
    def __init__(self, xlabel = None, ylabel = None, xlim = None, ylim =        None, xscale = 'linear', yscale = 'linear', ls = ['-', '--', '-.', ':'],        colors = ['C0', 'C1', 'C2', 'C3'], fig = None, axes = None, figsize = (3.5,     2.5), display = True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        backend_inline.set_matplotlib_formats('png')
        if self.fig is None:
            self.fig = plt.figure(figsize = self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v], linestyle = ls, color = color)[0])
            labels.append(k)

        #Figure specifications
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        plt.show()
        plt.pause(0.001) #Displays only for the pause time

