import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

TRAIN_COLOR = '#2a78d6'
VALIDATION_COLOR = '#eb6834'
SURFACE_COLOR = '#ffffff'
GRID_COLOR = '#e4e3df'
TEXT_COLOR = '#52514e'
MARKER_LIMIT = 50 #with more epochs than this, markers on every point only clutter the line


class AccuracyPlot:
    def __init__(self, epochs, live=True):
        #live=False records the values and only draws the figure when it is saved
        self.live = live
        if self.live:
            plt.ion()
        self.figure, (self.train_axis, self.validation_axis) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        self.figure.canvas.manager.set_window_title('Training progress')
        self.train_epochs, self.train_accuracy = [], []
        self.validation_epochs, self.validation_accuracy = [], []
        self.train_line = self.setup_axis(self.train_axis, 'Accuracy (training set)', TRAIN_COLOR, epochs)
        self.validation_line = self.setup_axis(self.validation_axis, 'Accuracy (validation set)', VALIDATION_COLOR, epochs)
        if epochs <= MARKER_LIMIT:
            for line in (self.train_line, self.validation_line):
                line.set(marker='o', markersize=5, markeredgecolor=SURFACE_COLOR, markeredgewidth=1.5)
        self.train_axis.set_ylabel('Accuracy', color=TEXT_COLOR)
        self.figure.tight_layout()
        if self.live:
            plt.show(block=False)

    def setup_axis(self, axis, title, color, epochs):
        axis.set_title(title, loc='left', fontweight='bold')
        axis.set_xlim(0, max(epochs - 1, 1))
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.set_xlabel('Epoch', color=TEXT_COLOR)
        axis.grid(axis='y', color=GRID_COLOR, linewidth=1)
        axis.set_axisbelow(True)
        axis.tick_params(colors=TEXT_COLOR)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            axis.spines[side].set_color(GRID_COLOR)
        line, = axis.plot([], [], color=color, linewidth=2, solid_capstyle='round', solid_joinstyle='round', in_layout=False)
        return line

    def add_train_accuracy(self, epoch, accuracy):
        self.train_epochs.append(epoch)
        self.train_accuracy.append(accuracy)

    def add_validation_accuracy(self, epoch, accuracy):
        self.validation_epochs.append(epoch)
        self.validation_accuracy.append(accuracy)

    def draw(self):
        self.train_line.set_data(self.train_epochs, self.train_accuracy)
        self.validation_line.set_data(self.validation_epochs, self.validation_accuracy)
        #latest value shown in the top right corner of each plot
        if self.train_accuracy:
            self.train_axis.set_title(f"{self.train_accuracy[-1]:.2%} at epoch {self.train_epochs[-1]}", loc='right', color=TEXT_COLOR)
        if self.validation_accuracy:
            self.validation_axis.set_title(f"{self.validation_accuracy[-1]:.2%} at epoch {self.validation_epochs[-1]}", loc='right', color=TEXT_COLOR)
        #zoom the shared y axis on the range where most of training happens; the first few epochs may fall below it
        values = self.train_accuracy + self.validation_accuracy
        if values:
            lower = math.floor(np.percentile(values, 5) * 20) / 20 - 0.05
            self.train_axis.set_ylim(max(0.0, lower), 1)

    def refresh(self):
        if not self.live:
            return
        self.draw()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def save(self, path):
        self.draw()
        self.figure.savefig(path, dpi=150)

    def show(self):
        #keeps the window open after training until the user closes it
        if not self.live:
            return
        self.refresh()
        plt.ioff()
        plt.show()
