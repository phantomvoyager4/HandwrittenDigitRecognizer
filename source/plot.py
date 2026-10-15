import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

TRAIN_COLOR = '#2a78d6'
TEST_COLOR = '#eb6834'
SURFACE_COLOR = '#ffffff'
GRID_COLOR = '#e4e3df'
TEXT_COLOR = '#52514e'


class AccuracyPlot:
    def __init__(self, epochs):
        plt.ion()
        self.figure, (self.train_axis, self.test_axis) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        self.figure.canvas.manager.set_window_title('Training progress')
        self.train_epochs, self.train_accuracy = [], []
        self.test_epochs, self.test_accuracy = [], []
        self.train_line = self.setup_axis(self.train_axis, 'Accuracy (training set)', TRAIN_COLOR, epochs)
        self.test_line = self.setup_axis(self.test_axis, 'Test_Accuracy (test set)', TEST_COLOR, epochs)
        #test accuracy is only measured every few epochs, so mark each measurement
        self.test_line.set(marker='o', markersize=7, markeredgecolor=SURFACE_COLOR, markeredgewidth=2)
        self.train_axis.set_ylabel('Accuracy', color=TEXT_COLOR)
        self.figure.tight_layout()
        plt.show(block=False)

    def setup_axis(self, axis, title, color, epochs):
        axis.set_title(title, loc='left', fontweight='bold')
        axis.set_xlim(0, epochs)
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
        line, = axis.plot([], [], color=color, linewidth=2, solid_capstyle='round', solid_joinstyle='round', clip_on=False, in_layout=False)
        return line

    def add_train_accuracy(self, epoch, accuracy):
        self.train_epochs.append(epoch)
        self.train_accuracy.append(accuracy)

    def add_test_accuracy(self, epoch, accuracy):
        self.test_epochs.append(epoch)
        self.test_accuracy.append(accuracy)

    def refresh(self):
        self.train_line.set_data(self.train_epochs, self.train_accuracy)
        self.test_line.set_data(self.test_epochs, self.test_accuracy)
        #latest value shown in the top right corner of each plot
        if self.train_accuracy:
            self.train_axis.set_title(f"{self.train_accuracy[-1]:.2%} at epoch {self.train_epochs[-1]}", loc='right', color=TEXT_COLOR)
        if self.test_accuracy:
            self.test_axis.set_title(f"{self.test_accuracy[-1]:.2%} at epoch {self.test_epochs[-1]}", loc='right', color=TEXT_COLOR)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def save(self, path):
        self.refresh()
        self.figure.savefig(path, dpi=150)

    def show(self):
        #keeps the window open after training until the user closes it
        self.refresh()
        plt.ioff()
        plt.show()
