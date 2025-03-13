"""
    TODO
"""

from osgar.node import Node
from osgar.lib.route import Convertor
from lib.localization import Localization


def list2xy(data):
    x = [coord[0] for coord in data]
    y = [coord[1] for coord in data]
    return x, y


class TreeNode(Node):

    def __init__(self, config, bus):
        super().__init__(config, bus)
        bus.register('tree')  # register a stream to be published
        self.verbose = False

    def on_pose3d(self, data):
        pass

    def on_color(self, data):
        pass

    def on_depth(self, data):
        pass


    def draw(self):
        # in verbose mode and with --draw parameter: draw a plot
        import matplotlib.pyplot as plt
        # x, y = list2xy(self.debug_gps_orig)
        # plt.plot(x, y, "k+-", label="gps orig")

        # plt.legend()
        # plt.axis('equal')
        # plt.show()


