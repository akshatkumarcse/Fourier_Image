from Image import Image
from Fourier import Fourier
from Plot import Plot


im = Image("house.jpg", (200, 200))
path = im.sort()
period, tup_circle_rads, tup_circle_locs = Fourier(n_approx = 1000, coord_1 = path).get_circles()
Plot(period, tup_circle_rads, tup_circle_locs, speed = 40).plot()
