import numpy as np

# Fourier.get_circles
from itertools import chain

class Fourier(object):
    def __init__(self, n_approx = 1000, coord_1 = None, coord_2 = None):
        if coord_1 is not None:
            temp = coord_1[:,:,0] + 1j * coord_1[:,:,1]
            self.complex_coord_1 = temp.reshape(temp.shape[0])
        if coord_2 is not None:
            temp = coord_2[:,:,0] + 1j * coord_2[:,:,1]
            self.complex_coord_2 = temp.reshape(temp.shape[0])
            # To ensure the two images then have the same "frequency"
            if self.complex_coord_2.size > self.complex_coord_1.size:
                self.complex_coord_1 = np.hstack((self.complex_coord_1, np.full((self.complex_coord_2.size - self.complex_coord_1.size), self.complex_coord_1[-1], dtype = np.complex_)))
            elif self.complex_coord_1.size > self.complex_coord_2.size:
                self.complex_coord_2 = np.hstack((self.complex_coord_2, np.full((self.complex_coord_1.size - self.complex_coord_2.size), self.complex_coord_2[-1], dtype = np.complex_)))
        else:
            self.complex_coord_2 = None

        # Avoid aliasing
        self.n_approx = self.complex_coord_1.size//2 if n_approx > self.complex_coord_1.size//2  else n_approx

    def get_circles(self, mode=1):
        if self.complex_coord_1 is not None and self.complex_coord_2 is not None:
            return self.get_two_circles_two_images()
        elif mode == 2:
            return self.get_two_circles_one_image()
        return self.get_one_circle_one_image()
        
    def get_one_circle_one_image(self):
        period = self.complex_coord_1.size
        time   = np.arange(period)
        circles_loc = np.zeros((2*(self.n_approx-1), period), dtype = np.complex_)
        circles_rad = np.zeros((2*(self.n_approx-1)), dtype = np.float_)

        for idx, multiple in enumerate(chain(range(-self.n_approx+1, 0), range(1, self.n_approx))):
            # Fourier coefficient
            cn = self.cn(time, period, multiple, self.complex_coord_1)
            # Radius of circle
            circles_rad[idx] = np.absolute(cn)
            # Location of point on circle
            circles_loc[idx, :] = self.polar_locations(time, period, multiple, cn)

        # Sorting big to small
        order = np.argsort(circles_rad)[::-1]
        circles_loc = circles_loc[order]
        circles_rad = circles_rad[order]

        # Location of each circle's center and the final point
        circles_loc = np.add.accumulate(circles_loc, 0)
                        
        return period, (circles_rad,), (circles_loc,)

    def cn(self, time, period, multiple, coordinates):
        c = coordinates * np.exp(-1j * (2*multiple*np.pi/period) * time)
        return c.sum() / period

    def polar_locations(self, time, period, multiple, fourier_coeff):
        return np.absolute(fourier_coeff) * np.exp(1j * ((2*multiple*np.pi/period) * time + np.angle(fourier_coeff)))

    def cartesian_locations(self, time, period, multiple, fourier_coeff):
        return fourier_coeff * np.exp(1j * ((2*multiple*np.pi/period) * time))
