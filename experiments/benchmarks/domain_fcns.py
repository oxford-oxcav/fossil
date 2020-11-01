import torch
import numpy as np


def square_init_data(domain, batch_size):
    """
    :param domain: list = [lower_bounds, upper_bounds]
                    lower_bounds, upper_bounds are lists
    :param batch_size: int
    :return:
    """

    r1 = torch.tensor(domain[0])
    r2 = torch.tensor(domain[1])
    square_uniform = (r1 - r2) * torch.rand(batch_size, len(domain)) + r2
    return square_uniform


# n-dim generalisation of circle and sphere
def round_init_data(centre, r, batch_size):
    """
    :param centre:
    :param r:
    :param batch_size:
    :return:
    """
    dim = len(centre)
    if dim == 1:
        return segment([centre[0]-r, centre[0]+r], batch_size)
    elif dim == 2:
        return circle_init_data(centre, r, batch_size)
    elif dim == 3:
        return sphere_init_data(centre, r, batch_size)
    else:
        return n_dim_sphere_init_data(centre, r, batch_size)

def slice_nd_init_data(centre, r, batch_size):
    """
    :param centre:
    :param r:
    :param batch_size:
    :return:
    """
    dim = len(centre)
    if dim == 2:
        return slice_init_data(centre, r, batch_size)
    elif dim == 3:
        return slice_3d_init_data(centre, r, batch_size)
    else:
        raise ValueError('Positive orthant not supported for more than 3 dimensions.')



# generates data for x>0, y>0
def slice_init_data(centre, r, batch_size):
    """
    :param centre: list/tuple/tensor containing the 'n' coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    r = np.sqrt(r)
    angle = (np.pi/2) * torch.rand(batch_size, 1)
    radius = r * torch.rand(batch_size, 1)
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    offset = torch.cat([x_coord, y_coord], dim=1)

    return torch.tensor(centre) + offset


# generates data for (X - centre)**2 <= radius
def circle_init_data(centre, r, batch_size):
    """
    :param centre: list/tuple/tensor containing the 'n' coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    border_batch = int(batch_size/10)
    internal_batch = batch_size-border_batch
    r = np.sqrt(r)
    angle = (2*np.pi) * torch.rand(internal_batch, 1)
    radius = r * torch.rand(internal_batch, 1)
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    offset = torch.cat([x_coord, y_coord], dim=1)

    angle = (2 * np.pi) * torch.rand(border_batch, 1)
    x_coord = r * np.cos(angle)
    y_coord = r * np.sin(angle)
    offset_border = torch.cat([x_coord, y_coord], dim=1)
    offset = torch.cat([offset, offset_border])

    return torch.tensor(centre) + offset


# generates data for (X - centre)**2 <= radius
def sphere_init_data(centre, r, batch_size):
    """
    :param centre: list/tupe/tensor containing the 3 coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    # spherical coordinates
    # x = r sin(theta) cos(phi)
    # y = r sin(theta) sin(phi)
    # z = r cos(theta)
    theta = (2*np.pi) * torch.rand(batch_size, 1)
    phi = np.pi * torch.rand(batch_size, 1)
    radius = r * torch.rand(batch_size, 1)
    x_coord = radius * np.sin(theta) * np.cos(phi)
    y_coord = radius * np.sin(theta) * np.sin(phi)
    z_coord = radius * np.cos(theta)
    offset = torch.cat([x_coord, y_coord, z_coord], dim=1)

    return torch.tensor(centre) + offset


# generates points in a n-dim sphere: X**2 <= radius**2
# adapted from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
# method 20: Muller generalised

def n_dim_sphere_init_data(centre, radius, batch_size):

    dim = len(centre)
    u = torch.randn(batch_size, dim)  # an array of d normally distributed random variables
    norm = torch.sum(u ** 2, dim=1) ** (0.5)
    r = radius * torch.rand(batch_size, dim) ** (1.0 / dim)
    x = torch.div(r*u, norm[:, None]) + torch.tensor(centre)

    return x


# generates data for (X - centre)**2 <= radius
def slice_3d_init_data(centre, r, batch_size):
    """
    :param centre: list/tupe/tensor containing the 3 coordinates of the centre
    :param radius: int
    :param batch_size: int
    :return:
    """
    # spherical coordinates
    # x = r sin(theta) cos(phi)
    # y = r sin(theta) sin(phi)
    # z = r cos(theta)
    theta = (np.pi/2) * torch.rand(batch_size, 1)
    phi = np.pi/2 * torch.rand(batch_size, 1)
    radius = r * torch.rand(batch_size, 1)
    x_coord = radius * np.sin(theta) * np.cos(phi)
    y_coord = radius * np.sin(theta) * np.sin(phi)
    z_coord = radius * np.cos(theta)
    offset = torch.cat([x_coord, y_coord, z_coord], dim=1)

    return torch.tensor(centre) + offset


def add_corners_2d(domain):
    """
    :param domain: list = [lower_bounds, upper_bounds]
                    lower_bounds, upper_bounds are lists
    :return:
    """

    ll_corner = torch.tensor(domain[0])
    ur_corner = torch.tensor(domain[1])
    lr_corner = torch.tensor([domain[1][0], domain[0][1]])
    ul_corner = torch.tensor([domain[0][0], domain[1][1]])

    return torch.stack([ll_corner, ur_corner, lr_corner, ul_corner])


def segment(dims, batch_size):
    return (dims[1] - dims[0]) * torch.rand(batch_size, 1) + dims[0]


def remove_init_unsafe_from_d(data, initials, unsafes):
    """
    :param data:
    :param initials:
    :param unsafes:
    :return:
    """
    center_init = torch.tensor(initials[0])
    center_unsafe = torch.tensor(unsafes[0])
    new_data = []
    for idx in range(len(data)):
        # if data belongs to init or unsafe, remove it
        if torch.norm(center_init - data[idx]) > initials[1]*1.2 and \
                torch.norm(center_unsafe - data[idx]) > unsafes[1]*1.2:
            new_data.append(data[idx])

    new_data = torch.stack(new_data)

    return new_data

def inf_bounds_n(n):
    return [inf_bounds] * n

if __name__ == '__main__':
    X = n_dim_sphere_init_data(centre=(0, 0, 0, 0), radius=3, batch_size=100000)
    # print(X)
    print(f'max module: {torch.sqrt(torch.max(torch.sum(X**2, dim=1)))}')
    print(f'min module: {torch.sqrt(torch.min(torch.sum(X ** 2, dim=1)))}')


class Rectangle:

    # TODO: I want to extend this to unbounded cuboids, but not sure how to.
    def __init__(self, lb, ub):
        self.name = 'square'
        self.lower_bounds = lb
        self.upper_bounds = ub
        self.dimension = len(lb)
    
    def generate_domain(self, x, _And):
        """
        param x: data point x
        param _And: And function for verifier
        returns: formula for domain
        """
        lower = _And(*[self.lower_bounds[i] <= x[i] for i in range(self.dimension)])
        upper = _And(*[x[i] <= self.upper_bounds[i] for i in range(self.dimension)])
        return _And(lower, upper) 
        
    def generate_data(self, batch_size):
        """
        param x: data point x
        returns: data points generated in relevant domain according to shape
        """
        return square_init_data([self.lower_bounds, self.upper_bounds], batch_size)



class Sphere:
    def __init__(self, centre, radius):
        self.name = 'sphere'
        self.centre = centre
        self.radius = radius
        self.dimension = len(centre)
    
    def generate_domain(self, x, _And):
        """
        param x: data point x
        param _And: And function for verifier
        returns: formula for domain
        """
        return _And(sum([(x[i] - self.centre[i])**2 for i in range(self.dimension)]) <= self.radius ** 2)
        
    def generate_data(self, batch_size):
        """
        param batch_size: number of data points to generate
        returns: data points generated in relevant domain according to shape
        """
        return round_init_data(self.centre, self.radius, batch_size)
