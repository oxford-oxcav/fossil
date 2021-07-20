import math
import numpy as np
from scipy.spatial.distance import euclidean

# Angular calculations,
# assuming that the angular positions
# are expressed as below:
#  _____________________________
# |                             |
# |               pi/2          |
# |               _|_           |
# |              / | \          |
# |        pi -----|---- 0      |
# |              \_|_/          |
# |                |            |
# |              3*(pi/2)       |
# |                             |
# |_____________________________|
#
#
# Also assuming that the coordinate system
# is expressed as below:
#
#  ----------------------------------------
# |              positive y                |
# |                   ^                    |
# |                   |                    |
# |                   |                    |
# | negative x <------ ------> positive x  |
# |                   |                    |
# |                   |                    |
# |                   v                    |
# |              negative y                |
#  ----------------------------------------


DEGREES_0 = 0
DEGREES_90 = math.pi / 2
DEGREES_180 = math.pi
DEGREES_270 = 3 * DEGREES_90
DEGREES_360 = 2 * math.pi


def fix_angular_value(angle):
    while angle >= DEGREES_360:
        angle -= DEGREES_360
    while angle < 0:
        angle += DEGREES_360
    return angle


def atanxy(x, y):
    return math.atan2(y, x)


def angle_between_points(source_point, target_point):
    source_x, source_y = source_point
    target_x, target_y = target_point

    dx = target_x - source_x
    dy = target_y - source_y

    return fix_angular_value(atanxy(dx, dy))


def angular_steering_error(source_point, source_angle, target_point):
    target_angle = angle_between_points(source_point, target_point)
    target_angle = fix_angular_value(target_angle - source_angle)
    if target_angle > DEGREES_180:
        target_angle -= DEGREES_360
    return target_angle


def hitting_position(source_point, source_angle, *, until_x=None, until_y=None, fix_angles=True):
    if until_x is None and until_y is None:
        raise ValueError('Either until_x or until_y was needed but got none of them')
    if until_x is not None and until_y is not None:
        raise ValueError('One of until_x or until_y was needed but got both of them')

    if fix_angles:
        source_angle = fix_angular_value(source_angle)
    source_x, source_y = source_point

    if until_x is not None:
        if until_x == source_x:
            return source_point
        elif until_x < source_x:
            #  until
            #    |
            #    |
            #    |   o
            #    |
            #    |
            if (DEGREES_0 <= source_angle <= DEGREES_90) or (DEGREES_270 <= source_angle <= DEGREES_360):
                return None
            elif DEGREES_90 <= source_angle < DEGREES_180:
                # towards top left
                # |
                # |\
                # | \
                # |  \
                # |   \
                # |    \
                # |-----o
                # | dx
                alpha = DEGREES_180 - source_angle
                dx = source_x - until_x
                dy = math.tan(alpha) * dx
                return (until_x, source_y + dy)
            elif DEGREES_180 <= source_angle < DEGREES_270:
                # towards bottom left
                # |
                # | dx
                # |-----o
                # |    /
                # |   /
                # |  /
                # | /
                # |/
                alpha = source_angle - DEGREES_180
                dx = source_x - until_x
                dy = math.tan(alpha) * dx
                return until_x, source_y - dy
            else:
                assert False, "execution should not have reached here"
        elif until_x > source_x:
            #    until
            #      |
            #      |
            # o    |
            #      |
            #      |
            if DEGREES_90 <= source_angle <= DEGREES_270:
                return None
            elif 0 <= source_angle < DEGREES_90:
                # towards top right
                #      |
                #     /|
                #    / |
                #   /  |
                #  /   |
                # o----|
                #   dx |
                #      |
                alpha = source_angle
                dx = until_x - source_x
                dy = math.tan(alpha) * dx
                return until_x, source_y + dy
            elif DEGREES_270 <= source_angle < DEGREES_360:
                # towards bottom right
                #      |
                #   dx |
                # o----|
                #  \   |
                #   \  |
                #    \ |
                #     \|
                #      |
                dx = until_x - source_x
                alpha = DEGREES_360 - source_angle
                dy = math.tan(alpha) * dx
                return until_x, source_y - dy
            else:
                assert False, "execution should not have reached here"
        else:
            assert False, "execution should not have reached here"
    elif until_y is not None:
        if source_y == until_y:
            return source_x, source_y
        elif until_y > source_y:
            # ------------
            #
            #     o
            if source_angle == 0 or (DEGREES_180 <= source_angle <= DEGREES_360):
                return None
            elif 0 <= source_angle < DEGREES_90:
                #
                # ------------
                #     |  /
                #  dy | /
                #     |/
                #     o
                alpha = DEGREES_90 - source_angle
                dy = until_y - source_y
                dx = math.tan(alpha) * dy
                # print("towards top right")
                # print("dy", dy, "dx", dx, "source_angle", source_angle)  # DEBUG
                return source_x + dx, until_y
            elif DEGREES_90 <= source_angle < DEGREES_180:
                #
                # ------------
                #  \  |
                #   \ | dy
                #    \|
                #     o
                alpha = source_angle - DEGREES_90
                dy = until_y - source_y
                dx = math.tan(alpha) * dy
                # print("towards top left")
                # print("dy", dy, "dx", dx, "alpha", alpha, "source_angle", source_angle)  # DEBUG
                return source_x - dx, until_y
            else:
                assert False, "execution should not have reached here"
        elif until_y < source_y:
            #     o
            #
            # ------------
            if DEGREES_0 <= source_angle <= DEGREES_180:
                return None
            elif DEGREES_270 <= source_angle <= DEGREES_360:
                #     o
                #     |\
                #     | \
                #     |  \
                # ------------
                alpha = source_angle - DEGREES_270
                dy = source_y - until_y
                dx = math.tan(alpha) * dy
                return source_x + dx, until_y
            elif DEGREES_180 <= source_angle < DEGREES_270:
                #     o
                #    /|
                #   / |
                #  /  |
                # ------------
                alpha = DEGREES_270 - source_angle
                dy = source_y - until_y
                dx = math.tan(alpha) * dy
                return source_x - dx, until_y
            else:
                raise Exception("Execution should not have reached here")
        else:
            raise Exception("Execution should not have reached here")
    raise Exception("Execution should not have reached here")


def distance_to_wall(source_point, angle, point_lb, point_ub, fix_angles=True):
    source_x, source_y = source_point
    x1, y1 = point_lb
    x2, y2 = point_ub
    assert (x1 <= source_x <= x2 and y1 <= source_y <= y2), "The source point is not within the specified box"
    if fix_angles:
        angle = fix_angular_value(angle)

    hits = []

    for x in (x1, x2):
        hit = hitting_position(source_point, angle, until_x=x)
        if hit is not None:
            hits.append(hit)

    for y in (y1, y2):
        hit = hitting_position(source_point, angle, until_y=y)
        if hit is not None:
            hits.append(hit)

    assert len(hits) > 0, "this should not have happened"

    min_distance = float('inf')
    for hit in hits:
        dist_hit = euclidean(source_point, hit)
        if dist_hit < min_distance:
            min_distance = dist_hit

    return min_distance


def passive_transform(point, angle):
    x, y = point
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)
    Rinv = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=float)
    v = np.array([[x], [y]], dtype=float)
    result = Rinv @ v
    return -result[1, 0], result[0, 0]
