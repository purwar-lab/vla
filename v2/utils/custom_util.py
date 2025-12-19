import math

def get_distance_between_two_points(p1, p2):
    return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2) + ((p2[2] - p1[2]) ** 2))

def convert_reflex_angle_to_negative_angle(theta_in_radians):
    sign = 1
    if theta_in_radians < 0:
        sign = -1
    theta_in_radians = abs(theta_in_radians)
    if theta_in_radians > math.pi:
        return sign * (theta_in_radians - (2 * math.pi))
    else:
        return sign * theta_in_radians