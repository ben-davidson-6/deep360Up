import tensorflow as tf

# this script has functions for output constraint
# cartesian coordinate has 4 output constraints (no constraint, clipping, sigmoid and normalization)
# spherical angles has 5 output constraints (no constraint, clipping, sigmoid, modulo, s2c c2s)
# please refer to our IEEE VR paper and TVCG paper

# clipping
def sph_clipping(x):
    [phi,theta] = tf.split(x, [1,1], 1)
    # compute sigmoid
    phi = tf.clip_by_value(phi,0,180)
    theta = tf.clip_by_value(theta,0,360)
    # multiply by 180 and 360
    output = tf.concat([phi,theta],1)
    return output

# sigmoid bounding
def sph_sigmoid_bounding(x):
    [phi,theta] = tf.split(x, [1,1], 1)
    # compute sigmoid
    phi = tf.sigmoid(phi)
    theta = tf.sigmoid(theta)
    # multiply by 180 and 360
    phi = tf.multiply(phi,180)
    theta = tf.multiply(theta,360)
    output = tf.concat([phi,theta],1)
    return output

# modulo
def sph_modulo(x):
    [phi_batch, theta_batch] = tf.split(x, [1, 1], 1)
    phi_batch = tf.floormod(phi_batch,360)
    theta_batch = tf.where(phi_batch > 180, theta_batch - 180, theta_batch)
    phi_batch = tf.where(phi_batch>180, 360-phi_batch, phi_batch)
    theta_batch = tf.floormod(theta_batch,360)
    output = tf.concat([phi_batch, theta_batch], 1)
    return  output

# s2c,c2s
def sph_s2c_c2s(sph):
    import math
    clamp_low = -0.999999
    clamp_high = 0.999999
    def spherical_2_cartesian(inputs):

        # Change it into Radians
        phi = inputs[0] * math.pi / float(180)
        theta = inputs[1] * math.pi / float(180)
        rho = 1

        # Change them into x, y, z cartesian coordinates
        # x = tf.clip_by_value(rho * tf.sin(phi) * tf.cos(theta), -0.999999999998, 0.9999999999998)
        # y = tf.clip_by_value(rho * tf.sin(phi) * tf.sin(theta), -0.999999999998, 0.9999999999998)
        # z = tf.clip_by_value(rho * tf.cos(phi), -0.999999999998, 0.9999999999998)
        x = tf.sin(phi) * tf.cos(theta)
        y = tf.sin(phi) * tf.sin(theta)
        z = tf.cos(phi)
        # Get them into
        cartesian = tf.stack([x, y, z])

        return cartesian

    def cartesian_2_spherical(inputs):
        # Input:
        #  x,y,z
        # Output:
        #  x,y,z (cartesian) that corresponds to rho,phi,theta
        # Goal:
        # convert spherical coordinates to cartesian coordinate
        # About convention :
        #  phi = arccos(z/r) , theta = arctan(y/x) range of phi [0,pi], range of theta [0,2pi]
        x = inputs[0]
        y = inputs[1]
        z = inputs[2]

        # x_2 = tf.pow(x, 2)
        # y_2 = tf.pow(y, 2)
        # z_2 = tf.pow(z, 2)
        #
        # epsilon = 1e-7
        theta = tf.atan2(y, x)
        # atan2 returns value of which range is [-pi,pi], range of theta is [0,2pi] so if theta is negative value,actual value is theta+2pi
        # if theta < 0:
        #     theta = theta + 2 * math.pi

        theta = tf.cond(theta < 0, lambda: tf.add(theta, 2 * math.pi), lambda: tf.add(theta, 0))

        # theta = theta % (2* PI) # potential ERROR : phi [0,pi] theta [0,2pi] but atan2 returns value [-pi,pi]

        # rho = x_2 + y_2 + z_2
        # rho = tf.sqrt(rho)

        z = tf.clip_by_value(z, clamp_low, clamp_high)
        phi = tf.acos(z)

        phi = phi * 180 / math.pi
        theta = theta * 180 / math.pi
        spherical = tf.stack([phi, theta])
        return spherical

    cart = tf.map_fn(spherical_2_cartesian, sph)
    # cosTheta = K.dot(y_true,y_pred)
    valid_sph = tf.map_fn(cartesian_2_spherical,cart)
    return valid_sph

# cartesian 3 constraints
# clipping
def cart_clipping(x):
    [x, y, z] = tf.split(x, [1, 1, 1], 1)
    x = tf.clip_by_value(x, -1, 1)
    y = tf.clip_by_value(y,-1,1)
    z = tf.clip_by_value(z,-1,1)
    output = tf.concat([x, y, z], 1)
    return output
# sigmoid
def cart_sigmoid_bounding(x):
    [x,y,z] = tf.split(x, [1,1,1], 1)
    # compute sigmoid
    x = tf.sigmoid(x)
    y = tf.sigmoid(y)
    z = tf.sigmoid(z)
    # multiply by 180 and 360
    x = tf.multiply(x,2)-1
    y = tf.multiply(y,2)-1
    z = tf.multiply(z,2)-1
    output = tf.concat([x,y,z],1)
    return output
# normalization
def cart_normalize(x):
    x = x/tf.norm(x)
    return x

