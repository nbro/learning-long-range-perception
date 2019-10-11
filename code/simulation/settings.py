# The coordinates to predict the output of the long-range sensor based on the output of the short-range sensor at a
# different time step.
target_coordinates = [(i + 100, j) for i in range(-400, 400 + 1, 50) for j in range(-400, 400 + 1, 50)]
