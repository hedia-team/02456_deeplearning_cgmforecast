import numpy as np

# Region A
regionA_x = np.array([0, 50, 50, 170, 385, 550, 550, 430, 280, 140, 30, 0, 0])
regionA_y = np.array([0, 0, 30, 145, 300, 450, 550, 550, 380, 170, 50, 50, 0])

# Region B
regionB_x = np.array([0, 120, 120, 260, 550, 550, 260, 70, 50, 30, 0])
regionB_y = np.array([0,  0,  30, 130, 250, 550, 550, 110, 80, 60, 60])

# Region C
regionC_x = np.array([0, 250, 250, 550, 550, 125, 80, 50, 25,  0])
regionC_y = np.array([0,  0,  40, 150, 550, 550, 215, 125, 100, 100])

# Region D
regionD_x = np.array([0, 550, 550, 50,  35,  0])
regionD_y = np.array([0,  0, 550, 550, 155, 150])

# Region E - everything else in range
regionE_x = np.array([0,  0, 550, 550])
regionE_y = np.array([0, 550, 550,  0])
