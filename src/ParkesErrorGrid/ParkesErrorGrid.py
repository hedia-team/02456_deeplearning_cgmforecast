# %%
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import path

#from shapely.geometry import Point
#from shapely.geometry.polygon import Polygon

# %%

# measurements from the reference instrument
#input_x = np.array([200, 200, 500, 200, 450, 75, 500, 25, 600, 50, 750, -50, 50, -50])

# measurements from the 'new' instrument
#input_y = np.array([200, 300, 300, 500, 150, 500, 100, 400, 50, 600, 750, 100, -50, -50])


def in_polygon(x_array: np.array, y_array: np.array, polygon_x: np.array, polygon_y: np.array):
    '''
    Checks if points are located in polygon. Returns boolean vector
    '''
    polygon_points = [(x, y) for (x, y) in zip(polygon_x, polygon_y)]
    points = np.hstack([x_array.reshape(-1, 1), y_array.reshape(-1, 1)])
    p = path.Path(polygon_points)
    return p.contains_points(points)


def parke_error_grid(ref_values: np.array,
                     pred_values: np.array,
                     title_string: str,
                     unit: str) -> Tuple[plt.plot, dict, dict]:
    '''
    Computes the number of points and percent in each zone of the parkes error grid and visualizes it. 
    '''

    # Could be nice to move to its own file
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

    if unit.lower() == 'mg/dl':
        div = 1

    elif unit.lower() == 'mmol/l':
        div = 18

    else:
        raise Exception('Wrong BGL unit!')

    # Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))

    plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

    plt.scatter(np.array(ref_values), np.array(pred_values), marker='o', color='black', s=8, alpha=0.2)

    plt.title("Parkes Error Grid - Model: " + title_string)
    plt.xlabel("Reference Concentration (" + unit + ")")
    plt.ylabel("Prediction Concentration (" + unit + ")")
    ticks = np.arange(0, int(400/div) + int(np.floor(50/div)), int(np.floor(50/div)))
    labels = [str(ticks[i]) for i in range(len(ticks))]
    plt.xticks(np.array(ticks) * div, labels=labels)
    plt.yticks(np.array(ticks) * div, labels=labels)
    plt.gca().set_facecolor('white')

    # Set axes lengths
    plt.gca().set_xlim([0, 400/div])
    plt.gca().set_ylim([0, 400/div])
    plt.gca().set_aspect((400)/(400))

    # Plot zone lines
    plt.plot(regionA_x, regionA_y, '-', c='black')
    plt.plot(regionB_x, regionB_y, '-', c='black')
    plt.plot(regionC_x, regionC_y, '-', c='black')
    plt.plot(regionD_x, regionD_y, '-', c='black')
    plt.plot(regionE_x, regionE_y, '-', c='black')

    # Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 230, "B", fontsize=15)
    plt.text(220, 350, "B", fontsize=15)
    plt.text(130, 350, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(75, 350, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(20, 275, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

    plt.plot([0, 500], [0, 500], ':', c='black')

    # compute statistics
    parke = {}
    parke['A'] = in_polygon(ref_values, pred_values, regionA_x, regionA_y)
    parke['B'] = in_polygon(ref_values, pred_values, regionB_x, regionB_y)
    parke['C'] = in_polygon(ref_values, pred_values, regionC_x, regionC_y)
    parke['D'] = in_polygon(ref_values, pred_values, regionD_x, regionD_y)
    parke['E'] = in_polygon(ref_values, pred_values, regionE_x, regionE_y)

    parke['E'] = parke['E'] & ~parke['D']
    parke['D'] = parke['D'] & ~parke['C']
    parke['C'] = parke['C'] & ~parke['B']
    parke['B'] = parke['B'] & ~parke['A']

    for zone in 'ABCDE':
        parke[zone] = parke[zone].sum()

    parke_prob = {}
    for zone in 'ABCDE':
        parke_prob[zone] = parke[zone] / len(ref_values)

    return plt, parke, parke_prob


# %%
#parke_error_grid(input_x, input_y, '', 'mg/dl')

# %%
