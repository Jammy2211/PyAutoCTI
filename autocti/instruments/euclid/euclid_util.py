def roe_corner_from(ccd_id, quadrant_id):

    row_index = ccd_id[-1]

    if (row_index in "123") and (quadrant_id == "E"):
        return (1, 0)
    elif (row_index in "123") and (quadrant_id == "F"):
        return (1, 1)
    elif (row_index in "123") and (quadrant_id == "G"):
        return (0, 1)
    elif (row_index in "123") and (quadrant_id == "H"):
        return (0, 0)
    elif (row_index in "456") and (quadrant_id == "E"):
        return (0, 1)
    elif (row_index in "456") and (quadrant_id == "F"):
        return (0, 0)
    elif (row_index in "456") and (quadrant_id == "G"):
        return (1, 0)
    elif (row_index in "456") and (quadrant_id == "H"):
        return (1, 1)
