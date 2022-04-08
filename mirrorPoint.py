# Function to mirror any Point about axes and origin of a cartesian coordinate system
# x0,y0: origin coordinates for the egenralised case of a shifted coordinate system

def mirrorPoint(x,y,x0,y0,str):
    
    if str == 'X':
        return x , 2*(y0) - y
    
    elif str == 'Y':
        return 2*(x0) - x, y
    
    elif str == 'Origin':
        return 2*(x0) - x, 2*(y0) - y