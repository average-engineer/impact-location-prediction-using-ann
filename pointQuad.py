# Function for deciding the quadrant of a point 
# Based on the Simualtion and Experimental setup of the impact experiment

def pointQuad(x,y,len):
    
    # x: X coordinate of the point
    # y: Y coordinate of the point
    # len: length of the list for storing quadrant of the impact point
    
    quadrant = [] # Empty List for storing the quadrant of impact
    if x >= 250 and x <= 325 and y >= 250 and y <= 325:
        for i in range(0,len):
            quadrant.append('Q1')
            
    elif x >= 175 and x <= 250 and y >= 250 and y <= 325:
        for i in range(0,len):
            quadrant.append('Q2')
            
    elif x >= 175 and x <= 250 and y >= 175 and y <= 250:
        for i in range(0,len):
            quadrant.append('Q3')
            
    elif x >= 250 and x <= 325 and y >= 175 and y <= 250:
        for i in range(0,len):
            quadrant.append('Q4')
            
    return quadrant
            