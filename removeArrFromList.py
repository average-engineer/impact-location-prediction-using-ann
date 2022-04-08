# Function for removing an array from an list of arrays
# Input -> Base List (list of arrays from which the array has to be removed)
#       -> Target Array (Array which has to be removed)
# The target array is first searched and compared with all element arrays of the base array
# Once a match is there, the index is stored and the array at that index is removed from the base array

def removeArrFromList(baseList,targetArr):
    
    # Looping through the elements of the base list
    for i in range(0,len(baseList)):
         # Checking for the match of the target array in the list
         if (targetArr == baseList[i]).all():
             baseList.pop(i) # Removing the array at that particular index
             break
         
    return baseList
