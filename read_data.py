import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import owncloud
from datetime import datetime
import numpy as np
from scipy import signal
import re
 
public_link = 'https://rwth-aachen.sciebo.de/s/Q114EFBAp1QP3fq'
folder_password = 'CIE_B'
oc = owncloud.Client.from_public_link(public_link, folder_password=folder_password)
epot_id = ['255']
files = oc.list('/EPOT_Data/', depth = 'infinity')
print(files)