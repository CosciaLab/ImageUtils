import sys, os


# Check if at least one argument is passed (excluding the script name)
if len(sys.argv) > 1:
    path = sys.argv[1]
    image_id = sys.argv[2]
else:
    print("No variable was passed.")

# path= '/Users/rdeliza/Desktop/Multiplex/P21E14_HN46_restitch'
# path= '/Users/rdeliza/Desktop/HN_Segmentation_Tests'
# image_id = 1

# Start
print('Job begins: ' + path)

# Importing variables
print('Importing parameters')
# Dynamically execute parameters.py
with open('parameters.py') as f:
    code = compile(f.read(), 'parameters.py', 'exec')
    exec(code)

# Report number of images
n_images = df.__len__()
print('Images: ' + n_images.__str__())

# Create output folder
quantification_path = os.path.join(path, 'quantification')
if not os.path.exists(quantification_path):
    os.mkdir(quantification_path)

# Import functions
from functions import *

# Execute functions
# Start with medians subtraction
print('Starting analysis')

# df['image_input'][image_id]
image_id = int(image_id)
print(df['image_input'][image_id])
median_data(df['image_input'][image_id])

'''
n_images = range(0, n_images)
for i in n_images:
    print(df['basename'][i])
    median_data(df['image_input'][i])
'''