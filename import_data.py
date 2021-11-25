
############################

# import_data

# Function to find all required files and store their path in an array

############################


import re
import glob
import pandas as pd


# Initializing a Pandas Dataframe to store the number of the subject, type of activty
# and path of the accelerometer data

folder_location = pd.DataFrame(columns=['Subject', 'Activity', 'Path'], dtype=str)


# IMPORTANT: Change the path below to wherever the data from Smartphone3 is stored
# IMPORTANTER: Make sure the '*' at the end of the path is present

folder = glob.glob('D:/CAME/Coursework/Sem_3/CIE/Exercises/Week4_ProjectA/Global_CSVs/Data/Smartphone3/*', recursive = False)


# Initializing a variable for indexing the data
i=0


for folder in folder:
    
    
    # Important: It has been assumed that everyone has named their folders 'upstairs', 'downstairs', 
    # and 'normal' respectively. Folders with other spellings would have not been imported
    
    if ('upstairs' in folder or 'downstairs' in folder or 'normal' in folder):
        
        
        # Obtaining the path of the folder in which the loop presently is
        # Converting the path which is output in the form of a list into a Pandas DataFrame
        # Possible to write it cleaner, but stuck to this for dearth of time
        
        path = pd.Series(glob.glob(folder))
        
        
        # Splitting 'path_pd' Dataframe entry located on the first column
        # to obtain strings to label the data
        
        split_data = re.split(r'_',path.iloc[0])
        
                
        # Picking relevant characters from the strings to label the data
        
        subject = pd.Series(split_data[3][-3:])
        activity = pd.Series(split_data[4][:-2])
        
        
        # Writing the data into the DataFrame
        
        folder_location.loc[i,'Subject'] = subject.iloc[0]
        folder_location.loc[i,'Activity'] = activity.iloc[0]
        folder_location.loc[i,'Path'] = path.iloc[0]
        
        
        i+=1

        
print(folder_location.iloc[55,2])


folder_location.to_csv('folder_location.csv', sep=',')