
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np

"""
membrane_stress = 2 
bending_stress = 3 

"""
section_row = 2










if __name__ == "__main__":

    x_data = pd.read_csv("scaled_data_x_data.csv")
    print(x_data.head(5))


    """
    train_test_split    
    """

    pd_data_input_y = pd.read_csv("7521number_final_output_data.csv")
    # y_data_input_y = pd_data_input_y.read_excel['path1_membrane_SINT'].values.tolist()

    y_data = pd.DataFrame(dtype=np.float64, columns=['y__data'])


    for i in range(0, len(y_data['y__data'])):
        # for i in range(len(pd_data_input_y)):
        y_data.loc[i, 'y__data'] = pd_data_input_y.iloc[i, section_row]
        # print(y_data.loc[i, 'y__data'])

    # print("train_y",y_data)
    y_data.to_csv('data.csv', index=False)





    X_tr, X_val, y_tr, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

    print(X_tr.shape)
    print(y_tr.shape)
    print(X_val.shape)
    print(y_val.shape)











