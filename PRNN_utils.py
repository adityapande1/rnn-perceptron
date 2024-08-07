import numpy as np
import pandas as pd


import numpy as np


def check_all_conditions(Vcap, Vdt, Vjj, Vnn, Vot, Wdt, Wjj, Wnn, Wot, W, T):
   
   """
   Checks RNN Perceptron Conditions
   """
   
   conditions =  []

   # Vcap conditions
   conditions.append(Vcap + Wdt > T)
   conditions.append(Vcap + Wjj > T)
   conditions.append(Vcap + Wnn > T)
   conditions.append(Vcap + Wot > T)

   # Vdt conditions1
   conditions.append(Vdt + Wdt > T)
   conditions.append(Vdt + Wjj < T)
   conditions.append(Vdt + Wnn < T)
   conditions.append(Vdt + Wot > T)
    # Vdt conditions2
   conditions.append(W + Vdt + Wdt > T)
   conditions.append(W + Vdt + Wjj < T)
   conditions.append(W + Vdt + Wnn < T)
   conditions.append(W + Vdt + Wot > T)

   # Vjj conditions 1
   conditions.append(Vjj + Wdt > T)
   conditions.append(Vjj + Wjj < T)
   conditions.append(Vjj + Wnn < T)
   conditions.append(Vjj + Wot > T)
   # Vjj conditions 2
   conditions.append(W + Vjj + Wdt > T)
   conditions.append(W + Vjj + Wjj < T)
   conditions.append(W + Vjj + Wnn < T)
   conditions.append(W + Vjj + Wot > T)

   #Vnn conditions1
   conditions.append(Vnn + Wdt > T)
   conditions.append(Vnn + Wjj < T)
   conditions.append(Vnn + Wnn < T)
   conditions.append(Vnn + Wot > T)
   #Vnn conditions2
   conditions.append(W + Vnn + Wdt > T)
   conditions.append(W + Vnn + Wjj < T)
   conditions.append(W + Vnn + Wnn < T)
   conditions.append(W + Vnn + Wot > T)

   conditions.append(W + Vot + Wdt > T)
   conditions.append(W + Vot + Wjj > T)
   conditions.append(W + Vot + Wnn > T)
   conditions.append(W + Vot + Wot > T)

   if np.sum(conditions) == len(conditions):
      print("ALL INEQUALITIES SATISFIED FOR THE GIVEN PARAMETERS")
      
   return conditions



def check_conditions(Vcap, Vdt, Vjj, Vnn, Vot, Wdt, Wjj, Wnn, Wot, W, T, verbose=False):
   
    """
    Checks RNN Perceptron Conditions
    """
    conditions =  []
   
    # CAP[^] conditions
    if Vcap + Wdt > T and verbose:
        string = f"Vcap [{Vcap}] + Wdt [{Wdt}] > T [{T}]\t\tSATISFIED"
        print(string)
    conditions.append(Vcap + Wdt > T)

    if Vcap + Wjj > T and verbose:
        string = f"Vcap [{Vcap}] + Wjj [{Wjj}] > T [{T}]\t\tSATISFIED"
        print(string)
    conditions.append(Vcap + Wjj > T)

    if Vcap + Wnn > T and verbose:
        string = f"Vcap [{Vcap}] + Wjj [{Wnn}] > T [{T}]\t\tSATISFIED"
        print(string)
    conditions.append(Vcap + Wnn > T)

    if Vcap + Wot > T and verbose:
        string = f"Vcap [{Vcap}] + Wjj [{Wot}] > T [{T}]\t\tSATISFIED"
        print(string)
        print()
    conditions.append(Vcap + Wot > T)



    # DT Conditions
    if W + Vdt + Wjj < T and verbose:
        string = f"W [{W}] + Vdt [{Vdt}] + Wjj [{Wjj}] < T [{T}]\tSATISFIED"
        print(string) 
    conditions.append(W + Vdt + Wjj < T)

    if W + Vdt + Wnn < T and verbose:
        string = f"W [{W}] + Vdt [{Vdt}] + Wnn [{Wnn}] < T [{T}]\tSATISFIED"
        print(string) 
        print()
    conditions.append(W + Vdt + Wnn < T)




    # JJ Consitions
    if Vjj + Wjj < T and verbose:
        string = f"Vjj [{Vjj}] + Wjj [{Wjj}] < T [{T}]\t\tSATISFIED"
        print(string) 
    conditions.append(Vjj + Wjj < T)

    if Vjj + Wnn < T and verbose:
        string = f"Vjj [{Vjj}] + Wnn [{Wnn}] < T [{T}]\t\tSATISFIED"
        print(string) 
    conditions.append(Vjj + Wnn < T)

    if W + Vjj + Wjj < T and verbose:
        string = f"W [{W}] + Vjj [{Vjj}] + Wjj [{Wjj}] < T [{T}]\tSATISFIED"
        print(string) 
    conditions.append(W + Vjj + Wjj < T)

    if W + Vjj + Wnn < T and verbose:
        string = f"W [{W}] + Vjj [{Vjj}] + Wnn [{Wnn}] < T [{T}]\tSATISFIED"
        print(string) 
        print()
    conditions.append(W + Vjj + Wnn < T)




    # NN Consitions
    if Vnn + Wot > T and verbose:
        string = f"Vnn [{Vnn}] + Wot [{Wot}] > T [{T}]\tSATISFIED"
        print(string) 
    conditions.append(Vnn + Wot > T)

    if W + Vnn + Wot > T and verbose:
        string = f"W [{W}] + Vnn [{Vnn}] + Wot [{Wot}] > T [{T}]\tSATISFIED"
        print(string)
        print() 
    conditions.append(W + Vnn + Wot > T)

    

    # OT Conditions
    if W + Vot + Wdt > T and verbose:
        string = f"W [{W}] + Vot [{Vot}] + Wdt [{Wdt}] > T [{T}]\tSATISFIED"
        print(string) 
    conditions.append(W + Vot + Wdt > T)

    if W + Vot + Wjj > T and verbose:
        string = f"W [{W}] + Vot [{Vot}] + Wjj [{Wjj}] > T [{T}]\tSATISFIED"
        print(string) 
    conditions.append(W + Vot + Wjj > T)

    if W + Vot + Wnn > T and verbose:
        string = f"W [{W}] + Vot [{Vot}] + Wnn [{Wnn}] > T [{T}]\tSATISFIED"
        print(string) 
    conditions.append(W + Vot + Wnn > T)

    if W + Vot + Wot > T and verbose:
        string = f"W [{W}] + Vot [{Vot}] + Wot [{Wot}] > T [{T}]\tSATISFIED"
        print(string) 
    conditions.append(W + Vot + Wot > T)

    check = np.sum(conditions) == len(conditions)
    if check:
        print("\nALL INEQUALITIES SATISFIED FOR THE GIVEN PARAMETERS")
    else:
        print("\nALL INEQUALITIES ARE NOT SATISFIED")
        print(conditions)
        
    return check



def pair2sample(left_index, right_index):
    """
    Takes in left index and right index to give the concatenated
    1-hot vector of both along with bias term -1 

    x = [---upper_input_1-hot--- -1 ---lower_input_1-hot---]
    The id:0 is used for ^ token 
    NN:1 DT:2 JJ:3 OT:4 for their ids
    """

    # Create an array of zeros
    array_left = np.zeros(5)
    array_right = np.zeros(5)

    # Set the value at the specified index to 1.0, and bias to -1.0
    array_left[left_index] = 1.0
    array_right[0] = -1.0
    array_right[right_index] = 1.0
    sample = np.concatenate((array_left, array_right))

    return sample



def tags2sentence(tags):

    """
    This function takes in tag and returns it in a perceptron input format
    the output is [x(1), x(2), ......., x(T)]
    where x(i) is a 10-d vector
    
    Parameters:
        tags list[int]: The POS tags of the sentence
    
    Returns:
        sentence list[list] : This is a sequence input to the model
    """

    new_tags = [0] + tags       # For the ^ start token
    sentence = []

    for idx in range(len(new_tags)-1):
        left_index = new_tags[idx]
        right_index = new_tags[idx+1]

        sentence.append(pair2sample(left_index, right_index))

    return np.array(sentence)



def calculate_grads(x, y, model):
    """
    Calculates Gradient Loss for a sequence <x(1), x(2), ...... , x(T)>
    
    """

    Y = y.copy()
    Y = np.insert(Y, 0, 0.0)  # Insert y(0) at the begining

    X = x.copy()
    # Insert a row of zeros at the top, call it x(0)
    new_row = np.zeros((1, X.shape[1]))  # Array row of all zeros
    X = np.insert(X, 0, new_row, axis=0)

    H = model.process_seq(sequence=x)
    
    dh_dp  = np.zeros_like(X)
    # Iterative calculation dh/dp 
    for idx in range(1, len(X)):
        dh_dp[idx] = X[idx] +  model.w*dh_dp[idx-1]

    # Calculation of dJdp
    dJdp = model.relu_dash((0.5 - Y)*H) * (0.5 - Y) 
    dJdp = dJdp.reshape(dJdp.shape[0],1)
    dJdp = dJdp*dh_dp
    dJdp = dJdp[1:].mean(axis=0)


    dh_dw  = np.zeros_like(H)
    # Iterative calculation dh/dp 
    for idx in range(1, len(dh_dw)):
        dh_dw[idx] = H[idx-1] +  model.w*dh_dw[idx-1]

    # Calculation of dJdw
    dJdw = model.relu_dash((0.5 - Y)*H) * (0.5 - Y) 
    dJdw = dJdw*dh_dw
    dJdw = dJdw[1:].mean()

    return dJdp, dJdw



def batch_calculate_grads(model, batch):

    derivatives_p = []
    derivatives_w = []

    for _, row in batch.iterrows():

        sent_tags = row.chunk_tags
        y = np.array(sent_tags)

        sent_pos_tags = row.pos_tags
        x = tags2sentence(sent_pos_tags)

        der_p, der_w = calculate_grads(model=model, x=x, y=y)
        derivatives_p.append(der_p)
        derivatives_w.append(der_w)

    derivatives_p = np.array(derivatives_p).mean(axis=0)
    derivatives_w = np.array(derivatives_w).mean()

    return derivatives_p, derivatives_w




def train_and_val(df_folds, val_index):
    """
    Returns train fold and val fold
    Parameters:
        df_folds :list[Dataframe]: List of dataframes
        val_index(int): {0, 1, 2, 3, .... ,N-1}
    Returns:
        train_df : train dataframe
        val_df : validation dataframe
    """

    train_df, val_df = None, None

    for idx, df in enumerate(df_folds):

        if idx == val_index:
            val_df = df.copy()
        else:
            train_df = pd.concat([train_df, df.copy()], axis=0)
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)


    return train_df, val_df


def prepare_folds(data, nof):
    """
    Creates folds from data
    Parameters:
        data :pandas Dataframe
        nof (int): no of folds
    Returns:
        df_folds: list[Dataframes]
    """
    # Shuffle the DataFrame rows
    data_shuffled = data.sample(frac=1, random_state=42).copy()

    # Calculate the number of rows per fold
    fold_size = len(data) // nof
    remainder_rows = len(data) % nof

    # Initialize an empty list to store the folds
    df_folds = []

    # Split the shuffled DataFrame into folds
    start_index = 0
    for fold_index in range(5):

        # Calculate the end index for the fold
        end_index = start_index + fold_size + (1 if fold_index < remainder_rows else 0)
        
        # Append the fold to the list of folds
        df_folds.append(data_shuffled.iloc[start_index:end_index].reset_index(drop=True))
        
        # Update the start index for the next fold
        start_index = end_index

    return df_folds

    

def process_CVresults(CVresults_dict, summarize=True):
    """
    ABOUT
        Processes CV results
    CVresults[key] = [best_val_loss, best_val_acc, best_val_sequence_acc,  best_val_epoch, best_params(array)]
    """
    folds = len(CVresults_dict)
    val_loss, val_acc, seq_acc = 0, 0, 0
    P, W= 0.0, 0.0
    val_acc_list =[]
    val_seq_acc_list =[]
    for key, val in CVresults_dict.items():
        
        val_loss += val[0]
    
        val_acc +=  val[1]
        val_acc_list.append(val[1])

        seq_acc +=  val[2]
        val_seq_acc_list.append(val[2])

        P = P + val[4][0]
        W = W + val[4][1]
        
    val_loss = val_loss/folds
    val_acc = np.round(val_acc/folds, 4)
    seq_acc = np.round(seq_acc/folds, 4)
    P = P/folds
    W = W/folds

    if summarize:
        print(f"BEST VALIDATION ACCURACY : {np.round(np.max(val_acc_list), 4)}")
        print(f"BEST VALIDATION SEQUENCE ACCURACY : {np.round(np.max(val_seq_acc_list), 4)}")
        print()
        print(f"AVERAGE VALIDATION ACCURACY : {val_acc}")
        print(f"AVERAGE VALIDATION SEQUENCE ACCURACY : {seq_acc}")


    return P, W