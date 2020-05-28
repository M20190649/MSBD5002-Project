import pandas as pd

def split_val(data, val_proportion):
 
    # shuffle the data
    data = data.sample(frac=1)
    
    # divide routes by intersection_id and tollgate_id (6 routes)
    dfA2 = data[(data.intersection_id_A == 1) & (data.tollgate_id_2 == 1)].reset_index(drop=True)
    dfA3 = data[(data.intersection_id_A == 1) & (data.tollgate_id_3 == 1)].reset_index(drop=True)
    dfB1 = data[(data.intersection_id_B == 1) & (data.tollgate_id_1 == 1)].reset_index(drop=True)
    dfB3 = data[(data.intersection_id_B == 1) & (data.tollgate_id_3 == 1)].reset_index(drop=True)
    dfC1 = data[(data.intersection_id_C == 1) & (data.tollgate_id_1 == 1)].reset_index(drop=True)
    dfC3 = data[(data.intersection_id_C == 1) & (data.tollgate_id_3 == 1)].reset_index(drop=True)
     
    # val_proportion for training set and 0.2 for validation set
    # stratified sampling
    val_set = pd.concat([
        dfA2[:int(dfA2.shape[0]*val_proportion)], \
        dfA3[:int(dfA3.shape[0]*val_proportion)], \
        dfB1[:int(dfB1.shape[0]*val_proportion)], \
        dfB3[:int(dfB3.shape[0]*val_proportion)], \
        dfC1[:int(dfC1.shape[0]*val_proportion)], \
        dfC3[:int(dfC3.shape[0]*val_proportion)]
    ]).reset_index(drop=True)
    
    val_set = val_set.dropna().reset_index(drop=True)

    train_set = pd.concat([
        dfA2[int(dfA2.shape[0]*val_proportion):], \
        dfA3[int(dfA3.shape[0]*val_proportion):], \
        dfB1[int(dfB1.shape[0]*val_proportion):], \
        dfB3[int(dfB3.shape[0]*val_proportion):], \
        dfC1[int(dfC1.shape[0]*val_proportion):], \
        dfC3[int(dfC3.shape[0]*val_proportion):]
    ]).reset_index(drop=True)

    train_set = train_set.dropna().reset_index(drop=True)
    
    print("number of training data = %s" % train_set.shape[0])
    print("number of validation data = %s" % val_set.shape[0])
 
    X_train = train_set.iloc[:,1:]
    y_train = train_set.iloc[:,0]
    X_val = val_set.iloc[:,1:]
    y_val = val_set.iloc[:,0]
    
    return X_train, y_train, X_val, y_val
 
    
def compute_MAPE(X, y, y_pred):
    df = pd.concat([X, y, y_pred], axis=1)
    df.rename(columns={0: "pred_time"}, inplace=True)
    df["ratio"] = ((df["avg_travel_time"] - df["pred_time"]) / df["avg_travel_time"]).abs()
    groups = df.groupby(["intersection_id_A","intersection_id_B", "intersection_id_C",
                         "tollgate_id_1", "tollgate_id_2", "tollgate_id_3"]).mean()
    MAPE = groups["ratio"].sum() / groups.shape[0]

    return MAPE

def get_submission(name, pred_time):
    dataset_test = pd.read_csv("data/submission_sample/submission_sample_travelTime.csv")
    dataset_test["avg_travel_time"] = pred_time
    dataset_test.to_csv("submission/" + name, index=False)