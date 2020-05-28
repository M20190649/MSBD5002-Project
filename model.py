import utils
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# LightGBM Model
class LGBM:
    def __init__(self, 
                 lr=5e-2, 
                 boosting_type='gbdt', 
                 obj='regression', 
                 metric='rmse', 
                 bagging_freq=1,
                 bagging_fraction=0.8,
                 feature_fraction=0.4):
        
        self.params = {
            'learning_rate': lr,
            'boosting_type': boosting_type,
            'objective': obj,
            'metric': metric,
            'bagging_freq': bagging_freq,
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
        }
        

    def train(self, data, val_proportion=0.2, max_round=24):
        MAPE_history = []
        lowest_MAPE = 1
        best_model = None
        best_boost_round = 1
        
        # split validation set and training set
        X_train, y_train, X_val, y_val = utils.split_val(data, val_proportion)

        # convert data into LightGBM data set format
        lgbm_train = lgb.Dataset(X_train, label=y_train)

        # train lgbm model
        for n in range(1, max_round+1):
            lgb_model = lgb.train(self.params, lgbm_train, num_boost_round=n)
            y_pred = pd.DataFrame(lgb_model.predict(X_val))
            MAPE_n = utils.compute_MAPE(X_val, y_val, y_pred)
            MAPE_history.append(MAPE_n)
            
            if n%20 == 0:
                print("round: %s/%s" % (n, max_round))
            
            if MAPE_n < lowest_MAPE:
                lowest_MAPE = MAPE_n
                best_model = lgb_model
                best_boost_round = n
        
        self.MAPE_history_ = MAPE_history
        self.lowest_MAPE_ = lowest_MAPE
        self.best_model_ = best_model
        self.best_boost_round_ = best_boost_round

   
    def predict(self, data):
        
        data = data.iloc[:,1:]
        return self.best_model_.predict(data)
      

# Random Forest Regression Model
class RandomForestRegression:
    
    def __init__(self):
        pass
        
    def train(self, data, val_proportion=0.2, max_n_estimators=150):
        MAPE_history = []
        lowest_MAPE = 1
        best_n_estimators = 1
        best_model = None
        
        # split validation set and training set
        X_train, y_train, X_val, y_val = utils.split_val(data, val_proportion)
        
        for n in range(1,max_n_estimators+1):
            reg = RandomForestRegressor(n_estimators=n, random_state=0)
            reg.fit(X_train, y_train)
            y_pred = pd.DataFrame(reg.predict(X_val))
            MAPE_n = utils.compute_MAPE(X_val, y_val, y_pred)
            MAPE_history.append(MAPE_n)

            if MAPE_n < lowest_MAPE:
                lowest_MAPE = MAPE_n
                best_model = reg
                best_n_estimators = n
    
            if n%2 == 0:
                print("n_estimators: %s/%s" % (n, max_n_estimators))
        
        self.MAPE_history_ = MAPE_history
        self.lowest_MAPE_ = lowest_MAPE
        self.best_model_ = best_model
        self.best_n_estimators_ = best_n_estimators
       
    def predict(self, data):
        data = data.iloc[:,1:]
        return self.best_model_.predict(data)

# BiLinearBlock in ResNet
class BiLinearBlock(torch.nn.Module):
    
    def __init__(self, D_in, H, D_out, device):
        
        super(BiLinearBlock, self).__init__()
        
        # An input layer
        self.input = torch.nn.Sequential(
            torch.nn.Linear(D_in, H)
        ).to(device)
        
        # The ouput layer
        self.output = torch.nn.Sequential(
            torch.nn.Linear(H, D_out)
        ).to(device)
        
        # The bilinear layer
        self.bilinear = torch.nn.Sequential(
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(H),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(H),
            torch.nn.Dropout(0.25)
        ).to(device)
        
        # An shorcut in the block
        self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        x = self.input(x)
        
        # Bilinear + shortcut
        out = self.bilinear(x)
        out += self.shortcut(x)
        
        # Bilinear + shortcut
        out2 = self.bilinear(out)
        out2 += self.shortcut(out)
        
        # Output
        out3 = self.output(out2)
        return out3

# Simple ResNet
class NN:
    
    def __init__(self, D_in, H, D_out, iteration, lr, device):
 
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.iteration = iteration
        self.lr = lr
        self.device = device
    
    def evaluate_MAPE(self, X, y, y_pred):
        X = pd.DataFrame(X.cpu().numpy())
        y = pd.DataFrame(y.cpu().numpy())
        y_pred = pd.DataFrame(y_pred.cpu().detach().numpy())
        df = pd.concat([X, y, y_pred], axis=1, ignore_index=True)
        
        # The index of 121 represents the true label -> avg_travel_time
        # The index of 121 represents the predicted travel_time
        df["ratio"] = ((df[121] - df[122]) / df[121]).abs()
        
        # The index of 37, 38, 39, 40, 41, 42 represent the one-hot encoding of intersactions and tollgates
        groups = df.groupby([37, 38, 39, 40, 41, 42]).mean()
        MAPE = groups["ratio"].sum() / groups.shape[0]
        
        return MAPE

    def train(self, data, val_proportion=0.2):
        
        # Split validation
        X_train, y_train, X_val, y_val = utils.split_val(data, val_proportion)
        
        # Transfer dataframe to cuda() type
        X_train = torch.from_numpy(X_train.to_numpy().astype(np.float32)).cuda()
        y_train = torch.from_numpy(y_train.to_numpy().astype(np.float32)).cuda()
        X_val = torch.from_numpy(X_val.to_numpy().astype(np.float32)).cuda()
        y_val = torch.from_numpy(y_val.to_numpy().astype(np.float32)).cuda()

        model = BiLinearBlock(self.D_in, self.H, self.D_out, self.device)

        loss_list = []
        MAPE = []
        lowest_MAPE = 1.0
        best_model = None
    
        # Loss function
        loss_fn = torch.nn.MSELoss(reduction='mean')      

        # Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for t in range(self.iteration):
            
            if t%100 == 0:
                print("iteration: %s/%s" % (t, self.iteration))
                
            y_train_pred = model(X_train).squeeze()
            loss = loss_fn(y_train_pred, y_train)
            loss_list.append(loss.item())
            
            y_val_pred = model(X_val).squeeze()
            MAPE_t = self.evaluate_MAPE(X_val, y_val, y_val_pred)
            MAPE.append(MAPE_t)
 
            
            if MAPE_t < lowest_MAPE:
                lowest_MAPE = MAPE_t
                best_model = model
            
            # zero gradient
            optimizer.zero_grad()
        
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
        
        self.loss_ = loss_list
        self.MAPE_ = MAPE
        self.lowest_MAPE_ = lowest_MAPE
        self.best_model_ = best_model
    
    def predict(self, data):
        data = torch.from_numpy(data.iloc[:,1:].to_numpy().astype(np.float32)).cuda()
        return self.best_model_(data)

    def visualize(self):
        # Loss curve
        fig = plt.figure(figsize=(15,4))
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        plt.sca(ax1)
        plt.title("The Loss on training set")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.plot(range(self.iteration), self.loss_)
        # MAPE on validation set
        plt.sca(ax2)
        plt.title("The MAPE on validation set")
        plt.xlabel("iteration")
        plt.ylabel("MAPE")
        plt.plot(range(self.iteration), self.MAPE_)

        plt.show()


# Feedforward Neural Network
class FNN:
    
    def __init__(self, D_in, H1, H2, D_out, iteration, lr, device):
        self.D_in = D_in
        self.H1 = H1
        self.H2 = H2
        self.D_out = D_out
        self.iteration = iteration
        self.lr = lr
        self.device = device
    
    def evaluate_MAPE(self, X, y, y_pred):
        X = pd.DataFrame(X.cpu().numpy())
        y = pd.DataFrame(y.cpu().numpy())
        y_pred = pd.DataFrame(y_pred.cpu().detach().numpy())
        df = pd.concat([X, y, y_pred], axis=1, ignore_index=True)
        
        # The index of 121 represents the true label -> avg_travel_time
        # The index of 121 represents the predicted travel_time
        df["ratio"] = ((df[121] - df[122]) / df[121]).abs()
        
        # The index of 37, 38, 39, 40, 41, 42 represent the one-hot encoding of intersactions and tollgates
        groups = df.groupby([37, 38, 39, 40, 41, 42]).mean()
        MAPE = groups["ratio"].sum() / groups.shape[0]
        return MAPE

    def train(self, data, val_proportion=0.2):
        
        # Split validation
        X_train, y_train, X_val, y_val = utils.split_val(data, val_proportion)
 
        # Transfer dataframe to cuda() type
        X_train = torch.from_numpy(X_train.to_numpy().astype(np.float32)).cuda()
        y_train = torch.from_numpy(y_train.to_numpy().astype(np.float32)).cuda()
        X_val = torch.from_numpy(X_val.to_numpy().astype(np.float32)).cuda()
        y_val = torch.from_numpy(y_val.to_numpy().astype(np.float32)).cuda()

        model = torch.nn.Sequential(
                    torch.nn.Linear(self.D_in, self.H1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.H1, self.H2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.H2, self.D_out)
                ).to(self.device)

        loss_list = []
        MAPE = []
        lowest_MAPE = 1.0
        best_model = None
    
        # Loss function
        loss_fn = torch.nn.MSELoss(reduction='mean')      

        # Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for t in range(self.iteration):
            
            if t%100 == 0:
                print("iteration: %s/%s" % (t, self.iteration))
                
            y_train_pred = model(X_train).squeeze()
            loss = loss_fn(y_train_pred, y_train)
            loss_list.append(loss.item())
            
            y_val_pred = model(X_val).squeeze()
            MAPE_t = self.evaluate_MAPE(X_val, y_val, y_val_pred)
            MAPE.append(MAPE_t)
 
            
            if MAPE_t < lowest_MAPE:
                lowest_MAPE = MAPE_t
                best_model = model
            
            # zero gradient
            optimizer.zero_grad()
        
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
        
        self.loss_ = loss_list
        self.MAPE_ = MAPE
        self.lowest_MAPE_ = lowest_MAPE
        self.best_model_ = best_model
    
    def predict(self, data):
        data = torch.from_numpy(data.iloc[:,1:].to_numpy().astype(np.float32)).cuda()
        return self.best_model_(data)

    def visualize(self):
        # Loss curve
        fig = plt.figure(figsize=(15,4))
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        plt.sca(ax1)
        plt.title("The Loss on training set")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.plot(range(self.iteration), self.loss_)
        # MAPE on validation set
        plt.sca(ax2)
        plt.title("The MAPE on validation set")
        plt.xlabel("iteration")
        plt.ylabel("MAPE")
        plt.plot(range(self.iteration), self.MAPE_)

        plt.show()
