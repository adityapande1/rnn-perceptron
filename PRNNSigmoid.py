import numpy as np
from PRNN_utils import tags2sentence, check_conditions

class PRNNSigmoid():

    def __init__(self, seed=15):
        np.random.seed(seed)      # Set the seed
        self.params = np.random.normal(0, 1, size=10)
        self.w = np.random.normal(0, 1, size=1)
    
    def step(self, x, threshold=0.5):
        out = (x>threshold).astype('float')
        return out  

    def sigmoid(self, x, max_val=10):
        
        x = np.clip(x, -max_val, max_val)
        out = 1 / (1 + np.exp(-x))
        return out
    
    def sigmoid_dash(self, x):
        sig_x = self.sigmoid(x)
        out = sig_x*(1-sig_x)
        return out


    def forward(self, x, h):
        '''
        Process x(t) and h(t-1) ie Single pass of RNN
        Parameters:
            x:np.array = [-upper_input_1-hot- -1 -lower_input_1-hot-]
            h:float {0,1}
        Returns:
            out(float): Sigmoid(params_trans x(t) + w h(t-1))
        '''
        c_t = np.dot(self.params, x) + self.w*h     # p_trans x(t) + w h(t-1)
        h_t = self.sigmoid(c_t)

        return c_t, h_t
    
    def process_seq(self, sequence, h_0=0.0):
        """
        Process the whole sequence
        Parameters:
            sequence List[List] 
        Returns:
            list of hidden states
        """
        
        hidden_states = [h_0]
        C_states = [0]  #Assume c(0) is 0
        
        h_tminus1 = h_0
        # Sequentially process the 
        for x_t in sequence:

            c_t, h_t = self.forward(x=x_t, h=h_tminus1)
            hidden_states.append(h_t[0])            # Just extract the numerical value
            C_states.append(c_t[0])
            
            h_tminus1 = h_t
        
        C = np.array(C_states).reshape(-1)
        H = np.array(hidden_states).reshape(-1)
        return C, H
    
    def predict_tags(self, sequence):
        ''''
        Predict Tags {0,1} using step function
        The op is [[y_cap(1), y_cap(2), .... y_cap(T)]]
        Each y_cap(i) is either 0 or 1
        '''

        C, H = self.process_seq(sequence)
        out = self.step(H).reshape(-1)[1:]
        return out
    

    def process_batch(self, batch):
        """
        Processes a batch of sequences throught the model(rnn)
        Parameters:
            batch (dtaframe) : containint the field <pos_tags>
        Oututput
            outputs list[numpy_array] : Output of each sequence through RNN, hidden state

        """

        H_outputs = []
        C_outputs = []
        for _, row in batch.iterrows():

            x = tags2sentence(row.pos_tags)
            C, H = self.process_seq(x)
            H  = H.reshape(-1)
            C  = C.reshape(-1)

            C_outputs.append(C)
            H_outputs.append(H)

        return C_outputs, H_outputs
    


    def view_params(self):
        '''
        prints perceptron parameters along with names
        '''
        print("PERCEPTRON PARAMETERS")

        print(f"Vcap : {self.params[0]}" , end = ' | ')
        print(f"Vnn : {self.params[1]}" , end = ' | ')
        print(f"Vdt : {self.params[2]}" , end = ' | ')
        print(f"Vjj : {self.params[3]}" , end = ' | ')
        print(f"Vot : {self.params[4]}" )
        print(f"T [Theta] : {self.params[5]}" , end = ' | ')
        print(f"Wnn : {self.params[6]}" , end = ' | ')
        print(f"Wdt : {self.params[7]}" , end = ' | ')
        print(f"Wjj : {self.params[8]}" , end = ' | ')
        print(f"Wot : {self.params[9]}")
        
        print(f"W : {self.w[0]}")



    def set_perfect_params(self):
        '''
        Params are of the form
        params = [Vcap, Vnn, Vdt, Vjj, Vot, [T]Theta, Wnn, Wdt, Wjj, Wot]
        '''
        print("RESETTING TO PERFECT PARAMETERS \n")
        self.params = np.array([1.5, .3, .1, .2, 2.5, 1.2, .3, 1.3, .2, 2.0])
        self.w[0] = 0.1
        self.view_params()



    def gradient_descent_step(self, grad_p, grad_w, lr=0.05):
        '''
        Updates the self. parama and self.w according to the fradient descent rule
        Parameters:
            grad_p : numpy array (10,)
            grad_w : sigle float
        '''
        self.params = self.params - lr*grad_p
        self.w = self.w - lr*grad_w



    def batch_CE_loss(self, batch):
        """
        Processes a batch of sequences and calculates the ReLU loss
        Parameters:
            batch (dtaframe) : containint the field <pos_tags> and <chunk_tags>
        Oututput
            Total Loss Relu 
        """

        total_loss = 0
        for _, row in batch.iterrows():

            sent_pos_tags = row.pos_tags
            X = tags2sentence(sent_pos_tags)
            
            _, H = self.process_seq(X)
            H = H[1:]         # Exclude h(0)

            sent_tags = row.chunk_tags  
            Y = np.array(sent_tags)
            
            loss = -(Y*np.log(H) + (1-Y)*np.log(1-H))
            loss = loss.mean()

            total_loss += loss

        total_loss = total_loss/len(batch)

        return total_loss



    def batch_accuracy(self, batch):

        correct = 0 # Predictions that match
        total = 0   # Total predictions

        for _, row in batch.iterrows():

            sent_pos_tags = row.pos_tags
            x = tags2sentence(sent_pos_tags)

            sent_tags = row.chunk_tags  
            y_target = np.array(sent_tags)

            y_pred = self.predict_tags(x) 
            correct += np.sum(y_pred == y_target)
            total += len(y_pred)
            
        acc = (correct/total)*100       # Accuracy in percentage

        return acc
    

    def batch_sentence_accuracy(self, batch):
        
        match = 0
        
        for _, row in batch.iterrows():

            sent_pos_tags = row.pos_tags
            x = tags2sentence(sent_pos_tags)

            sent_tags = row.chunk_tags  
            y_target = np.array(sent_tags)

            y_pred = self.predict_tags(x) 
            if np.array_equal(y_pred, y_target):
                match +=1

        sent_acc = (match/len(batch))*100

        return sent_acc


    def set_parameter(self, Vcap=1.5, Vnn=.3, Vdt=.1, Vjj=.2, Vot=2.5, T=1.2, Wnn=.3, Wdt=1.3, Wjj=.2, Wot=2.0, W=.10):

        self.params[0] = Vcap
        self.params[1] = Vnn
        self.params[2] = Vdt
        self.params[3] = Vjj
        self.params[4] = Vot
        self.params[5] = T
        self.params[6] = Wnn
        self.params[7] = Wdt
        self.params[8] = Wjj
        self.params[9] = Wot

        self.w[0] = W

    def does_RNN_satisfy_conditions(self):
        """
        Checks whether the RNN satisfies the inequality conditions
        """
        check_conditions(Vcap = np.round(self.params[0],4), 
                         Vnn = np.round(self.params[1],4), 
                         Vdt = np.round(self.params[2],4), 
                         Vjj = np.round(self.params[3],4), 
                         Vot = np.round(self.params[4],4),
                         T = np.round(self.params[5],4), 
                         Wnn = np.round(self.params[6],4), 
                         Wdt = np.round(self.params[7],4), 
                         Wjj = np.round(self.params[8],4), 
                         Wot = np.round(self.params[9],4), 
                         W = np.round(self.w[0],4), verbose=True)
























    






    

