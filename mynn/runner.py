import numpy as np
import os
from tqdm import tqdm
from mynn.op import L2Regularization

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, early_stopping=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        # L2 regularization
        if self.model.l2_reg:
            self.l2_reg = L2Regularization(self.model)
        else:
            self.l2_reg = None

        self.early_stopping = early_stopping # EarlyStopping
            
    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
       

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if self.early_stopping is not None:
            self.early_stopping.save_dir = save_dir

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
                
                if train_X.shape[0] == 0:
                    continue
                logits = self.model(train_X)

                trn_loss = self.loss_fn(logits, train_y)

                # L2 regularization loss
                if self.l2_reg is not None:
                    trn_loss += self.l2_reg.forward()

                self.train_loss.append(trn_loss)
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)
                
                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()
                if self.l2_reg is not None:
                    self.l2_reg.backward()

                self.optimizer.step()
                # if self.scheduler is not None:
                    # self.scheduler.step()

                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    # self.train_loss.append(trn_loss)
                    # trn_score = self.metric(logits, train_y)
                    # self.train_scores.append(trn_score)
                
                    # dev_score, dev_loss = self.evaluate(dev_set)
                    # self.dev_scores.append(dev_score)
                    # self.dev_loss.append(dev_loss)
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            if self.scheduler is not None:
                    self.scheduler.step()

            if self.early_stopping:
                if self.early_stopping.check(dev_score, self.model):
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
            else:
                if dev_score > best_score:
                    save_path = os.path.join(save_dir, 'best_model.pickle')
                    self.save_model(save_path)
                    print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                    best_score = dev_score
        
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X, is_train=False) # 
        loss = self.loss_fn(logits, y)
        if self.l2_reg is not None:
            loss += self.l2_reg.forward()
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)