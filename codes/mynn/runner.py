import cupy as cp
import os
from tqdm import tqdm

class RunnerM:
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, l2_reg=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.l2_reg = l2_reg

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        X, y = train_set

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

        self.best_score = 0

        for epoch in range(num_epochs):
            idx = cp.random.permutation(len(X))
            X, y = X[idx], y[idx]

            epoch_train_loss = 0.0
            epoch_train_score = 0.0
            batch_count = 0


            for iteration in range(0, len(X), self.batch_size):
                train_X = X[iteration:iteration+self.batch_size]
                train_y = y[iteration:iteration+self.batch_size]

                if len(train_X.shape) == 2:
                    train_X = train_X.reshape(-1, 1, 28, 28)
                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                if self.l2_reg:
                    trn_loss += self.l2_reg.forward(logits)
                trn_score = self.metric(logits, train_y)
                self.loss_fn.backward()
                if self.l2_reg:
                    self.l2_reg.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                epoch_train_loss += trn_loss.item()
                epoch_train_score += trn_score.item()
                batch_count += 1

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")

            self.train_loss.append(epoch_train_loss / batch_count)
            self.train_scores.append(epoch_train_score / batch_count)

            dev_score, dev_loss = self.evaluate(dev_set)
            self.dev_scores.append(dev_score)
            self.dev_loss.append(dev_loss)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"[Train] loss: {self.train_loss[-1]:.4f}, score: {self.train_scores[-1]:.4f}")
            print(f"[Dev] loss: {self.dev_loss[-1]:.4f}, score: {self.dev_scores[-1]:.4f}\n")

            if dev_score > self.best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {self.best_score:.5f} --> {dev_score:.5f}")
                self.best_score = dev_score

    def evaluate(self, data_set):
        X, y = data_set
        if len(X.shape) == 2:
            X = X.reshape(-1, 1, 28, 28)
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score.item(), loss.item()
    
    def save_model(self, save_path):
        self.model.save_model(save_path)