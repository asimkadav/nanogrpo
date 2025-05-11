"""Utility for logging training metrics to CSV."""
import os
import csv

class CSVLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['step', 'loss', 'mean_r', 'kl', 'entropy', 'token_adv'])
    
    def write(self, step, loss, r, kl, ent, adv):
        self.writer.writerow([step, loss, r, kl, ent, adv])
        self.f.flush()
    
    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close() 