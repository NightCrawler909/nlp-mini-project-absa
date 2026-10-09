import time
import psutil
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logging.info(f"{self.name} took {self.interval:.4f} seconds.")

def measure_inference_time(model_predict_fn, inputs, num_runs=10):
    start = time.time()
    for _ in range(num_runs):
        model_predict_fn(inputs)
    avg_time = (time.time() - start) / (len(inputs) * num_runs)
    return avg_time

def count_parameters(model):
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except:
        return 0 # For sklearn models
