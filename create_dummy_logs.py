import time
from tensorboardX import SummaryWriter
import os

log_dir = 'dummy_logs'
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)

print(f"Writing dummy logs to {log_dir}...")

for i in range(10):
    writer.add_scalar('dummy_scalar', i * 0.1, i)
    writer.add_scalar('another_scalar', i * 0.5, i)
    time.sleep(0.1)

writer.close()

print("Finished writing dummy logs.")
