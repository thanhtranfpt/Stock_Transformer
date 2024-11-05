import lightning as L
# import pytorch_lightning as pl
import subprocess


# Tạo Custom Callback để Log GPU Usage
class GPUUsageLogger(L.Callback):
    def __init__(self, log_interval=1):
        self.log_interval = log_interval  # Log sau mỗi vài steps

    def log_gpu_usage(self, trainer, pl_module):
        # Chạy lệnh `nvidia-smi` để lấy thông tin về GPU
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE
        )
        gpu_usage = result.stdout.decode('utf-8').strip().split('\n')

        for i, usage in enumerate(gpu_usage):
            memory_used, memory_total, utilization = usage.split(',')
            memory_used = float(memory_used)
            memory_total = float(memory_total)
            utilization = float(utilization)

            # Log GPU usage vào TensorBoard
            trainer.logger.experiment.add_scalar(f"GPU_Usage/GPU_{i}/Memory_Used_MB", memory_used, trainer.global_step)
            trainer.logger.experiment.add_scalar(f"GPU_Usage/GPU_{i}/Memory_Total_MB", memory_total, trainer.global_step)
            trainer.logger.experiment.add_scalar(f"GPU_Usage/GPU_{i}/Utilization_Percent", utilization, trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log GPU usage sau mỗi vài steps
        if (batch_idx + 1) % self.log_interval == 0:
            self.log_gpu_usage(trainer, pl_module)