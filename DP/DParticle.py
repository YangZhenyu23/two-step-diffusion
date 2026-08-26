
import numpy as np
import itertools
from functools import partial
from tqdm import trange
from networks import MLP,ResMLP
import os
import json
import datetime
import sys
from set_seed import set_seed

from KS_data_3d import KSDataGenerator
from gaussian import GMMDataGenerator
from onedim import OneDimDataGenerator
from onedim_positive import OneDimPositiveDataGenerator
from twodim import TwoDimDataGenerator

import matplotlib.pyplot as plt  # 添加这一行

import time
from solver import get_metric_solver
from torch.utils.tensorboard import SummaryWriter


# ==========================================
# 1. 距离求解器工厂 (Distance Solvers)
# ==========================================

import torch

print(torch.get_default_dtype())

# ==========================================
# 3. DPNet 类 (通用 Solver 版)
# ==========================================

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from tqdm import trange
from torch.optim.lr_scheduler import StepLR

# 修改后：


class DPNet:
    def __init__(self, distance_solver_fn, params):
        layers=params["layers"]
        model_type = params["model_type"]
        lr = params["lr"]
        scheduler_config = params["scheduler_config"]
        device = params.get("device","cuda")



        self.device = device
        # 假设之前定义的 MLP 类已经引入
        if model_type=="MLP":
            self.model = MLP(layers).to(device)
        if model_type=="ResMLP":
            self.model=ResMLP(layers).to(device)
        self.distance_solver = distance_solver_fn
        self.dim = layers[0]

        # 优化器与调度器 (对应 optax.exponential_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50000, eta_min=1e-6)

        # --- 核心修改部分 ---
        self.scheduler = None
        if scheduler_config is not None:
            # 根据配置字典里的 type 来决定实例化哪种 scheduler
            if scheduler_config['type'] == 'StepLR':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config['step_size'],
                    gamma=scheduler_config['gamma']
                )


        self.loss_log = []
#         self.scheduler=optim.lr_scheduler.ReduceLROnPlateau(
#     self.optimizer,
#     mode='min',       # 如果监控的是 Loss，选 'min'；如果是 Accuracy，选 'max'
#     factor=0.5,       # 触发条件时，学习率乘以 0.1 (相当于 decay_rate)
#     patience=5000,      # 忍受多少个 epoch Loss 不下降？这里设为 10
#     verbose=True      # 打印每次降低学习率的通知
# )


    def loss_fn(self, x, y,it):
        # x: [Batch, N, Dim_in], y: [Batch, N, Dim_out]
        # PyTorch 的 Linear 层会自动处理多维输入，相当于 JAX 的 vmap(vmap(model))
        f = self.model(x)

        # 计算距离
        dist_values = self.distance_solver(f, y, device=self.device)
        return torch.sum(dist_values)

    def train(self, dataset, writer=None,params=None):
        n_iter = params.get("n_iter",1000)
        n_batch_renew=params.get("n_batch_renew",1)
        save_path =params.get("save_path",None)
        patience=params.get("patience",5000)
        self.model.train()
        data_iter = iter(dataset)
        pbar = trange(n_iter)

        best_loss = float('inf')
        patience_counter = 0

        # 预定义最佳模型保存路径
        best_model_path = f"{save_path}/best_model.pth"

        try:
            batch = next(data_iter)
        except StopIteration:
            return
        start_time = time.time()

        for it in pbar:
            if it % n_batch_renew == 0 and it > 0:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataset)
                    batch = next(data_iter)

            x, y = [torch.as_tensor(b, device=self.device) for b in batch]

            self.optimizer.zero_grad()
            loss = self.loss_fn(x, y,it)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            # self.scheduler.step()
            self.optimizer.step()

            loss_val = loss.item()
            self.scheduler.step()
            self.loss_log.append([it, loss_val])
            if writer is not None:
                writer.add_scalar('Train/Loss', loss.item(), it)

            # self.scheduler.step()

                # --- 核心改进：保存最佳模型 ---
            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0

                # 只有提供了 save_path 才会触发保存
                if save_path:
                    self.save_model(best_model_path)
            else:
                patience_counter += 1

            pbar.set_postfix({
                'Loss': f"{loss_val:.6f}",
                'Best': f"{best_loss:.6f}",
                'Patience': f"{patience_counter}/{patience}"
            })
            if patience is not None:
                if patience_counter >= patience:
                    print(f"\n[Early Stop] at iteration {it}. Best Loss: {best_loss:.6f}")
                    break

            # 定期保存普通 Checkpoint
            if (it + 1) % 1000 == 0 and save_path:
                self.save_model(f"{save_path}/{it + 1}.pth")
        end_time = time.time()
        total_duration = end_time - start_time
        avg_speed=total_duration / max(1, it + 1)
        formatted_time = str(datetime.timedelta(seconds=int(total_duration)))
        print(f"\nTraining Finished!")
        print(f"Total Time: {formatted_time} ({total_duration:.2f} seconds)")
        print(f"Average speed: {total_duration / max(1, it + 1):.4f} s/it")
        # 格式化时间为 时:分:秒

        if save_path:
            log_path = f"{save_path}/loss_log.npy"
            np.save(log_path, np.array(self.loss_log))
            print(f"Loss log saved to {log_path}")
                # 只有当 log 不为空时才画图
            if len(self.loss_log) > 0:
                plt.figure(figsize=(10, 6))  # 设置画布大小
                log_data = np.array(self.loss_log)

                # log_data[:, 0] 是迭代次数 (it)
                # log_data[:, 1] 是 Loss 值 (loss_val)
                plt.plot(log_data[:, 0], log_data[:, 1], label='Training Loss')

                plt.title("Training Loss Curve")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.grid(True, alpha=0.3)  # 以此添加网格，方便观察
                plt.legend()

                # 保存图片
                img_path = f"{save_path}/loss_curve.png"
                plt.savefig(img_path)
                print(f"Loss curve saved to {img_path}")

                # 关闭画布，防止在循环中占用过多内存
                plt.close()
        writer.close()

        return self.loss_log,total_duration,avg_speed,it+1


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class DParticleExperiment:
    AUTO_FOLDER_NAMES = {
        "gaussian": "gaussian_exp",
        "KS": "ks_exp",
        "one_dim": "one_dim_exp",
        "one_dim_positive": "one_dim_positive_exp",
        "two_dim": "two_dim_exp",
    }

    def __init__(self, custom_params=None):
        # 默认参数
        self.default_params = {
            "data_type": "one_dim",  # "gaussian", "KS", "one_dim", "one_dim_positive", "two_dim"
            "layers": [2, 512,512,2],  # 默认 dim+1 和 dim
            "model_type": "MLP",
            "batch_size": 2000,
            "type": "w2",  # 默认 max
            "lr": 0.001,
            "scheduler_config": {
                "type": "StepLR",
                "step_size": 500,
                "gamma": 0.9
            },
            "n_iter": 100000,
            "n_batch_renew": 1000,
            "patience": 5000,
            "n_projections": 1000,
            "seed": 88,
            "n_gamma_renew": 500,
            "n_dict": 1,
            "dim": 1, # 默认 dim
            "use_folder_name": False,
            "folder_name":None,
            "timestamp":True,
            "N":None,
            "repeat_idx": None,
            "data_seed": None,
            "train_seed": None
        }

        # 如果传入自定义参数，进行合并
        if custom_params is not None:
            self.update_params(custom_params)

    def update_params(self, custom_params):
        """更新参数，检查是否有不在默认参数中的新键"""
        invalid_keys = set(custom_params.keys()) - set(self.default_params.keys())
        if invalid_keys:
            raise KeyError(f"Invalid keys found in the input dictionary: {', '.join(invalid_keys)}")

        # 合并默认参数和自定义参数
        self.default_params.update(custom_params)

    def resolve_folder_name(self):
        if not self.params.get("use_folder_name", False):
            return None
        folder_name = self.params.get("folder_name")
        if folder_name is not None:
            return folder_name
        return self.AUTO_FOLDER_NAMES.get(self.params["data_type"], self.params["data_type"])

    def resolve_repeat_suffix(self):
        repeat_idx = self.params.get("repeat_idx")
        if repeat_idx is None:
            return ""
        return f"_repeat{int(repeat_idx):02d}"

    def resolve_train_seed(self):
        train_seed = self.params.get("train_seed")
        if train_seed is None:
            return self.params["seed"]
        return train_seed

    def set_up_experiment(self):
        # 使用最终的参数
        self.params = self.default_params
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.params["data_type"] == "two_dim" and self.params["dim"] != 2:
            print(f"data_type='two_dim' requires dim=2. Overriding dim from {self.params['dim']} to 2.")
            self.params["dim"] = 2
        self.dim=self.params["dim"]
        if self.params["data_type"] == "two_dim":
            layers = list(self.params["layers"])
            if layers[0] != 2 or layers[-1] != 2:
                self.params["layers"] = [2, *layers[1:-1], 2]
                print(f"Adjusted layers for two_dim data: {layers} -> {self.params['layers']}")
        if self.params["layers"] is None:
            self.params["layers"][0]=self.dim+1
            self.params["layers"][-1]=self.dim

        # Create a unique experiment folder
        exp_dir = f"./experiments/"
        self.params["folder_name"] = self.resolve_folder_name()
        if self.params["folder_name"] is not None:
            exp_dir=exp_dir+self.params["folder_name"]+"/"
        if self.params["timestamp"]:
            exp_dir=exp_dir+self.timestamp
        if self.params["N"]:
            exp_dir=exp_dir+str(self.params["N"])
        exp_dir=exp_dir+f"dim{self.params['dim']}_{self.params['type']}{self.resolve_repeat_suffix()}"


        self.params["save_path"] = exp_dir
        run_name = f"{self.timestamp}_dim{self.params['dim']}_{self.params['type']}_lr{self.params['lr']}"
        if self.params["N"] is not None:
            run_name += f"_N{self.params['N']}"
        run_name += self.resolve_repeat_suffix()
        writer = SummaryWriter(
            log_dir=f"./runs/{run_name}")
        os.makedirs(exp_dir, exist_ok=True)
        if self.params["data_type"] == "gaussian":
            self.params["data_filepath"]=f'./GMM_data/'
        if self.params["data_type"] == "one_dim":
            self.params["data_filepath"]=f'./OneDim_data/'
        if self.params["data_type"] == "one_dim_positive":
            self.params["data_filepath"]=f'./OneDimPositive_data/'
        if self.params["data_type"] == "two_dim":
            self.params["data_filepath"]=f'./TwoDim_data/'

        # Save parameters to JSON
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(self.params, f, indent=4)

        return exp_dir, writer

    def build_dataset(self):
        if self.params["data_type"] == "gaussian":
            dataset = GMMDataGenerator(mode="train", filepath=self.params["data_filepath"],
                                   batch_size=self.params["batch_size"], dim=self.params["dim"], weights=[0.5])
        elif self.params["data_type"] == "KS":
            dataset =KSDataGenerator(mode="train", filepath=self.params["data_filepath"], n_dict=self.params["n_dict"], batch_size=self.params["batch_size"])
        elif self.params["data_type"] == "one_dim":
            dataset=OneDimDataGenerator(mode="train", filepath=self.params["data_filepath"],batch_size=self.params["batch_size"],N=self.params["N"])
        elif self.params["data_type"] == "one_dim_positive":
            dataset=OneDimPositiveDataGenerator(
                mode="train",
                filepath=self.params["data_filepath"],
                batch_size=self.params["batch_size"],
                N=self.params["N"],
                repeat_idx=self.params.get("repeat_idx"),
                data_seed=self.params.get("data_seed"),
            )
        elif self.params["data_type"] == "two_dim":
            dataset=TwoDimDataGenerator(
                mode="train",
                filepath=self.params["data_filepath"],
                batch_size=self.params["batch_size"],
                N=self.params["N"],
                dim=2,
                repeat_idx=self.params.get("repeat_idx"),
                data_seed=self.params.get("data_seed"),
            )
        else:
            dataset=None

        return dataset

    def initialize_data_and_model(self, exp_dir):
        # Set the random seed
        train_seed = self.resolve_train_seed()
        set_seed(train_seed)

        # Initialize the solver and model
        metric_solver = get_metric_solver(self.params)
        model = DPNet(metric_solver, self.params)
        print(f"Initializing Data... (Log saved to {exp_dir})")
        if self.params.get("repeat_idx") is not None:
            print(
                f"Repeat setup: repeat_idx={self.params['repeat_idx']}, "
                f"data_seed={self.params.get('data_seed')}, train_seed={train_seed}"
            )

        # Dataset initialization
        dataset = self.build_dataset()

        return model, dataset



    def start_training(self, model, dataset, writer):
        # Log the parameters and start training
        print("Training config:", json.dumps(self.params, indent=2))
        print("Training...")

        # Train the model
        train_info, total_duration, avg_speed,total_step= model.train(dataset=dataset, writer=writer, params=self.params)

        # Save the results
        result_data = {
            "total_duration": total_duration,
            "avg_speed": avg_speed,
            "total_step": total_step,
        }
        exp_dir = self.params["save_path"]
        with open(os.path.join(exp_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=4)
        print("Done.")

    def run(self):
        # Set up the experiment and initialize the model and data
        exp_dir, writer = self.set_up_experiment()
        model, dataset = self.initialize_data_and_model(exp_dir)



        # Start the training process
        self.start_training(model, dataset, writer)


def main():
    # 定义dim的范围，从8到40（包括8和40）
    dim_values = [2] # 8 to 40


    # 定义参数组合，类型(type)和学习率(lr)
    types = ["w2"] #, "sliced","max"
    lrs = [0.01] #0.001, 0.001,
    n_batch_renew = [1000]  #,1,1000


    n_values = sorted((int(10 ** i) for i in np.linspace(2, 4, 30)), reverse=True)

    for dim in dim_values:
        for n_value in n_values:
            print(f"Running experiment for dim={dim}, type={types[0]}, lr={lrs[0]}")

            # 创建传入的参数字典
            custom_params = {
                "data_type": "two_dim",
                "dim": dim,  # 动态指定 dim
                "type": types[0],  # 动态指定 type
                "lr": lrs[0],  # 动态指定 lr
                "n_batch_renew": n_batch_renew[0], # 固定的 n_batch_renew
                "use_folder_name": True,
                "timestamp":True,
                "N": n_value,
                "timestamp":False,
                "batch_size": int(n_value/2)
            }

            # 创建 DParticleExperiment 实例并运行
            try:
                experiment = DParticleExperiment(custom_params)  # 创建实验实例
                experiment.run()  # 运行实验
                print("done")
            except KeyError as e:
                print(f"Error: {e}")  # 捕获无效的参数并打印错误信息
if __name__ == "__main__":

    main()
