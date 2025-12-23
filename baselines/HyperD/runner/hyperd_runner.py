from typing import Dict, Union, Tuple
import torch
from basicts.runners import SimpleTimeSeriesForecastingRunner


class HyperDRunner(SimpleTimeSeriesForecastingRunner):
    """
    HyperD 专用 Runner。

    主要功能：
    1. 继承 SimpleTimeSeriesForecastingRunner 的数据预处理和 Forward 流程。
    2. 重写 train_iters：显式提取模型返回的 'dual_view_loss' 并加到总 Loss 中。
    3. 增加日志记录：单独记录 'train/dual_view_loss' 以便观察图结构学习情况。
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def init_training(self, cfg: Dict):
        """
        初始化训练组件。
        在此处注册额外的 Meter 来记录 dual_view_loss。
        """
        super().init_training(cfg)
        # 注册一个新仪表盘，用于记录辅助损失
        # 格式 '{:.4f}' 表示保留4位小数
        self.register_epoch_meter('train/dual_view_loss', 'train', '{:.4f}')

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """
        训练迭代过程。
        核心修改：将 dual_view_loss 加入总 loss。
        """

        # 计算当前的全局 step 数
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index

        # 1. 前向传播 (Forward)
        # 这里的 forward_return 包含了 {'prediction': ..., 'target': ..., 'dual_view_loss': ...}
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # 2. 课程学习 (Curriculum Learning) 处理
        # 如果配置了 CL，截断预测步长
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
            forward_return['target'] = forward_return['target'][:, :cl_length, :, :]

        # 3. 计算主任务 Loss (例如 MAE)
        loss = self.metric_forward(self.loss, forward_return)

        # 获取用于加权平均的权重 (valid_num)
        weight = self._get_metric_weight(forward_return['target'])

        # 4. 【核心修改】检查并累加辅助 Loss (Dual View Loss)
        if 'dual_view_loss' in forward_return:
            aux_loss = forward_return['dual_view_loss']

            # 将辅助 loss 加到总 loss 中，参与反向传播
            loss += aux_loss

            # 单独记录辅助 loss 到日志
            self.update_epoch_meter('train/dual_view_loss', aux_loss.item(), weight)

        # 5. 更新主 Loss 日志
        self.update_epoch_meter('train/loss', loss.item(), weight)

        # 6. 计算并记录其他评估指标 (RMSE, MAPE 等)
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item(), weight)

        return loss
