from typing import List, Optional

import torch

from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)
from trainer import Trainer
from feature import fourier_time_features_from_frequency,lags_for_fourier_time_features_from_frequency
from estimator import PyTorchEstimator
from utils import get_module_forward_input_names
from time_grad_network import TimeGradPredictionNetwork,TimeGradTrainingNetwork

class TimeGradEstimator(PyTorchEstimator):
    def __init__(
        self,
        input_size: int,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "LSTM",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        conditioning_length: int = 100,
        diff_steps: int = 100,
        loss_type: str = "l2",
        beta_end=0.1,
        beta_schedule="linear",
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        **kwargs,
    ) -> None:
        super().__init__(trainer=trainer, **kwargs)

        self.freq = freq # 时序数据的频率 1H or 15min
        self.context_length = (
            context_length if context_length is not None else prediction_length
        ) # context_length 默认为 prediction_length

        self.input_size = input_size # 输入数据的维度 1484?
        self.prediction_length = prediction_length # 预测长度 24 or 96
        self.target_dim = target_dim # 目标维度 序列个数 样例为370 
        self.num_layers = num_layers # 网络层数 2
        self.num_cells = num_cells # 网络单元数 40
        self.cell_type = cell_type # 网络单元类型 GRU or LSTM
        self.num_parallel_samples = num_parallel_samples # 并行采样数 100
        self.dropout_rate = dropout_rate # dropout率 0.1
        self.cardinality = cardinality  # 基数 1
        self.embedding_dimension = embedding_dimension # 嵌入维度 5

        self.conditioning_length = conditioning_length # 条件长度 100
        self.diff_steps = diff_steps   # diffusion 采样步长 100
        self.loss_type = loss_type # 损失函数类型 l2 or l1
        self.beta_end = beta_end # beta的最终值 0.1
        self.beta_schedule = beta_schedule # beta的变化方式 linear or cosine etc.
        self.residual_layers = residual_layers # 残差层数 8
        self.residual_channels = residual_channels # 残差通道数 8
        self.dilation_cycle_length = dilation_cycle_length # 膨胀周期长度 2

        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=freq) # freq_str: 时序数据的频率 1H
        ) # 时序特征的滞后序列 [1,24,168]

        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        ) # FourierTimeFeatures(freq_str=freq), # 时序特征

        self.history_length = self.context_length + max(self.lags_seq) # 历史长度 context_length + max(lags_seq) 24 + 168 = 192
        self.pick_incomplete = pick_incomplete # 是否选择不完整的数据 False
        self.scaling = scaling # 是否进行标准化 True

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )# 对训练数据进行采样，平均每个batch中的样本数为1.0，最小历史长度为history_length，最小未来长度为prediction_length

        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length, 
        ) # 对验证数据进行采样，最小历史长度为history_length，最小未来长度为prediction_length
        

    def create_transformation(self) -> Transformation:
        """时序数据的转换

        Returns:
            Transformation: 转换链
        """
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ), # 将目标数据转换为numpy数组
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ), # 将目标数据的维度扩展为(1, T)
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),# 添加观测值指示器
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),# 添加时间特征
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),# 将时间特征垂直堆叠
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),# 设置静态特征
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ), # 目标维度指示器
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1), # 将静态特征转换为numpy数组,维度为1
            ]
        ) # 转换链

    def create_instance_splitter(self, mode: str)-> InstanceSplitter:
        """创建实例分割器

        Args:
            mode (str): 模式

        Returns:
            InstanceSplitter: 实例分割器
        """
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode] # 通过当前数据集选择不同的采样器

        return InstanceSplitter(
            target_field=FieldName.TARGET, # 目标字段
            is_pad_field=FieldName.IS_PAD, # 是否填充字段
            start_field=FieldName.START, # 开始字段
            forecast_start_field=FieldName.FORECAST_START, # 预测开始字段
            instance_sampler=instance_sampler, # 采样器
            past_length=self.history_length, # 历史长度
            future_length=self.prediction_length, # 未来长度
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ], # 时间序列字段
        ) + (
            RenameFields( # 重命名字段
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )# 重命名目标字段 重写了__add__方法
        )# 实例分割器

    def create_training_network(self, device: torch.device) -> TimeGradTrainingNetwork:
        return TimeGradTrainingNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            diff_steps=self.diff_steps,
            loss_type=self.loss_type,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            conditioning_length=self.conditioning_length,
        ).to(device)
    
    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: TimeGradTrainingNetwork,
        device: torch.device,
    ) -> Predictor:
        prediction_network = TimeGradPredictionNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            diff_steps=self.diff_steps,
            loss_type=self.loss_type,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            conditioning_length=self.conditioning_length,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device)

        copy_parameters(trained_network, prediction_network) # 复制参数
        input_names = get_module_forward_input_names(prediction_network) # 获取模型的输入名称
        prediction_splitter = self.create_instance_splitter("test") # 创建测试集实例分割器 

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter, # 输入转换
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )
