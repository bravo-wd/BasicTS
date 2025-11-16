from torch import nn

from ..config.pdformer_config import PDFormerConfig


class PDFormer(nn.Module):
    """
    Paper: PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction
    Link: https://ojs.aaai.org/index.php/AAAI/article/view/25556
    Official Code: https://github.com/BUAABIGSCity/PDFormer
    Conference: AAAI-23
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, config: PDFormerConfig):
        super().__init__()
