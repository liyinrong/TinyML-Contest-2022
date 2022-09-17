import torch.nn as nn
import torch.quantization

class IEGMNetQ(nn.Module):
    def __init__(self, model_fp32):
        super(IEGMNetQ, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, input):
        model_input = self.quant(input)
        model_output = self.model_fp32(model_input)
        output = self.dequant(model_output)
        return output

    def fuse_layers(self):
        self.model_fp32.fuse_layers()
        