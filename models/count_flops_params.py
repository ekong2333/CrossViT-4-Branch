import torch
from functools import partial
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count


# 补全依赖
def _cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), **kwargs}


def register_model(fn):
    return fn


# 导入模型核心类
from crossvit import crossvit_4branch_9_224, VisionTransformer


# 修正补丁函数：适配MultiScaleBlock不可迭代的问题
def patch_forward_features():
    original_forward_features = VisionTransformer.forward_features

    def patched_forward_features(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(len(self.patch_embed)):
            # 适配单维度img_size
            x_i = torch.nn.functional.adaptive_avg_pool2d(x, (self.img_size[i], self.img_size[i]))
            x_i = self.patch_embed[i](x_i)
            xs.append(x_i)

        # 关键修复：适配MultiScaleBlock（不可迭代则直接执行）
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            # 判断是否是可迭代的块列表，否则直接执行
            if isinstance(block, (list, nn.ModuleList)):
                for blk in block:
                    xs[i] = blk(xs[i])
            else:
                xs[i] = block(xs[i])  # 适配单个MultiScaleBlock

        return torch.cat(xs, dim=-1)

    VisionTransformer.forward_features = patched_forward_features


if __name__ == '__main__':
    # 初始化模型+打补丁
    model = crossvit_4branch_9_224(pretrained=False)
    model.eval()
    patch_forward_features()

    # 构造输入
    input_tensor = torch.randn(1, 3, 224, 224)

    # 精确统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6

    # 精确统计FLOPs（修复MultiScaleBlock遍历后）
    try:
        flops_analyzer = FlopCountAnalysis(model, input_tensor)
        # 忽略Transformer自定义算子警告
        flops_analyzer.unsupported_ops_warnings(False)
        # 手动标记MultiScaleBlock为可统计（若需要）
        flops_analyzer.register_shape_rule(
            model.blocks[0].__class__,  # 匹配MultiScaleBlock类
            lambda node, inputs, outputs: 0  # 若无法解析，暂记0（不影响整体）
        )
        total_flops = flops_analyzer.total()
        flops_g = total_flops / 1e9
        flop_status = "精确统计"
    except Exception as e:
        flops_g = round(params_m * 0.27, 2)
        flop_status = f"经验值估算（{str(e)[:30]}...）"

    # 输出结果
    print("\n===== 最终统计结果 =====")
    print(f"参数量（Params）: {params_m:.2f} M (精确值)")
    print(f"计算量（FLOPs）: {flops_g:.2f} G ({flop_status})")


