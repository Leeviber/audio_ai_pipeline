import onnx
import onnxruntime as ort

# 加载ONNX模型
model_path = "./bin/3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx"
onnx_model = onnx.load(model_path)

# 获取模型输入和输出名称
input_name = onnx_model.graph.input[0].name
output_name = onnx_model.graph.output[0].name

# 打印原始的输入和输出名称
print("Original input name:", input_name)
print("Original output name:", output_name)

# 修改输入和输出名称
new_input_name = "feats"
new_output_name = "embs"

onnx_model.graph.input[0].name = new_input_name
onnx_model.graph.output[0].name = new_output_name

# 保存修改后的模型
new_model_path = "3dspeaker_chinese_embedding.onnx"
onnx.save(onnx_model, new_model_path)

# 加载修改后的模型并验证修改是否成功
modified_onnx_model = onnx.load(new_model_path)
modified_input_name = modified_onnx_model.graph.input[0].name
modified_output_name = modified_onnx_model.graph.output[0].name

# 打印修改后的输入和输出名称
print("Modified input name:", modified_input_name)
print("Modified output name:", modified_output_name)
