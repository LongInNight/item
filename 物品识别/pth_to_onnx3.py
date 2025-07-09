import torch
import torch.nn as nn
from torchvision import models, transforms
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import os


def convert_pytorch_to_onnx():
    """将PyTorch模型转换为ONNX格式"""

    # 模型参数配置
    model_path = 'best_model5.pth'
    onnx_path = 'best_model5.onnx'
    num_classes = 4  # 根据你的实际类别数量调整
    class_names = ['刀', '打火机', '背景', '药品']  # 根据你的实际类别调整

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return False

    print("开始转换PyTorch模型到ONNX...")

    # 1. 重建模型结构
    def create_model(num_classes):
        model = models.resnet18(pretrained=False)  # 不需要预训练权重
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model

    # 2. 加载训练好的模型
    device = torch.device('cpu')  # 转换时使用CPU
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"✓ 成功加载模型: {model_path}")

    # 3. 创建示例输入 (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. 设置ONNX导出参数
    input_names = ['input']
    output_names = ['output']

    # 动态轴设置（可选，允许不同的batch size）
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }

    try:
        # 5. 导出为ONNX
        torch.onnx.export(
            model,  # 要导出的模型
            dummy_input,  # 示例输入
            onnx_path,  # 输出文件路径
            export_params=True,  # 存储训练好的参数权重
            opset_version=14,  # ONNX版本 (>13)
            do_constant_folding=True,  # 是否执行常量折叠优化
            input_names=input_names,  # 输入名称
            output_names=output_names,  # 输出名称
            dynamic_axes=dynamic_axes,  # 动态轴
            verbose=False  # 是否打印详细信息
        )

        print(f"✓ 成功导出ONNX模型: {onnx_path}")

    except Exception as e:
        print(f"✗ ONNX导出失败: {str(e)}")
        return False

    # 6. 验证ONNX模型
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过")

        # 打印模型信息
        print(f"\nONNX模型信息:")
        print(f"- ONNX版本: {onnx_model.opset_import[0].version}")
        print(f"- 输入形状: {onnx_model.graph.input[0].type.tensor_type.shape}")
        print(f"- 输出形状: {onnx_model.graph.output[0].type.tensor_type.shape}")

    except Exception as e:
        print(f"✗ ONNX模型验证失败: {str(e)}")
        return False

    # 7. 测试ONNX模型推理
    try:
        print("\n测试ONNX模型推理...")

        # 创建ONNX Runtime会话
        ort_session = ort.InferenceSession(onnx_path)

        # 准备测试输入
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # ONNX推理
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)

        # PyTorch推理（用于对比）
        with torch.no_grad():
            torch_output = model(torch.from_numpy(test_input))

        # 比较输出差异
        diff = np.abs(ort_outputs[0] - torch_output.numpy())
        max_diff = np.max(diff)

        print(f"✓ ONNX推理测试成功")
        print(f"- PyTorch vs ONNX 最大差异: {max_diff:.8f}")

        if max_diff < 1e-5:
            print("✓ 输出一致性验证通过")
        else:
            print("⚠ 输出存在较大差异，请检查")

    except Exception as e:
        print(f"✗ ONNX推理测试失败: {str(e)}")
        return False

    # 8. 保存模型配置信息
    config_info = {
        'model_type': 'ResNet18',
        'num_classes': num_classes,
        'class_names': class_names,
        'input_shape': [1, 3, 224, 224],
        'input_name': input_names[0],
        'output_name': output_names[0],
        'onnx_opset_version': 14,
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }

    import json
    with open('model_config5.json', 'w', encoding='utf-8') as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)

    print(f"✓ 模型配置已保存: model_config5.json")

    # 9. 获取文件大小信息
    pytorch_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB

    print(f"\n文件大小对比:")
    print(f"- PyTorch模型: {pytorch_size:.2f} MB")
    print(f"- ONNX模型: {onnx_size:.2f} MB")

    print(f"\n✅ 转换完成！")
    print(f"输出文件:")
    print(f"- ONNX模型: {onnx_path}")
    print(f"- 配置文件: model_config5.json")

    return True


def test_onnx_with_real_image(onnx_path, image_path, class_names):
    """使用真实图片测试ONNX模型"""

    if not os.path.exists(image_path):
        print(f"测试图片不存在: {image_path}")
        return

    print(f"\n使用真实图片测试ONNX模型: {image_path}")

    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载和预处理图片
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    input_numpy = input_tensor.numpy()

    # ONNX推理
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_numpy}
    ort_outputs = ort_session.run(None, ort_inputs)

    # 处理输出
    logits = ort_outputs[0][0]  # 移除batch维度
    probabilities = np.exp(logits) / np.sum(np.exp(logits))  # softmax
    predicted_class_idx = np.argmax(probabilities)

    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    print(f"预测结果: {predicted_class}")
    print(f"置信度: {confidence:.4f}")
    print("所有类别概率:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {class_name}: {prob:.4f}")


if __name__ == "__main__":
    # 检查必要的库
    try:
        import onnx
        import onnxruntime

        print("✓ 所需库已安装")
    except ImportError as e:
        print("✗ 缺少必要的库，请安装:")
        print("pip install onnx onnxruntime")
        exit(1)

    # 执行转换
    success = convert_pytorch_to_onnx()

    if success:
        print("\n" + "=" * 60)
        print("转换成功！你现在可以使用ONNX模型进行推理。")
        print("=" * 60)

        # 可选：测试真实图片（取消注释使用）
        # class_names = ['刀', '打火机', '药品']
        # test_onnx_with_real_image('best_model.onnx', 'test_image.jpg', class_names)
    else:
        print("\n转换失败，请检查错误信息。")
    #可选：测试真实图片（取消注释使用）
    class_names = ['刀', '打火机', '背景', '药品']
    test_onnx_with_real_image('best_model5.onnx', '医用药瓶_0005.jpg', class_names)