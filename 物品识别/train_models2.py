import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 数据增强（用于训练）
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    data_dir = './photo'  # 数据集路径
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # 创建数据加载器
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 获取类别名称
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f'类别: {class_names}')
    print(f'类别数量: {num_classes}')
    print(f'数据集大小: {len(dataset)}')

    # 创建模型
    def create_model(num_classes):
        # 加载预训练的ResNet18模型
        model = models.resnet18(pretrained=True)

        # 冻结特征提取层的参数
        for param in model.parameters():
            param.requires_grad = False

        # 修改最后的全连接层
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        return model

    # 创建模型实例
    model = create_model(num_classes)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 修改后的训练函数 - 只保存最佳模型
    def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=20):
        model.train()
        train_losses = []
        train_accuracies = []

        # 记录最佳准确率和对应的模型状态
        best_accuracy = 0.0
        best_model_state = None
        best_epoch = 0

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 统计
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                if batch_idx % 10 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

            # 计算平均损失和准确率
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = correct_predictions / total_samples

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

            # 检查是否是最佳准确率
            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_epoch = epoch + 1
                # 深拷贝模型状态
                best_model_state = model.state_dict().copy()
                print(f'✓ 新的最佳准确率: {best_accuracy:.4f} (Epoch {best_epoch})')

            # 更新学习率
            scheduler.step()

        return train_losses, train_accuracies, best_model_state, best_accuracy, best_epoch

    # 开始训练
    print('开始训练...')
    train_losses, train_accuracies, best_model_state, best_accuracy, best_epoch = train_model(
        model, dataloader, criterion, optimizer, scheduler, num_epochs=50
    )

    # 只保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model5.pth')
        print(f'\n最佳模型已保存为 best_model5.pth')
        print(f'最佳准确率: {best_accuracy:.4f} (来自第 {best_epoch} 轮)')
    else:
        print('未找到有效的最佳模型状态')

    # 保存模型信息到文本文件
    with open('model_info5.txt', 'w', encoding='utf-8') as f:
        f.write(f'最佳模型信息\n')
        f.write(f'=' * 30 + '\n')
        f.write(f'最佳准确率: {best_accuracy:.4f}\n')
        f.write(f'最佳轮次: {best_epoch}\n')
        f.write(f'总训练轮次: {len(train_accuracies)}\n')
        f.write(f'类别数量: {num_classes}\n')
        f.write(f'类别名称: {class_names}\n')
        f.write(f'类别映射: {dict(enumerate(class_names))}\n')
        f.write(f'数据集大小: {len(dataset)}\n')
        f.write(f'批次大小: {batch_size}\n')

    # 绘制训练曲线，标记最佳点
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, 'g-', label='Training Accuracy')
    plt.axhline(y=best_accuracy, color='r', linestyle='--', alpha=0.7, label=f'Best Acc: {best_accuracy:.4f}')
    plt.axvline(x=best_epoch - 1, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(train_losses, 'b-', alpha=0.7, label='Loss')
    plt.plot([x * max(train_losses) for x in train_accuracies], 'g-', alpha=0.7, label='Accuracy (scaled)')
    plt.axvline(x=best_epoch - 1, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    plt.title('Loss & Accuracy Overview')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 加载最佳模型进行最终测试
    print('\n加载最佳模型进行最终测试...')
    model.load_state_dict(best_model_state)

    # 创建一个简单的测试函数
    def test_model_on_dataset(model, dataloader, class_names):
        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(len(class_names)))
        class_total = list(0. for i in range(len(class_names)))

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 计算每个类别的准确率
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        overall_acc = 100 * correct / total
        print(f'\n最佳模型在训练数据上的性能:')
        print(f'整体准确率: {overall_acc:.2f}%')

        for i in range(len(class_names)):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f'{class_names[i]} 准确率: {class_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')

        return overall_acc

    # 测试最佳模型
    final_accuracy = test_model_on_dataset(model, dataloader, class_names)

    print(f'\n=' * 50)
    print(f'训练完成总结:')
    print(f'最佳模型来自第 {best_epoch} 轮训练')
    print(f'最佳准确率: {best_accuracy:.4f}')
    print(f'最终验证准确率: {final_accuracy:.2f}%')
    print(f'模型已保存为: best_model5.pth')
    print(f'训练信息已保存为: model_info5.txt')
    print(f'=' * 50)


# 加载保存的最佳模型进行预测
def load_best_model_for_inference(model_path, num_classes):
    """加载最佳模型用于推理"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def predict_single_image(model_path, image_path, class_names):
    """使用最佳模型预测单张图片"""
    # 预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载模型
    model = load_best_model_for_inference(model_path, len(class_names))

    # 加载和预处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0)

    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    confidence = probabilities[predicted.item()].item()

    return predicted_class, confidence, probabilities


if __name__ == '__main__':
    # Windows多进程支持
    import multiprocessing

    multiprocessing.freeze_support()

    # 运行主函数
    main()

    # 使用示例（取消注释来使用）
    # class_names = ['刀', '打火机', '药品']  # 根据你的实际类别调整
    # predicted_class, confidence, probs = predict_single_image('best_model.pth', 'test_image.jpg', class_names)
    # print(f'预测结果: {predicted_class}, 置信度: {confidence:.4f}')
