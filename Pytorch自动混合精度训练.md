## 什么是Pytorch框架的自动混合精度训练？
**自动混合精度训练**是指同时使用`torch.autocast`和`torch.amp.GradScaler`进行训练。<br>
通过`torch.autocast`对选定的区域进行自动类型转换，以提高性能同时保持准确性。前向传播时，使用低精度计算`float16`，而反向传播计算梯度时，采用`float32`。<br>
`torch.amp.GradScaler`有助于方便地执行梯度缩放。<br>
梯度缩放通过最小化梯度下溢来提高使用`float16`（在`CUDA`和`XPU`上默认为此类型）梯度的网络的收敛性。<br>
例如下面代码展示了如何使用`autocast`，自动类型转换：<br>
```python
import torch.amp.GradScaler
import torch.autocast
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        \# Updates the scale for next iteration.
        scaler.update()
```
上面代码18-20行，展示的是如何利用`autocast`实例，将已定义的模型的参数自动精度转换，然后计算loss。<br>
当退出`autocast`上下文时，将模型输出转换成`float32`，便于后续计算损失和反向传播。<br>
在代码25行，通过自动缩放损失值，然后执行反向传播；<br>
代码30行，更新缩放后的梯度，在其内部会首先将梯度取消缩放，然后更加优化原则迭代更新模型参数。
