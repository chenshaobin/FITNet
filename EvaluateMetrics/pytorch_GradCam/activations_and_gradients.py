class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for index, target_layer in enumerate(target_layers):
            if index == 0:
                self.handles.append(
                    target_layer.register_forward_hook(
                        self.save_activation))
                # Backward compitability with older pytorch versions:
                if hasattr(target_layer, 'register_full_backward_hook'):
                    self.handles.append(
                        target_layer.register_full_backward_hook(
                            self.save_gradient))
                else:
                    self.handles.append(
                        target_layer.register_backward_hook(
                            self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_forward_hook(
                        self.save_activation_1))
                # Backward compitability with older pytorch versions:
                if hasattr(target_layer, 'register_full_backward_hook'):
                    self.handles.append(
                        target_layer.register_full_backward_hook(
                            self.save_gradient_1))
                else:
                    self.handles.append(
                        target_layer.register_backward_hook(
                            self.save_gradient_1))

    def save_activation(self, module, input, output):
        activation = output
        # print('before reshape_transform activation shape:', activation.shape)
        if self.reshape_transform is not None:
            activation = self.reshape_transform[0](activation)
        # print('after reshape_transform activation shape:', activation.shape)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform[0](grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def save_activation_1(self, module, input, output):
        activation = output
        # print('before reshape_transform activation shape:', activation.shape)
        if self.reshape_transform is not None:
            activation = self.reshape_transform[1](activation)
        # print('after reshape_transform activation shape:', activation.shape)
        self.activations.append(activation.cpu().detach())

    def save_gradient_1(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform[1](grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []

        # print(f"gradients and activations:{self.model(x).shape}")
        # return self.model(x)
        # 用于ROP 3 分类的Dual SwinTransformer网络模型返回有4个变量
        print(f"gradients and activations:{self.model(x)[0].shape}")
        return self.model(x)[0]

    def release(self):
        for handle in self.handles:
            handle.remove()
