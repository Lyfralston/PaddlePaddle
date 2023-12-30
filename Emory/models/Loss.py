import paddle


class MultiTaskLoss(paddle.nn.Layer):

    def __init__(self, num_tasks):
        super(MultiTaskLoss, self).__init__()
        out_0 = paddle.create_parameter(shape=paddle.ones(shape=num_tasks).
            shape, dtype=paddle.ones(shape=num_tasks).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(
            shape=num_tasks)))
        out_0.stop_gradient = not True
        self.sigma = out_0

    def forward(self, *losses):
        losses = paddle.concat(x=[loss.unsqueeze(axis=0) for loss in losses])
        loss = 0.5 / paddle.pow(x=self.sigma, y=2) * losses
        return loss.sum() + self.sigma.log().sum()
