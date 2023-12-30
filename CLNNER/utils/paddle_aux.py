
# This file is generated by PaConvert ToolKit, please Don't edit it!
import paddle

def min_class_func(self, *args, **kwargs):
    if 'other' in kwargs:
        kwargs['y'] = kwargs.pop('other')
        ret = paddle.minimum(self, *args, **kwargs)
    elif len(args)==1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.minimum(self, *args, **kwargs)
    else:
        if 'dim' in kwargs:
            kwargs['axis'] = kwargs.pop('dim')

        if 'axis' in kwargs or len(args) >= 1:
            ret = paddle.min(self, *args, **kwargs), paddle.argmin(self, *args, **kwargs)
        else:
            ret = paddle.min(self, *args, **kwargs)

    return ret

def max_class_func(self, *args, **kwargs):
    if 'other' in kwargs:
        kwargs['y'] = kwargs.pop('other')
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args)==1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if 'dim' in kwargs:
            kwargs['axis'] = kwargs.pop('dim')

        if 'axis' in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "min", min_class_func)
setattr(paddle.Tensor, "max", max_class_func)

def add(self, *args, **kwargs):
    if 'other' in kwargs:
        y = kwargs['other']
    elif 'y' in kwargs:
        y = kwargs['y']
    else:
        y = args[0]

    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        if alpha != 1:
            if not isinstance(y, paddle.Tensor):
                y = paddle.to_tensor(alpha * y)
            else:
                y = alpha * y
    else:
        if not isinstance(y, paddle.Tensor):
            y = paddle.to_tensor(y)

    return paddle.add(self, y)

setattr(paddle.Tensor, 'add', add)
