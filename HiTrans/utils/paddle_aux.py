
# This file is generated by PaConvert ToolKit, please Don't edit it!
import paddle

def view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        key = [k for k in kwargs.keys()]
        return paddle.view(self, shape_or_dtype = kwargs[key[0]])

setattr(paddle.Tensor, 'view', view)
