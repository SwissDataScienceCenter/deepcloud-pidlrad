import importlib


def get_model(x3d_mean, x3d_std, x2d_mean, x2d_std, args):
    model_classes = {
        "mlp": "mlp.MlpIg",
        "mlp_sw": "mlp.MlpSharedWeights",
        "unet": "unet.UnetIg",
        "lstm": "lstm.LstmIg",
        "lstm_sw": "lstm.LstmIgSharedWeights",
        "vit": "vit.ViT",
    }

    model_class_path = model_classes.get(args.model)

    if model_class_path is None:
        raise NotImplementedError(f"Model {args.model} not implemented")

    module_name, class_name = model_class_path.rsplit(".", 1)

    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
    except ModuleNotFoundError:
        raise ImportError(f"Could not import module {module_name}")

    try:
        model_class = getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Module '{module_name}' has no class '{class_name}'")

    return model_class(x3d_mean, x3d_std, x2d_mean, x2d_std, args)
