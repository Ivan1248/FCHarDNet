from ptsemseg.models import get_model

def fc_hardnet_70(n_classes):
    return get_model(dict(arch='hardnet'), n_classes=n_classes)
