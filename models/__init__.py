from .SCD_MCLNet import SCDMCLNet

def build_model(settings: dict, num_classes: int=7, variant: str='full') -> SCDMCLNet:
    if not isinstance(settings, dict):
        raise TypeError('model settings must be a dict')
    return SCDMCLNet(settings=settings, num_classes=num_classes, variant=variant)
__all__ = ['SCDMCLNet', 'build_model']
