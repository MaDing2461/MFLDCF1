from .network.customnet import make_patch_xceptionnet
from .network.netutils import init_net
def create_model():
    layer = 'block2'
    netD = make_patch_xceptionnet(layer)
    return init_net(netD)