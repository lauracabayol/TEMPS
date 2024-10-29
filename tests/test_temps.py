import numpy as np
from loguru import logger
import torch

from temps.temps_arch import EncoderPhotometry, MeasureZ
from temps.temps import TempsModule


def test():
    nn_features = EncoderPhotometry()
    nn_features.load_state_dict(
        torch.load("data/models/modelF_DA.pt", map_location=torch.device("cpu"))
    )
    nn_z = MeasureZ(num_gauss=6)
    nn_z.load_state_dict(
        torch.load("data/models/modelZ_DA.pt", map_location=torch.device("cpu"))
    )

    temps_module = TempsModule(nn_features, nn_z)

    col = np.array(
        [0.54804805, 1.81142339, 0.63354394, 0.7356338, 1.3578122, 0.90108565]
    )
    ztrue = 0.4446

    z, pz, odds = temps_module.get_pz(
        input_data=torch.Tensor(col).unsqueeze(0), return_pz=True, return_flag=True
    )

    zdiff = np.abs(z - ztrue).mean()

    logger.info(f"zdiff: {zdiff}")
    logger.info("test passed")

    assert zdiff < 0.01


test()


# %%
