from ParticleSimulation.decorators import for_all_positional
import numpy as np

def test_ParticleSimulation():
    ids = np.array([0,1,2,3,4], dtype = np.uint16)
    decorate_all_pos(ids, ids)

@for_all_positional
def decorate_all_pos(objID, para1):
    assert objID == para1, "Indices confused!"

