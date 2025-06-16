import torch

# helper functions
def extrapolate_pressures(x3d, x2d):
    pres_ecrad_in_idx = 2
    pres_sfc_ecrad_in_idx = 0

    pres_flvl = x3d[:, :, pres_ecrad_in_idx]

    pres_hlvl = torch.sqrt(pres_flvl[:,1:] * pres_flvl[:,:-1])

    pres_top = (pres_hlvl[:,0] / (torch.sqrt(pres_hlvl[:,0]*pres_hlvl[:,1])+1e-9))
    pres_surface =  x2d[:, pres_sfc_ecrad_in_idx]
    pres = torch.cat((pres_top[:,None], pres_hlvl, pres_surface[:, None]), dim=-1)[...,None]
    return pres


def calculate_heating_rates(y, x3d, x2d):
    # assumed output order is: [lw_up, lw_dn, sw_up, sw_dn]
    g = 9.80665
    cd = 1005
    cv = 1855
    qv_ecrad_in_idx = 5
    qv = x3d[..., qv_ecrad_in_idx, None]

    pres = extrapolate_pressures(x3d, x2d)
    
    heating_rate = ((-g / (cd * (1-qv) + cv * qv)) / \
                    ((pres[..., :-1, :] - pres[...,1:,:]) + 1e-7)) * \
                        ((y[..., :-1, [0, 2]] - y[..., :-1, [1, 3]]) - \
                        (y[..., 1:, [0, 2]] - y[..., 1:, [1, 3]])) * 24*60*60
    return heating_rate
