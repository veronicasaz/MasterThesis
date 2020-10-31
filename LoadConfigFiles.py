import yaml
import numpy as np

import AstroLibraries.AstroLib_Basic as AL_BF 
from AstroLibraries import AstroLib_Ephem as AL_Eph

class SimsFlan_config:
    def __init__(self):
        with open("./confFiles/SimsFlan_config.yml") as file:
            SimsFlanagan_config = yaml.load(file, Loader=yaml.FullLoader)

        self.Nimp = SimsFlanagan_config['SimsFlan']['Nimp']
        self.date0 = SimsFlanagan_config['SimsFlan']['date0']
        self.t0 = AL_Eph.DateConv(self.date0,'calendar') #To JD


        bounds = SimsFlanagan_config['bounds']
        bn_v0_angle = tuple([z * np.pi for z in bounds['v0_angle'] ])
        bn_vf_angle = tuple([z * np.pi for z in bounds['vf_angle'] ])
        bn_t0 = tuple([z +  self.t0.JD_0 for z in bounds['t0'] ])
        bn_t_t = ( AL_BF.days2sec(bounds['t_t'][0]), AL_BF.days2sec(bounds['t_t'][1]) )
        bn_deltav_ang = tuple([z * np.pi for z in bounds['deltav_ang'] ])
        
        self.bnds = (bounds['v0'], bn_v0_angle, bn_v0_angle, \
                bounds['vf'], bn_vf_angle, bn_vf_angle, \
                bn_t0, bn_t_t)
        for i in range(self.Nimp): # 3 times because impulses are 3d vectors
            self.bnds += (bounds['deltav_mag'], bn_deltav_ang, bn_deltav_ang)


class OPT_config:
    def __init__(self):
        with open("./confFiles/OPT_config.yml") as file:
            OPT_config = yaml.load(file, Loader=yaml.FullLoader)

        self.MBH = OPT_config['MBH']
        self.MBH_generateDatabase = OPT_config['MBH_generateDatabase']
        self.EA = OPT_config['EA']
        self.CS = OPT_config['coordS']

class Fitness_config:
    def __init__(self):
        with open("./confFiles/Fitness_config.yml") as file:
            Fit_config = yaml.load(file, Loader=yaml.FullLoader)

        self.FEAS = Fit_config['FEASIB']

class ANN:
    def __init__(self):
        with open("./confFiles/ANN.yml") as file:
            ANN_config = yaml.load(file, Loader = yaml.FullLoader)
        
        self.ANN_archic = ANN_config['Architecture']
        self.ANN_train = ANN_config['Training']

class ANN_reg:
    def __init__(self):
        with open("./confFiles/ANN_reg.yml") as file:
            ANN_config = yaml.load(file, Loader = yaml.FullLoader)
        
        self.ANN_datab = ANN_config['Database']
        self.ANN_archic = ANN_config['Architecture']
        self.ANN_train = ANN_config['Training']

class ANN_GAN:
    def __init__(self):
        with open("./confFiles/ANN_GAN.yml") as file:
            ANN_config = yaml.load(file, Loader = yaml.FullLoader)
        
        self.ANN_datab = ANN_config['Database']
        self.Discriminator = ANN_config['Discriminator']
        self.Generator = ANN_config['Generator']
