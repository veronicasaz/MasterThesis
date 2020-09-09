import yaml

class SimsFlan_config:
    def __init__(self):
        with open("./confFiles/SimsFlan_config.yml") as file:
            SimsFlanagan_config = yaml.load(file, Loader=yaml.FullLoader)

        self.Nimp = SimsFlanagan_config['SimsFlan']['Nimp']
        self.date0 = SimsFlanagan_config['SimsFlan']['date0']
        self.t0 = AL_Eph.DateConv(date0,'calendar') #To JD

        bounds = SimsFlanagan_config['bounds']
        bounds[1] *= np.pi
        bnds = (bounds[0], bounds[1], bounds[1], \
                bounds[2], bounds[3], bounds[3], \
                bounds[4], bounds[5])


# bnd_v0 = (0, 5e3) 
#     bnd_v0_angle = (0., 2*np.pi)
#     bnd_vf = ( 0.0, 9e3) # Relative to the planet
#     # bnd_vf = ( v_escape *0.9, v_escape *1.1)
#     bnd_vf_angle = (0., 2*np.pi)
#     bnd_t0 = (t0.JD_0, t0.JD_0+1000) # Launch date
#     # bnd_m0 = (0, 200) # Mass should never be 0 as you add dry mass
#     bnd_t_t = (AL_BF.days2sec(200), AL_BF.days2sec(900) )
#     bnd_deltavmag = (0., 1.) # magnitude
#     bnd_deltavang = (-np.pi, np.pi) # angle

SimsFlan_config()