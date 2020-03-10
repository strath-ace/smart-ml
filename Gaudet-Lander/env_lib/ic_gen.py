import numpy as np

class Landing_icgen(object):

    def __init__(self, attitude_parameterization=None, 
                 downrange = (500,1500 , -70, -10), crossrange = (-500,500 , -30,30),  altitude = (1900,2100,-90,-70) ,
                 yaw   = (0.0, 0.0, 0.0, 0.0) , 
                 pitch = (np.pi/4, np.pi/4, 0.0, 0.0),
                 roll  = (0.0, 0.0, 0.0, 0.0),
                 scale=None, debug=False, adapt_apf_v0=True,
                 noise_u=np.zeros(3), noise_sd=np.zeros(3), mass_uncertainty=0.0, g_uncertainty=(0.0, 0.0), l_offset=0.0,
                 inertia_uncertainty_diag=0.0, inertia_uncertainty_offdiag=0.0): 
        self.downrange = downrange
        self.crossrange = crossrange
        self.altitude = altitude
        self.attitude_parameterization=attitude_parameterization
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

        self.noise_u = noise_u
        self.noise_sd = noise_sd
        self.mass_uncertainty = mass_uncertainty
        self.g_uncertainty = g_uncertainty
        self.l_offset = l_offset
        self.nominal_g = -3.7114

        self.inertia_uncertainty_diag = inertia_uncertainty_diag
        self.inertia_uncertainty_offdiag = inertia_uncertainty_offdiag 
    
        self.debug = debug
        self.adapt_apf_v0 = adapt_apf_v0

        assert downrange[0] <= downrange[1]
        assert downrange[2] <= downrange[3]
        assert crossrange[0] <= crossrange[1]
        assert crossrange[2] <= crossrange[3]
        assert altitude[0] <= altitude[1]
        assert altitude[2] <= altitude[3]

        if scale is not None:
            self.downrange = tuple([x / scale for x in downrange])
            self.crossrange = tuple([x / scale for x in crossrange])
            self.altitude = tuple([x / scale for x in altitude])
 

    def show(self):
        print('Landing_icgen:')
        print('    downrange                   : ',self.downrange)
        print('    crossrange                  : ',self.crossrange)
        print('    altitude                    : ',self.altitude)
 
    def set_ic(self , lander, dynamics):

        dynamics.noise_u = np.random.uniform(low=-self.noise_u, high=self.noise_u,size=3)
        dynamics.noise_sd = self.noise_sd

        dynamics.g[2] =  np.random.uniform(low=self.nominal_g * (1 - self.g_uncertainty[0]), 
                                           high=self.nominal_g * (1 + self.g_uncertainty[0]))
        
        dynamics.g[0:2] = np.random.uniform(low=-self.nominal_g * self.g_uncertainty[1],
                                            high=self.nominal_g * self.g_uncertainty[1],
                                            size=2)

        dynamics.l_offset = np.random.uniform(low=-self.l_offset,
                                           high=self.l_offset,
                                           size=3)
                                   
        lander.init_mass = np.random.uniform(low=lander.nominal_mass * (1 - self.mass_uncertainty), 
                                             high=lander.nominal_mass * (1 + self.mass_uncertainty))
     
        r_downrange = np.random.uniform(low=self.downrange[0], high=self.downrange[1])
        r_crossrange = np.random.uniform(low=self.crossrange[0], high=self.crossrange[1])
        r_altitude = np.random.uniform(low=self.altitude[0], high=self.altitude[1])

        a_yaw   =  np.random.uniform(low=self.yaw[0], high=self.yaw[1])
        a_pitch =  np.random.uniform(low=self.pitch[0], high=self.pitch[1])
        a_roll  =  np.random.uniform(low=self.roll[0], high=self.roll[1])

        v_downrange = np.random.uniform(low=self.downrange[2], high=self.downrange[3])
        v_crossrange = np.random.uniform(low=self.crossrange[2], high=self.crossrange[3])
        v_altitude = np.random.uniform(low=self.altitude[2], high=self.altitude[3])

        w_yaw   =  np.random.uniform(low=self.yaw[2], high=self.yaw[3])
        w_pitch =  np.random.uniform(low=self.pitch[2], high=self.pitch[3])
        w_roll  =  np.random.uniform(low=self.roll[2], high=self.roll[3])

        lander.state['position'] = np.asarray([r_downrange,r_crossrange,r_altitude])
        lander.state['velocity'] = np.asarray([v_downrange,v_crossrange,v_altitude])

        lander.state['attitude'] = self.attitude_parameterization.euler3212q(np.asarray([a_yaw, a_pitch, a_roll]))
        lander.state['attitude_321'] = np.asarray([a_yaw, a_pitch, a_roll])
 
        lander.state['w'] = np.asarray([w_yaw, w_pitch, w_roll])
 
        lander.state['thrust'] = np.ones(3)*lander.min_thrust 
        lander.state['mass']   = lander.init_mass

        it_noise1 = np.random.uniform(low=-self.inertia_uncertainty_offdiag, 
                                      high=self.inertia_uncertainty_offdiag, 
                                      size=(3,3))
        np.fill_diagonal(it_noise1,0.0)
        it_noise1 = (it_noise1 + it_noise1.T)/2
        it_noise2 = np.diag(np.random.uniform(low=-self.inertia_uncertainty_diag,
                            high=self.inertia_uncertainty_diag,
                            size=3))
        lander.inertia_tensor = lander.nominal_inertia_tensor + it_noise1 + it_noise2
        if self.adapt_apf_v0:
            lander.apf_v0 = np.linalg.norm(lander.state['velocity'])

        lander.apf_state = False 
    
        if self.debug:
            print(dynamics.g, lander.state['mass'])
            print(lander.inertia_tensor)
