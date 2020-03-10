import numpy as np
import math

class Quaternion_attitude(object):
    def __init__(self):
        print('Quaternion_attitude')

    def get_error_norm(self,relative_attitude):
        target_attitude1 = np.asarray([1.,0.,0.,0.])
        target_attitude2 = -target_attitude1
        att_error = np.minimum(
                                np.linalg.norm(relative_attitude-target_attitude1) ,
                                np.linalg.norm(relative_attitude-target_attitude2) )
        return att_error
   
    def get_body_to_inertial_DCM(self,attitude):
        dcm_NB = EP2DCM(attitude).T
        return dcm_NB

    def qdot(self,attitude,w):
        return EPdot(attitude, w)
        
    def fix_attitude(self,attitude):
        attitude /= np.linalg.norm(attitude)  # normalize quaternions
        return attitude

    def get_random_attitude(self):
        attitude = np.random.uniform(low=-1.0,high=1.0,size=4)
        attitude /= np.linalg.norm(attitude)
        return attitude

    def dcm2q(self,C): 
        q = DCM2EP(C)
        return q

    def q2dcm(self,q):
        C = EP2DCM(q)
        return C

    def q2Euler321(self,q):
        return EP2Euler321(q)

    #def euler3212q(self,e):
    #    return Euler3212EP(e)
  
    def euler3212q(self,e):
        #
        # use DCM instead of direct mapping from Euler321 to EP to insure we always
        # get short path
        #
        C = Euler3212DCM(e)
        return DCM2EP(C)
 
    def sub(self,q1,q2):
        return subEP(q1,q2)

    def size(self):
        return 4

class Euler_attitude(object):

    """
    Note to self (I was confused):
        yaw  -> rotation around z axis      
        pitch -> rotation around y -axis
        roll -> rotation around x -axis
        [yaw, pitch, roll]
        [w1,  w2,    w3]

        So a rotation around x-axis (roll) should involve we (I had thought this was aa bug)
        rotation around y-axis (pitch) :  w2
        rotation around z-axis (yaw) : w3

        I guess that is why it is called 321 (z axis first)

    """
    def __init__(self):
        pass

    def get_error_norm(self,relative_attitude):
        att_error = np.linalg.norm(relative_attitude)
        return att_error

    def get_body_to_inertial_DCM(self,attitude):
        dcm_NB = Euler3212DCM(attitude).T
        return dcm_NB

    def qdot(self,attitude,w):
        return Euler321dot(attitude, w)

    def fix_attitude(self,attitude):
        for i in range(attitude.shape[0]):
            attitude[i] = picheck(attitude[i]) 
        return attitude

    def get_random_attitude(self):
        bounds = [np.pi, np.pi/2 , np.pi]
        attitude = np.random.uniform(low=-bounds,high=bounds,size=3)
        return attitude

    def dcm2q(self,C):
        q = DCM2Euler321(C)
        return q

    def q2dcm(self,q):
        C = Euler3212DCM(q)
        return C

    def sub(self,q1,q2):
        return subEuler321(q1,q2)

    def q2Euler321(self,q):
        return q 

    def euler3212q(self,e):
        return e

    def size(self):
        return 3

class MRP_attitude(object):
    def __init__(self):
        pass

    def get_error_norm(self,relative_attitude):
        return np.linalg.norm(relative_attitude)

    def get_body_to_inertial_DCM(self,attitude):
        dcm_NB = MRP2DCM(attitude).T
        return dcm_NB

    def qdot(self,attitude,w):
        return MRPdot(attitude, w)

    def fix_attitude(self,attitude):
        if np.dot(attitude, attitude) > 1:
            attitude = -attitude / np.dot(attitude, attitude)
        return attitude

    def get_random_attitude(self): 
        q0 = np.random.uniform(low=0.2,high=0.9)
        q13 = np.random.uniform(low=-1.0,high=1.0,size=4)
        q = np.hstack((q0,q13))
        q /= np.linalg.norm(q)
        s = EP2MRP(q)
        if np.dot(s,s) > 1:
            s = -s / np.dot(s,s)
        return s
    
    def dcm2q(self,C):
        q = DCM2MRP(C)
        return q

    def q2dcm(self,q):
        C = MRP2DCM(q)
        return C

    def sub(self,q1,q2):
        return subMRP(q1,q2)
 
    def size(self):
        return 3


def EP2MRP(q):
    """
    EP2MRP(Q1)

        Q = EP2MRP(Q1) translates the Euler parameter vector Q1
        into the MRP vector Q.
    """

    if q[0] < 0:
        q = -q;

    q1 = q[1]/(1+q[0]);
    q2 = q[2]/(1+q[0]);
    q3 = q[3]/(1+q[0]);

    return np.asarray([q1,q2,q3])
 
def MRPdot(q,w):
    B = np.zeros((3,3))
    B[0,0] = 1 - np.dot(q,q)+2*q[0]**2
    B[0,1] = 2*(q[0]*q[1]-q[2])
    B[0,2] = 2*(q[0]*q[2]+q[1])
    B[1,0] = 2*(q[1]*q[0]+q[2])
    B[1,1] = 1 - np.dot(q,q) + 2*q[1]**2
    B[1,2] = 2*(q[1]*q[2]-q[0])
    B[2,0] = 2*(q[2]*q[0]-q[1])
    B[2,1] = 2*(q[2]*q[1]+q[0])
    B[2,2] = 1 - np.dot(q,q) + 2*q[2]**2
    qdot = B.dot(w) / 4
    return qdot

def MRP2DCM(q):
    qs = skew(q)
    qm = np.dot(q,q)
    C = np.identity(3) + (8*qs.dot(qs) - 4*(1-qm)*qs) / (1+qm)**2
    return C



def EP2PRV(q):
    """
    EP2PRV(Q1)

        Q = EP2PRV(Q1) translates the Euler parameter vector Q1
        into the principal rotation vector Q.
    """

    p = 2*math.acos(q[0]);
    sp = math.sin(p/2);
    q1 = q[1]/sp*p;
    q2 = q[2]/sp*p;
    q3 = q[3]/sp*p;

    return np.asarray([q1,q2,q3])

def Euler3212EP(e):
    """
    Euler3212EP(E)

        Q = Euler3212EP(E) translates the 321 Euler angle
        vector E into the Euler parameter vector Q.
    """

    c1 = math.cos(e[0]/2);
    s1 = math.sin(e[0]/2);
    c2 = math.cos(e[1]/2);
    s2 = math.sin(e[1]/2);
    c3 = math.cos(e[2]/2);
    s3 = math.sin(e[2]/2);

    q0 = c1*c2*c3+s1*s2*s3;
    q1 = c1*c2*s3-s1*s2*c3;
    q2 = c1*s2*c3+s1*c2*s3;
    q3 = s1*c2*c3-c1*s2*s3;
    return np.asarray([q0,q1,q2,q3])

def subEuler321(e,e1):
    """
    subEuler321(E,E1)

        E2 = subEuler321(E,E1) computes the relative
        (3-2-1) Euler angle vector from E1 to E.
    """

    C = Euler3212DCM(e);
    C1 = Euler3212DCM(e1);
    C2 = C*C1.T;
    e2 = DCM2Euler321(C2);

    return e2;

def DCM2Euler321(C):
    """
    C2Euler321

        Q = C2Euler321(C) translates the 3x3 direction cosine matrix
        C into the corresponding (3-2-1) Euler angle set.
    """
    q = np.zeros(3)
    q[0] = np.arctan2(C[0,1],C[0,0]);
    q[1] = np.arcsin(-C[0,2]);
    q[2] = np.arctan2(C[1,2],C[2,2]);
    return q;

def PRV2elem(r):
    """
    PRV2elem(R)

        Q = PRV2elem(R) translates a prinicpal rotation vector R
        into the corresponding principal rotation element set Q.
    """
    #q = np.matrix("0.;0.;0.;0.");
    q = np.zeros(4)
    q[0] = math.sqrt((r.T*r)[0]);
    q[1] = r[0]/q[0];
    q[2] = r[1]/q[0];
    q[3] = r[2]/q[0];
    return q;

def PRV2EP(qq1):
    """"
    PRV2EP(Q1)

        Q = PRV2EP(Q1) translates the principal rotation vector Q1
        into the Euler parameter vector Q.
    """

    #q = np.matrix("0.;0.;0.;0.");
    q = np.zeros(4)
    q1 = PRV2elem(qq1);
    sp = math.sin(q1[0]/2);
    q[0] = math.cos(q1[0]/2);
    q[1] = q1[1]*sp;
    q[2] = q1[2]*sp;
    q[3] = q1[3]*sp;

    return q;

def EP2PRV(q):
    """
    EP2PRV(Q1)

        Q = EP2PRV(Q1) translates the Euler parameter vector Q1
        into the principal rotation vector Q.
    """

    p = 2*math.acos(q[0]);
    sp = math.sin(p/2);
    q1 = q[1]/sp*p;
    q2 = q[2]/sp*p;
    q3 = q[3]/sp*p;
    return np.asarray([q1,q2,q3])
    #return np.matrix([[q1],[q2],[q3]]);


def subEP(b1,b2):
    """
    subEP(B1,B2)

        Q = subEP(B1,B2) provides the Euler parameter vector
        which corresponds to relative rotation from B2
        to B1.
    """

    q = np.zeros(4)
    q[0] =  b2[0]*b1[0]+b2[1]*b1[1]+b2[2]*b1[2]+b2[3]*b1[3];
    q[1] = -b2[1]*b1[0]+b2[0]*b1[1]+b2[3]*b1[2]-b2[2]*b1[3];
    q[2] = -b2[2]*b1[0]-b2[3]*b1[1]+b2[0]*b1[2]+b2[1]*b1[3];
    q[3] = -b2[3]*b1[0]+b2[2]*b1[1]-b2[1]*b1[2]+b2[0]*b1[3];

    return q;

def subMRP(q1,q2):
    """
    subMRP(Q1,Q2)

        Q = subMRP(Q1,Q2) provides the MRP vector
        which corresponds to relative rotation from Q2
        to Q1.
    """
    q1sq = np.dot(q1,q1) 
    q2sq = np.dot(q2,q2)
    
    q = (1-q2sq)*q1 - (1-q1sq)*q2 + 2*np.cross(q1,q2);
    q = q / (1 + q1sq * q2sq + 2*np.dot(q1,q2) );

    return q;

def rad2deg(phi):
    return phi * 180 / np.pi

def skew(v):
    if len(v) == 4: v = v[:3]/v[3]
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T

def dcm2_321(DCM):
    phi1 = np.arctan2(DCM[0,1],DCM[0,0])
    phi2 = -np.arcsin(DCM[0,2])
    phi3 = np.arctan2(DCM[1,2],DCM[2,2])
    return phi1,phi2,phi3

def Euler321dot(q, w):
    psi =   q[0]
    theta = q[1]
    phi =   q[2]
    B = [[0.0 ,             np.sin(phi),                       np.cos(phi) ] ,
         [0.0,              np.cos(phi)*np.cos(theta)  ,  -np.sin(phi)*np.cos(theta) ],
         [np.cos(theta),    np.sin(phi)*np.sin(theta),   np.cos(phi)*np.sin(theta) ] ]
    B = np.asarray(B) 
    qdot = 1 / np.cos(theta) * B.dot(w)
    return qdot

def EPdot(beta1,w):
    beta = beta1.copy()

    B = [[-beta[1], -beta[2], -beta[3]],
        [ beta[0], -beta[3], beta[2]],
        [beta[3],  beta[0], -beta[1]],
        [-beta[2], beta[1], beta[0]] ]
    B = np.asarray(B)
    betadot = 1./2 * B.dot(w)
    return betadot

def EP2DCM(q):
    b0 = q[0]
    b1 = q[1]
    b2 = q[2]
    b3 = q[3]
    C = np.zeros((3,3))
    C[0,0] = b0**2 + b1**2 - b2**2 - b3**2
    C[0,1] = 2*(b1*b2 + b0*b3)
    C[0,2] = 2*(b1*b3 - b0*b2)
    C[1,0] = 2*(b1*b2 - b0*b3)
    C[1,1] = b0**2 - b1**2 + b2**2 - b3**2
    C[1,2] = 2*(b2*b3 + b0*b1)
    C[2,0] = 2*(b1*b3 + b0*b2)
    C[2,1] = 2*(b2*b3 - b0*b1)
    C[2,2] = b0**2 - b1**2 - b2**2 + b3**2
    return C

def Euler3212DCM(q):
    """
    Euler3212C

        C = Euler3212C(Q) returns the direction cosine
        matrix in terms of the 3-2-1 Euler angles.
        Input Q must be a 3x1 vector of Euler angles.
    """
    
    st1 = np.sin(q[0]);
    ct1 = np.cos(q[0]);
    st2 = np.sin(q[1]);
    ct2 = np.cos(q[1]);
    st3 = np.sin(q[2]);
    ct3 = np.cos(q[2]);

    C = np.zeros((3,3))
    C[0,0] = ct2*ct1;
    C[0,1] = ct2*st1;
    C[0,2] = -st2;
    C[1,0] = st3*st2*ct1-ct3*st1;
    C[1,1] = st3*st2*st1+ct3*ct1;
    C[1,2] = st3*ct2;
    C[2,0] = ct3*st2*ct1+st3*st1;
    C[2,1] = ct3*st2*st1-st3*ct1;
    C[2,2] = ct3*ct2;

    return C;



 
def DCM2EP(C):
    """
    C2EP

        Q = C2EP(C) translates the 3x3 direction cosine matrix
        C into the corresponding 4x1 Euler parameter vector Q,
        where the first component of Q is the non-dimensional
        Euler parameter Beta_0 >= 0. Transformation is done
        using the Stanley method.
    """

    tr = np.trace(C);
    b2 = np.zeros(4)
    b2[0] = 1.
    b2[0] = (1+tr)/4;
    b2[1] = (1+2*C[0,0]-tr)/4;
    b2[2] = (1+2*C[1,1]-tr)/4;
    b2[3] = (1+2*C[2,2]-tr)/4;

    case = np.argmax(b2);
    b = b2;
    if   case == 0:
        b[0] = math.sqrt(b2[0]);
        b[1] = (C[1,2]-C[2,1])/4/b[0];
        b[2] = (C[2,0]-C[0,2])/4/b[0];
        b[3] = (C[0,1]-C[1,0])/4/b[0];
    elif case == 1:
        b[1] = math.sqrt(b2[1]);
        b[0] = (C[1,2]-C[2,1])/4/b[1];
        if b[0]<0:
            b[1] = -b[1];
            b[0] = -b[0];
        b[2] = (C[0,1]+C[1,0])/4/b[1];
        b[3] = (C[2,0]+C[0,2])/4/b[1];
    elif case == 2:
        b[2] = math.sqrt(b2[2]);
        b[0] = (C[2,0]-C[0,2])/4/b[2];
        if b[0]<0:
            b[2] = -b[2];
            b[0] = -b[0];
        b[1] = (C[0,1]+C[1,0])/4/b[2];
        b[3] = (C[1,2]+C[2,1])/4/b[2];
    elif case == 3:
        b[3] = math.sqrt(b2[3]);
        b[0] = (C[0,1]-C[1,0])/4/b[3];
        if b[0]<0:
            b[3] = -b[3];
            b[0] = -b[0];
        b[1] = (C[2,0]+C[0,2])/4/b[3];
        b[2] = (C[1,2]+C[2,1])/4/b[3];

    return np.squeeze(b);

def EP2Euler321(q):
    """
    EP2Euler321

        E = EP2Euler321(Q) translates the Euler parameter vector
        Q into the corresponding (3-2-1) Euler angle set.
    """

    q0 = q[0];
    q1 = q[1];
    q2 = q[2];
    q3 = q[3];

    e1 = math.atan2(2*(q1*q2+q0*q3),q0*q0+q1*q1-q2*q2-q3*q3);
    e2 = math.asin(-2*(q1*q3-q0*q2));
    e3 = math.atan2(2*(q2*q3+q0*q1),q0*q0-q1*q1-q2*q2+q3*q3);

    return np.asarray([e1,e2,e3]);

def DCM2q_old(C):
    print(np.trace(C))
    b0 = 1./2*np.sqrt(np.trace(C) + 1.)
    b1 = (C[1,2] - C[2,1]) / (4. * b0)
    b2 = (C[2,0] - C[0,2]) / (4. * b0)
    b3 = (C[0,1] - C[1,0]) / (4. * b0)
    return np.asarray([b0,b1,b2,b3])

def DCM2MRP(C):
    """
    C2MRP

        Q = C2MRP(C) translates the 3x3 direction cosine matrix
        C into the corresponding 3x1 MRP vector Q where the
        MRP vector is chosen such that |Q| <= 1.
    """

    b = C2EP(C);

    q = np.zeros(3)
    q[0] = b[1]/(1+b[0]);
    q[1] = b[2]/(1+b[0]);
    q[2] = b[3]/(1+b[0]);

    return q;

def DCM(a,b):
    """
        Create the DCM that maps a->b, i.e., BN
        i.e., inertial (n) -> body  (b)
    """

    eps = 1e-9
    a += eps
    b += eps
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    V = skew(v)
    I = np.identity(3)
    ### begin MOD
    if s > 1e-8:
        R = I + V + V.dot(V) * (1-c)/s**2
    else:
        R = I
    ### end mod
    return R

def DCM2(a,b):
    """
        Create the DCM that maps a->b, i.e., BN
        i.e., inertial (n) -> body  (b)
    """

    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    V = skew(v)
    I = np.identity(3)
    ### begin MOD
    if s > 1e-8:
        R = I + V + V.dot(V) * (1-c)/s**2
    else:
        R = I
    ### end mod
    return R

def picheck(x):
    """
        Picheck(x)

        Makes sure that the angle x lies within +/- Pi.
    """
    if x > np.pi:
        return x - 2*np.pi 
    if x< -np.pi:
        return x + 2*np.pi
    return x

