import SimpleITK as sitk
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.ndimage as nimg
from numba import njit
import tensorflow as tf  # tf.__version__: 1.12.0

import scipy.io as sio

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from sklearn.decomposition import PCA
import time

spread = 70
spatial_weight = 1.0


def get_plane_normal(cp, wp):
    #cp, wp = cd_pos, walk_pos[id]
    if cp<15: cp=15
    vs = wp[cp-5:cp+5+1] - wp[cp:cp+11]
    v = np.mean(vs, axis=0)
    v = v/np.sqrt(np.sum(v**2))
    d = np.sum(v*wp[cp]) # ax+by+cz-d=0
    return v, d


def get_slice(v, p, mask, epsilon=1e-9):
    # v,d,p = n, d, walk_pos[id][cd_pos]
    # ref_normal: [0,0,1] (axial view), our normal: v
    ref_normal = np.array([0,0,1])
    vu = v/np.sqrt(np.sum(v**2))
    vu = np.where(vu==0, epsilon, vu)
    ref_normal = np.where(ref_normal==0, epsilon, ref_normal)

    costheta = np.dot(ref_normal, vu)
    e = np.cross(ref_normal, vu)
    if np.sum(e)!=0: e = e/np.sqrt(np.sum(e**2))
    e = np.where(e == 0, epsilon, e)

    c = costheta
    s = np.sqrt(1 - c * c)
    C = 1 - c
    x,y,z = e[0], e[1], e[2]
    rmat = np.array([[x * x * C + c,  x * y * C - z * s,  x * z * C + y * s],
                     [y * x * C + z * s, y * y * C + c,  y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s,  z * z * C + c]])

    px, py, pz = np.meshgrid(np.arange(-spread,spread), np.arange(-spread,spread), np.arange(0,1))
    points = np.concatenate([px,py,pz], axis=-1)
    new_points = np.matmul(points, rmat.T)
    new_points += p
    new_points = np.int32(new_points+0.5)

    new_points[new_points<0] = 0
    for i in range(3):
        a = new_points[:,:,i]
        a[a>=mask.shape[i]] = mask.shape[i]-1
        new_points[:,:,i] = a

    return mask[new_points[:,:,1], new_points[:,:,0], new_points[:,:,2]], new_points


def segment_orifice_in_slice(slice):
    distance_map = nimg.distance_transform_edt(slice)
    dist_f = nimg.maximum_filter(distance_map, 30)
    local_max = peak_local_max(dist_f, indices=False, min_distance=30, labels=slice)
    markers = nimg.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance_map, markers, mask=slice)
    orifice_label = labels[spread-1, spread-1]
    orifice = labels.copy()
    orifice[labels==orifice_label] = 1
    orifice[labels!=orifice_label] = 0
    return orifice


def get_eigen(orifice, slice_pts):
    px, py, pz = slice_pts[..., 0], slice_pts[..., 1], slice_pts[..., 2]
    px, py, pz = px[orifice==1], py[orifice==1], pz[orifice==1]
    pp = np.concatenate([px[:,np.newaxis], py[:,np.newaxis], pz[:,np.newaxis]], axis=1)
    pp = pp
    pp_mean = np.mean(pp, axis=0)
    pp = pp- np.mean(pp, axis=0)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(pp/100)
    return pca.explained_variance_[:2], pca.components_[:2], pp_mean


def get_axis_points(orifice, slice_pts, pt_mean, eigval, eigvec):
    half_length = 100*np.sqrt(2.0*2.0*eigval)
    plus_end, minus_end = pt_mean + eigvec*half_length[:,np.newaxis], pt_mean - eigvec*half_length[:,np.newaxis]

    maskover = np.zeros(shape=orifice.shape, dtype=orifice.dtype)

    diff = slice_pts - pt_mean
    diff = np.sum(diff ** 2, axis=2)
    a = np.argsort(diff.flatten())
    au = np.unravel_index(a[0], diff.shape)
    maskover[au[0], au[1]] = 2.0

    diff = slice_pts - plus_end[0,]
    diff = np.sum(diff**2, axis=2)
    a = np.argsort(diff.flatten())
    au = np.unravel_index(a[0], diff.shape)
    major0 = slice_pts[au[0], au[1],:]
    maskover[au[0], au[1]] = 2.0
    diff = slice_pts - plus_end[1,]
    diff = np.sum(diff ** 2, axis=2)
    a = np.argsort(diff.flatten())
    au = np.unravel_index(a[0], diff.shape)
    minor0 = slice_pts[au[0], au[1], :]
    maskover[au[0], au[1]] = 2.0

    diff = slice_pts - minus_end[0,]
    diff = np.sum(diff**2, axis=2)
    a = np.argsort(diff.flatten())
    au = np.unravel_index(a[0], diff.shape)
    major1 = slice_pts[au[0], au[1],:]
    maskover[au[0], au[1]] = 2.0
    diff = slice_pts - minus_end[1,]
    diff = np.sum(diff ** 2, axis=2)
    a = np.argsort(diff.flatten())
    au = np.unravel_index(a[0], diff.shape)
    minor1 = slice_pts[au[0], au[1], :]
    maskover[au[0], au[1]] = 2.0
    return [major0, major1], [minor0, minor1], maskover


def show(im):
    plt.figure()
    plt.imshow(im, cmap='gray')


def load_dicom(dicom_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    vol = sitk.GetArrayFromImage(image)
    vol = np.transpose(vol, axes=(1, 2, 0))
    vol = np.flip(vol, axis=2)
    vol = np.flip(vol, axis=0)

    vol[vol < -100] = -100
    vol[vol > 900] = -100

    m = np.mean(vol)
    std = np.std(vol)

    add = np.int32(((vol - m) / std) * 32.0 + 0.5)

    vol = 128 + add
    vol[vol < 0] = 0
    vol[vol > 255] = 255
    return vol


def crop_points(seed):
    Y, X, Z = vol.shape

    xmin, xmax = seed[1] - 150, seed[1] + 50
    ymin, ymax = seed[0] - 150, seed[0] + 50
    zmin, zmax = seed[2] - 10, seed[2] + 150

    xmin, xmax = np.clip(xmin, 1, X - 2), np.clip(xmax, 1, X - 2)
    ymin, ymax = np.clip(ymin, 1, Y - 2), np.clip(ymax, 1, Y - 2)
    zmin, zmax = np.clip(zmin, 1, Z - 2), np.clip(zmax, 1, Z - 2)
    return xmin, xmax, ymin, ymax, zmin, zmax, X, Y, Z


def denoise(vol):
    denoised = nimg.median_filter(vol[ymin:ymax, xmin:xmax, zmin:zmax], 3)
    vol[ymin:ymax, xmin:xmax, zmin:zmax] = denoised

@njit
def forward_pass(geo, ds, vol, seed, xmax, xmin, ymax, ymin, zmax, zmin, r):
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            for k in range(zmin, zmax):
                g_b_min = geo[i, j, k]
                vol_ijk = vol[i, j, k]
                for ii in range(-1, 1):
                    for jj in range(-1, 1):
                        for kk in range(-1, 1):
                            g_a = geo[i + ii, j + jj, k + kk]
                            di = vol[i + ii, j + jj, k + kk] - vol_ijk
                            g_ab = di ** 2 + spatial_weight * (ii**2 + jj**2 + kk**2)
                            g_b = g_a + g_ab
                            if g_b < g_b_min:
                                g_b_min = g_b
                geo[i, j, k] = g_b_min

    return geo

@njit
def backward_pass(geo, ds, vol, seed, xmax, xmin, ymax, ymin, zmax, zmin, r):
    for i in range(ymax, ymin, -1):
        for j in range(xmax, xmin, -1):
            for k in range(zmax, zmin, -1):
                g_b_min = geo[i,j,k]
                vol_ijk = vol[i,j,k]
                for ii in range(0, 2):
                    for jj in range(0, 2):
                        for kk in range(0, 2):
                            g_a = geo[i+ii, j+jj, k+kk]
                            di = vol[i+ii, j+jj, k+kk] - vol_ijk
                            g_ab = di**2 + spatial_weight * (ii**2 + jj**2 + kk**2)
                            g_b = g_a + g_ab
                            if g_b < g_b_min:
                                g_b_min = g_b
                geo[i, j, k] = g_b_min

    return geo

@njit
def update_geo(geo, ds_forward, ds_backward, vol, seed, xmax, xmin, ymax, ymin, zmax, zmin, r):
    # forward pass
    geo = forward_pass(geo, ds_forward, vol, seed, xmax, seed[1] - r, ymax, seed[0] - r, zmax, seed[2] - r, r)

    # backward pass
    geo = backward_pass(geo, ds_backward, vol, seed, xmax, xmin, ymax, ymin, zmax, zmin, r)

    # forward pass
    geo = forward_pass(geo, ds_forward, vol, seed, xmax, xmin, ymax, ymin, zmax, zmin, r)

    # backward pass
    geo = backward_pass(geo, ds_backward, vol, seed, xmax, xmin, ymax, ymin, zmax, zmin, r)

    return geo


def geo_trans(vol, seed, xmax, xmin, ymax, ymin, zmax, zmin, r=5):
    geo = np.ones(shape=vol.shape, dtype=np.float32) * 99999.0
    geo[seed[0] - r:seed[0] + r, seed[1] - r:seed[1] + r, seed[2] - r:seed[2] + r] = 0.0
    ds = np.meshgrid(np.arange(-1, 1), np.arange(-1, 1), np.arange(-1, 1))
    ds_forward = np.sqrt(ds[0] ** 2 + ds[1] ** 2 + ds[2] ** 2)
    ds = np.meshgrid(np.arange(0, 2), np.arange(0, 2), np.arange(0, 2))
    ds_backward = np.sqrt(ds[0] ** 2 + ds[1] ** 2 + ds[2] ** 2)
    geo = update_geo(geo, ds_forward, ds_backward, vol, seed, xmax, xmin, ymax, ymin, zmax, zmin, r)
    return geo


def segment(vol, seed, threshold, xmax, xmin, ymax, ymin, zmax, zmin):
    geo = geo_trans(vol, seed, xmax, xmin, ymax, ymin, zmax, zmin)
    geo = np.sqrt(geo)
    notroi_marker = -1
    geo[geo==np.max(geo)] = notroi_marker
    threshold = threshold * np.max(geo)
    seg = np.zeros(shape=vol.shape, dtype=np.int32)
    seg[geo<=threshold] = 1
    seg[geo==notroi_marker] = 0
    geo[geo==notroi_marker] = 0
    return seg, geo


def mask_dt(mask):
    return nimg.distance_transform_cdt(mask, metric='cityblock')

def cd_walk(seed, mask):
    dt = np.zeros(shape=mask.shape, dtype=np.float32)
    dt[ymin:ymax, xmin:xmax, zmin:zmax] = mask_dt(mask[ymin:ymax, xmin:xmax, zmin:zmax])
    x = seed.copy()
    wps = []
    cds = []
    visited = np.ones(shape=mask.shape, dtype=np.int32)
    trend = np.ones(shape=[3,3,3], dtype=np.int32)*(-1)
    trend[0,:,:] = 1
    trend[:,0,:] = 1
    trend[:,:,1] = 1
    for i in range(300):
        visited[x[0], x[1], x[2]] = -1
        E = dt[x[0]-1:x[0]+2,x[1]-1:x[1]+2,x[2]-1:x[2]+2]*visited[x[0]-1:x[0]+2,x[1]-1:x[1]+2,x[2]-1:x[2]+2]
        E[E<0] = 0
        E = E*trend
        max_pos = np.unravel_index(np.argmax(E), E.shape)
        x = x + np.array(max_pos) - 1
        #print(x)
        wps.append(x)
        cds.append(dt[x[0], x[1], x[2]])

    return np.array(wps), np.array(cds)

C_state_size = 50
C_n_conv = 3

class World:
    def __init__(self):
        self.dist, self.gt, self.pos = None, None, None
        self.pos = 0

    def set_world(self, dist, gt):
        self.dist, self.gt = np.float32(np.array(dist)), np.float32(gt)

    def set_pos(self, pos):
        self.pos = pos

    def get_state(self):
        one_hot_pos = np.zeros(dtype=np.float32, shape=self.dist.shape)
        one_hot_pos[self.pos] = 1.0
        state = np.concatenate([self.dist, one_hot_pos])
        return state

    def move(self, action):
        pos_prev = self.pos
        dist_prev = np.abs(pos_prev-self.gt)
        if action == 0:
            self.pos += 1
        else:
            self.pos -= 1
        dist_now = np.abs(self.pos-self.gt)
        r = -1.0
        if dist_now < dist_prev:
            r = 1.0
        if dist_now <= 1.0:
            r = 2.0
        if self.pos < 0 or self.pos>=300:
            r = -10.0
            self.pos = 0
        return r, self.get_state()


class World_p:
    def __init__(self, N_max=1000):  # dist::Nx300, gt: N, pos: N
        self.dist, self.gt, self.pos, self.N = None, None, None, None
        self.N_max = N_max

    def set_world(self, dist, gt):
        self.dist, self.gt = np.float32(np.array(dist)), np.squeeze(np.float32(gt))
        self.N = len(self.dist)

    def set_pos(self, pos):
        self.pos = np.squeeze(pos)

    def get_state(self, size=C_state_size): # state: Nx50
        h = size//2
        dist_pad = np.pad(self.dist, ((0,0),(h,h)), mode='constant')
        rows = np.arange(0, self.N)[:,np.newaxis]
        cols = np.repeat(np.arange(0, size)[np.newaxis,:], self.N, 0)
        cols = cols + self.pos[:,np.newaxis]
        state = dist_pad[rows, cols]
        return state

    def move(self, action): # action: N, r:N
        pos_prev = self.pos
        dist_prev = np.abs(pos_prev-self.gt)
        self.pos[action==0] += 1
        self.pos[action==1] -= 1
        dist_now = np.abs(self.pos-self.gt)
        r = -1.0*np.ones(shape=[self.N], dtype=np.float32)
        r[dist_now<dist_prev] = 1.0
        r[dist_now<=1.0] = 2.0
        r[self.pos<0] = -10.0
        r[self.pos>=300] = -10.0
        self.pos[self.pos<0] = 0
        self.pos[self.pos>=300] = 299
        return r


class Agent:
    def __init__(self, state_length=C_state_size, learn_rate=1e-5, lamda=0e-2):
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=lamda)
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, state_length])
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.policy_old = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.learning_rate = learn_rate
        self.build_model()
        pass

    def conv(self, state):
        layer = tf.reshape(state, [-1, C_state_size, 1])
        n_conv = C_n_conv
        n=8
        feat_dim = C_state_size
        for i in range(n_conv):
            layer = tf.layers.conv1d(inputs=layer, filters=n * (2 ** i), kernel_size=3, activation=tf.nn.relu, padding='same',
                                     kernel_regularizer=self.regularizer)
            #layer = tf.layers.conv1d(inputs=layer, filters=n * (2 ** i), kernel_size=3, activation=tf.nn.relu, padding='same')
            layer = tf.layers.max_pooling1d(inputs=layer, pool_size=2, strides=2)
            feat_dim = feat_dim//2

        layer = tf.reshape(layer, [-1, feat_dim * n * (2**(n_conv-1))])
        return layer

    def policy(self, state):
        layer = state
        for i in range(2):
            layer = tf.layers.dense(inputs=layer, units=32//(i+1), activation=tf.nn.relu, kernel_regularizer=self.regularizer)
        layer = tf.layers.dense(inputs=layer, units=2, activation=None, kernel_regularizer=self.regularizer)
        layer = tf.nn.softmax(layer, 1)
        layer = tf.clip_by_value(layer, 0.1, 0.9)
        return layer

    def compute_loss(self, pi, a, pi_old, advantage):
        a_one_hot = tf.one_hot(indices=a, depth=2, on_value=1.0, off_value=0.0)
        a_probs = tf.multiply(pi, a_one_hot)
        a_probs = tf.reduce_sum(a_probs, axis=1)
        rt = a_probs/pi_old
        clipped_loss = -tf.reduce_mean(tf.reduce_min([rt*advantage,
                                                      tf.clip_by_value(rt, 0.8, 1.2)*advantage]))
        return clipped_loss

    def build_model(self):
        feat = self.conv(self.state)
        self.pi = self.policy(feat)
        pi_loss = self.compute_loss(self.pi, self.actions, self.policy_old, self.advantage)
        pi_loss = pi_loss + tf.losses.get_regularization_loss()
        self.pi_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(pi_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100)

    def get_pi(self, state):
        if len(state.shape) < 2:
            state = state[np.newaxis,...]
        return self.sess.run(self.pi, {self.state:state})

    def optimize(self, state, action, advantage, pi_old):
        self.sess.run(self.pi_opt, {self.state:state, self.actions:action,
                                    self.advantage:advantage, self.policy_old:pi_old})


world = World_p()
agent = Agent()
agent.saver.restore(agent.sess, 'net_cd-rl-patch_size_%d/best'%C_state_size)


def one_step(epsilon=0.7):
    #print(world.N)
    state = world.get_state()
    #print(state.shape)
    policy = agent.get_pi(state) # policy: Nx2
    action = np.argmax(policy, axis=1) # action: N
    random_action = np.random.randint(0, 2, [len(action)])
    random_probs = np.random.random([len(action)])
    action[random_probs > epsilon] = random_action[random_probs > epsilon]
    reward = world.move(action)
    return state, action, reward, policy[np.arange(0, len(action)), action]


def episode_history(pos=10, max_step=300, epsilon=0.7):
    pt, s_, a_, r_, p_ = [], [], [], [], []
    world.set_pos(pos)
    #pt.append(world.pos)
    for i in range(max_step):
        s, a, r, p = one_step(epsilon)
        s_.extend(s)
        a_.extend(a)
        r_.extend(r)
        p_.extend(p)
        pt.append(world.pos.copy())  # pos: N
    return pt, s_, a_, r_, p_


def episode(pos=10, max_step=300, epsilon=0.7):
    pt, s_, a_, r_, p_ = [], [], [], [], []
    world.set_pos(pos)
    for i in range(max_step):
        s, a, r, p = one_step(epsilon)
        s_.extend(s)
        a_.extend(a)
        r_.extend(r)
        p_.extend(p)
    pt = world.pos  # pos: N
    return pt, s_, a_, r_, p_


def explore(max_episode=10, max_step=300, epsilon=0.7, pos=None):
    pt, s, a, r, p = [], [], [], [], []
    for e in range(max_episode):
        if pos is None:
            pos = np.random.randint(10, 290, [world.N])
        pt_, s_, a_, r_, p_ = episode(pos, max_step, epsilon)
        pt.append(pt_)
        s.extend(s_)
        a.extend(a_)
        r.extend(r_)
        p.extend(p_)
    return pt, s, a, r, p


def explore_multi_ims(ims, gts, max_episode=10, max_step=300, epsilon=0.7):
    s, a, r, p = [], [], [], []
    n= len(ims)*max_episode
    ims_e = np.float32(np.repeat(ims, max_episode, 0))
    gts_e = np.repeat(gts, max_episode, 0)

    for i in range(len(ims_e)):
        ims_e[i] = ims_e[i]/np.mean(ims_e[i])

    batch_size = float(world.N_max)
    rand_id = np.arange(0, n)
    for batch in range(np.int32(np.ceil(n / batch_size))):
        start, end = np.int32(batch * batch_size), np.int32((batch + 1) * batch_size)
        if end > n:
            end = n
        m = rand_id[start:end]
        world.set_world(ims_e[m], gts_e[m])
        _, s_, a_, r_, p_ = explore(1, max_step, epsilon)
        s.extend(s_)
        a.extend(a_)
        r.extend(r_)
        p.extend(p_)
    return s, a, r, p


def test_multi_ims(ims, gts, max_episode=1, max_step=300, epsilon=1.0, init_pos=150):
    pt, r = [], []
    ims = np.float32(ims)
    for i in range(len(ims)):
        ims[i] = ims[i]/np.mean(ims[i])
    world.set_world(ims, gts)
    pos = np.repeat(np.array([init_pos]), len(ims), axis=0)
    pt, _, _, r, _ = episode(pos, max_step, epsilon)
    return pt, r


def test(images, init_pos=150):
    images = np.concatenate([images[np.newaxis,:], images[np.newaxis,:]], axis=0)
    gts = np.array([[30],[30]])
    y, _ = test_multi_ims(images, gts, init_pos=init_pos)
    return y


def rotate_3d_vector(v, phi,theta,psi ): # phi, theta, psi: rotation about x,y,z-axes
    #v=np.array([0,0,1])
    #phi, theta, psi = 0.0, np.pi/2, 0.0
    A = np.array([[np.cos(theta)*np.cos(psi), -np.cos(phi)*np.sin(psi) + np.sin(phi)* np.sin(theta)*np.cos(psi), np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi)],
                 [np.cos(theta)*np.sin(psi), np.cos(phi)*np.cos(psi) + np.sin(phi)* np.sin(theta)*np.sin(psi), -np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)],
                  [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]])
    u = np.matmul(A, v[:, np.newaxis])[:, 0]

    return u


def refine_plane(n, seg, p):
    #p = wps[y]
    #start = time.perf_counter()
    slice, slice_pts = get_slice(v=n, p=p, mask=seg)
    orifice = segment_orifice_in_slice(slice)
    eigval, eigvec, pt_mean = get_eigen(orifice, slice_pts)
    area_best = eigval[0]*eigval[1]
    area_init = area_best
    v_best = n
    for angle_x in range(-20, 21, 10):
        for angle_y in range(-20, 21, 10):
            for angle_z in range(-20, 21, 10):
                v = rotate_3d_vector(n, angle_x, angle_y, angle_z)
                slice, slice_pts = get_slice(v=v, p=p, mask=seg)
                orifice = segment_orifice_in_slice(slice)
                eigval, eigvec, pt_mean = get_eigen(orifice, slice_pts)
                area = eigval[0]*eigval[1]
                if area < area_best:
                    area_best = area
                    v_best = v
    n1 = v_best
    area_init1 = area_best
    for angle_x in range(-10, 11, 5):
        for angle_y in range(-10, 11, 5):
            for angle_z in range(-10, 11, 5):
                v = rotate_3d_vector(n1, angle_x, angle_y, angle_z)
                slice, slice_pts = get_slice(v=v, p=p, mask=seg)
                orifice = segment_orifice_in_slice(slice)
                eigval, eigvec, pt_mean = get_eigen(orifice, slice_pts)
                area = eigval[0]*eigval[1]
                if area < area_best:
                    area_best = area
                    v_best = v
    n2 = v_best
    area_init2 = area_best
    for angle_x in range(-5, 6, 2):
        for angle_y in range(-5, 6, 2):
            for angle_z in range(-5, 6, 2):
                v = rotate_3d_vector(n1, angle_x, angle_y, angle_z)
                slice, slice_pts = get_slice(v=v, p=p, mask=seg)
                orifice = segment_orifice_in_slice(slice)
                eigval, eigvec, pt_mean = get_eigen(orifice, slice_pts)
                area = eigval[0] * eigval[1]
                if area < area_best:
                    area_best = area
                    v_best = v
    #print(time.perf_counter() - start)
    #print(area_init, area_init1, area_init2, area_best)

    return v_best


dicom_dir = 'F:\\LAA\\paper data 2020\\test_dicom'
seed = np.array([204, 296, 201])
threshold = 0.1

print('loading dicom...')
vol = load_dicom(dicom_dir)
xmin, xmax, ymin, ymax, zmin, zmax, X, Y, Z = crop_points(seed)

for i in range(2):
    denoise(vol)

print('precompiling functions...')
seg, geo = segment(vol, seed, threshold, seed[0]+3, seed[0], seed[1]+3, seed[1], seed[2]+3, seed[2])

timer_start = time.perf_counter()

seg, geo = segment(vol, seed, threshold, xmax, xmin, ymax, ymin, zmax, zmin)

wps, cds = cd_walk(seed, seg)

#cds = nimg.uniform_filter1d(cds, size=7)

y = test(cds, 290)
y = y[0]
print(wps[y, :])

n, d = get_plane_normal(cp=y, wp=wps)

start = time.perf_counter()
v_best = refine_plane(n=np.array([n[1], n[0], n[2]]), seg=seg, p=np.array([wps[y][1], wps[y][0], wps[y][2]]))

slice, slice_pts = get_slice(v=v_best, p=np.array([wps[y][1], wps[y][0], wps[y][2]]), mask=seg)
orifice = segment_orifice_in_slice(slice)
eigval, eigvec, pt_mean = get_eigen(orifice, slice_pts)
major, minor, maskover = get_axis_points(orifice, slice_pts, pt_mean, eigval, eigvec)

print(time.perf_counter() - start)
timer_stop = time.perf_counter()
print(timer_stop-timer_start)




















