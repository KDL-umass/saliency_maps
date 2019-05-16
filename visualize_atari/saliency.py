import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

searchlight = lambda I, mask: I*mask + gaussian_filter(I, sigma=3)*(1-mask) # choose an area NOT to blur
occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def run_through_model(model, obs, mode='actor'):
    _, value, _, _, a_logits = model.step(obs)
    return value if mode == 'critic' else a_logits

def score_frame(model, history, ix, r, d, interp_func, mode='actor'):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    assert mode in ['actor', 'critic'], 'mode must be either "actor" or "critic"'
    orig_obs = history['ins'][ix]
    L = run_through_model(model, orig_obs, mode=mode) #without mask

    scores = np.zeros((int(84/d)+1,int(84/d)+1)) # saliency scores S(t,i,j)
    for i in range(0,84,d):
        for j in range(0,84,d):
            processed_obs = np.copy(orig_obs)
            for f in [0,1,2,3]: #because atari passes 4 frames per round
                mask = get_mask(center=[i,j], size=[84,84], r=r)
                processed_obs[0,:,:,f] = interp_func(orig_obs[0,:,:,f], mask) # perturb input I -> I'
            l = run_through_model(model, processed_obs, mode=mode) #with mask
            scores[int(i/d),int(j/d)] = 0.5*np.sum(np.power((L-l),2)) #score is eq 2 from paper

    pmax = scores.max()
    scores = imresize(scores, size=[84,84], interp='bilinear').astype(np.float32)
    return pmax * scores / np.max(scores)

def saliency_on_atari_frame(saliency, atari, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = np.max(saliency)
    S = np.zeros((110, 84))
    S[18:102, :] = saliency
    S = imresize(saliency, size=[atari.shape[0],atari.shape[1]], interp='bilinear').astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= np.min(S)
    S = fudge_factor * pmax * S / np.max(S)

    I = atari.astype(np.uint16)
    I[:,:,channel] += S.astype(np.uint16)
    I = np.clip(I, 1., 255.).astype(np.uint8)
    return I

def get_env_meta(env_name):
    meta = {}
    if env_name=="Pong-v0":
        meta['critic_ff'] = 600 ; meta['actor_ff'] = 500
    elif env_name=="Breakout-v0" or env_name=="BreakoutToyboxNoFrameskip-v4":
        meta['critic_ff'] = 600 ; meta['actor_ff'] = 300
    elif env_name=="SpaceInvaders-v0":
        meta['critic_ff'] = 400 ; meta['actor_ff'] = 400
    elif env_name=="AmidarToyboxNoFrameskip-v4":
        meta['critic_ff'] = 600 ; meta['actor_ff'] = 300
    else:
        print('environment "{}" not supported'.format(env_name))
    return meta