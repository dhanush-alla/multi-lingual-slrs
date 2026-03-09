import numpy as np, os
for prefix in ['asl_train','asl_test','isl']:
    path=os.path.join('data','processed',f'{prefix}_landmarks.npy')
    if os.path.exists(path):
        arr=np.load(path)
        print(prefix, arr.shape, arr.dtype, arr.nbytes)
    else:
        print(prefix, 'missing')
