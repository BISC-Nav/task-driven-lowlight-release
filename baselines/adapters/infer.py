"""Auditable official-network inference. No target image is accepted by this API."""
import argparse, hashlib, importlib.util, json, os, pathlib, sys, time
import numpy as np
from PIL import Image

BASE = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS = BASE.parent / 'weights'

def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def torch_model(method):
    import torch
    import torch.nn.functional as F
    import cv2
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if method == 'retinexformer':
        src = BASE/'third_party/Retinexformer'
        net = module_file('official_retinexformer', src/'basicsr/models/archs/RetinexFormer_arch.py').RetinexFormer(
            in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2])
        weight = WEIGHTS/'Retinexformer_LOL_v1.pth'
        state = torch.load(weight, map_location='cpu', weights_only=True)
        state = state['params'] if 'params' in state else state
        details = {'padding_multiple':4, 'GT_mean':False, 'self_ensemble':False}
    else:
        sys.path.insert(0, str(BASE/'third_party/SNR-Aware'))
        from models.archs.low_light_transformer import low_light_transformer
        net = low_light_transformer(nf=64, nframes=5, groups=8, front_RBs=1, back_RBs=1,
                                   predeblur=True, HR_in=True, w_TSA=True)
        weight = WEIGHTS/'SNR_LOLv1.pth'
        state = torch.load(weight, map_location='cpu', weights_only=True)
        details = {'internal_size':[400,608], 'preprocessing':'official test4, 5x5 box blur, SNR mask',
                   'temporal_input_frames':1, 'align_corners':False}
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    net.load_state_dict(state, strict=True)
    net.cuda().eval()
    def infer(rgb):
        x = torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda()
        h,w = rgb.shape[:2]
        with torch.inference_mode():
            if method == 'retinexformer':
                y = net(F.pad(x, (0,(-w)%4,0,(-h)%4), mode='reflect'))[:,:,:h,:w]
            else:
                nf = cv2.blur(rgb*255.0, (5,5))/255.0
                nf = torch.from_numpy(nf.copy()).permute(2,0,1)[None].cuda()
                dark = x[:,0:1]*0.299+x[:,1:2]*0.587+x[:,2:3]*0.114
                light = nf[:,0:1]*0.299+nf[:,1:2]*0.587+nf[:,2:3]*0.114
                mask = light/((dark-light).abs()+0.0001)
                mask = (mask/(mask.flatten(1).max(1).values[:,None,None,None]+0.0001)).clamp(0,1)
                y = net(F.interpolate(x,size=(400,608),mode='bilinear',align_corners=False),
                        F.interpolate(mask,size=(400,608),mode='bilinear',align_corners=False))
                y = F.interpolate(y,size=(h,w),mode='bilinear',align_corners=False)
        return y[0].permute(1,2,0).cpu().numpy()
    return infer, {'framework':torch.__version__, 'device':torch.cuda.get_device_name(),
                   'strict_weight_load':True, 'state_tensors':len(state),
                   'weights':str(weight), 'weight_sha256':sha(weight), **details}

def tf_model(method):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.reset_default_graph()
    src = BASE/'third_party'/('RetinexNet' if method=='retinexnet' else 'KinD')
    sys.path.insert(0,str(src))
    if method=='retinexnet':
        # Only the obsolete TensorFlow import is adapted; official functions are retained.
        source=(src/'model.py').read_text().replace('import tensorflow as tf', 'import tensorflow.compat.v1 as tf')
        ns={'__name__':'official_retinexnet', '__file__':str(src/'model.py')}
        exec(compile(source, str(src/'model.py'), 'exec'),ns)
        inp=tf.placeholder(tf.float32,[None,None,None,3])
        r,i=ns['DecomNet'](inp,layer_num=5)
        relit=ns['RelightNet'](i,r)
        output=r*relit
        groups=[('DecomNet',src/'model/Decom/RetinexNet-tensorflow'),
                ('RelightNet',src/'model/Relight/RetinexNet-tensorflow')]
    else:
        from skimage import color, filters
        ns=module_file('official_kind',src/'model.py')
        inp=tf.placeholder(tf.float32,[None,None,None,3])
        r,i=ns.DecomNet_simple(inp)
        rin=tf.placeholder(tf.float32,[None,None,None,3])
        iin=tf.placeholder(tf.float32,[None,None,None,1])
        ratio=tf.placeholder(tf.float32,[None,None,None,1])
        restored=ns.Restoration_net(rin,iin)
        adjusted=ns.Illumination_adjust_net(iin,ratio)
        groups=[('DecomNet',WEIGHTS/'KinD/checkpoint/decom_net_train/model.ckpt'),
                ('Illumination_adjust_net',WEIGHTS/'KinD/checkpoint/illumination_adjust_net_train/model.ckpt'),
                ('Restoration_net',WEIGHTS/'KinD/checkpoint/Restoration_net_train/model.ckpt')]
    sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8,inter_op_parallelism_threads=2,
                                          device_count={'GPU':0}))
    loaded=[]
    for group,weight in groups:
        variables=[v for v in tf.trainable_variables() if v.name.startswith(group+'/')]
        disk=dict(tf.train.list_variables(str(weight)))
        assert variables
        for v in variables:
            key=v.name.split(':')[0]
            assert key in disk and disk[key]==v.shape.as_list(), (key,disk.get(key),v.shape)
        tf.train.Saver(var_list=variables).restore(sess,str(weight))
        loaded.append({'group':group,'variables':len(variables),'weight':str(weight),
                       'data_sha256':sha(str(weight)+'.data-00000-of-00001')})
    assert not len(sess.run(tf.report_uninitialized_variables()))
    def infer(rgb):
        h,w=rgb.shape[:2]
        if method=='retinexnet':return sess.run(output,{inp:rgb[None]})[0]
        x=np.pad(rgb,((0,(-h)%16),(0,(-w)%16),(0,0)),mode='reflect')[None]
        rd,il=sess.run([r,i],{inp:x})
        rr=sess.run(restored,{rin:rd,iin:il})
        ia=sess.run(adjusted,{iin:il,ratio:np.full_like(il,5.0)})
        smooth=filters.gaussian(color.rgb2gray(rd[0]),3)
        suppression=np.minimum((smooth*2)**0.5,1)[None,:,:,None]
        fusion=il*x+(1-il)*(rr*suppression*ia)
        return fusion[0,:h,:w]
    return infer,{'framework':tf.__version__,'device':'CPU','strict_weight_load':True,'weights':loaded,
                  'illumination_ratio':5.0 if method=='kind' else None,'GT_required':False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',required=True,choices=['retinexformer','snr','retinexnet','kind'])
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    files=sorted(pathlib.Path(args.input).glob('*.png'))
    if args.limit:files=files[:args.limit]
    assert files,'No input PNG files'
    infer,info=(torch_model if args.method in ['retinexformer','snr'] else tf_model)(args.method)
    first=np.asarray(Image.open(files[0]).convert('RGB'),dtype=np.float32)/255.0
    t=time.perf_counter();warm=infer(first);cold=time.perf_counter()-t
    repeat=infer(first)
    info.update({'method':args.method,'input':args.input,'output':args.output,'count':len(files),
                 'input_shape':list(first.shape),'warmup_seconds':cold,
                 'same_input_repeat_max_abs':float(np.max(np.abs(warm-repeat))),
                 'test_target_used':False,'status':'running','quantization':'round_clip_RGB_uint8'})
    timings=[];checks=[]
    for j,path in enumerate(files):
        x=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255.0
        t=time.perf_counter();y=infer(x);elapsed=time.perf_counter()-t
        assert y.shape==x.shape and np.isfinite(y).all(),(path,y.shape)
        image=np.rint(np.clip(y,0,1)*255).astype(np.uint8)
        dest=out/path.name;Image.fromarray(image).save(dest)
        timings.append(elapsed)
        checks.append({'name':path.name,'input_sha256':sha(path),'output_sha256':sha(dest),
                       'raw_min':float(y.min()),'raw_max':float(y.max()),'seconds':elapsed})
        if j%16==0:print(json.dumps({'method':args.method,'frame':j,'seconds':elapsed}),flush=True)
    info.update({'status':'passed','inference_mean_seconds':float(np.mean(timings)),
                 'inference_p50_seconds':float(np.median(timings)),'files':checks})
    (out/'inference_audit.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:v for k,v in info.items() if k!='files'}),flush=True)

if __name__=='__main__':main()
