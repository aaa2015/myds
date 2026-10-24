#!/usr/bin/env python3
"""Independently inspect MoE workspace full-model raw evidence, on CPU only.

No runner analysis routine, inference binary, model execution, Metal API, or
service operation is invoked. A tiny temporary C library accelerates byte FNV;
its only entry point hashes a supplied memory span. Archives are never extracted.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tarfile
import tempfile

FLAG='MYDS_PREFILL_MOE_WORKSPACE'
PRODUCTION_BINARY='29058d822c95d86ffdc6e84798fe87093d7f8c094b1002f1250177b12066d72b'
PRODUCTION_LIBRARY='c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81'
IDENTITY_KEYS=('argv_raw_hex','myds_env_raw_hex','cwd','log','nice','binary','metallib',
               'binary_sha256','metallib_sha256','config','config_sha256','effective_myds_env')
SCALAR_ORDER=['ctx_pos','rng_state','max_seq_len','max_new_tokens','seed_u32','kv_capacity',
              'pcache_len','anchor_len','continue_from_cache','v41_valid','next_position',
              'engram_position','snapshot_allocated_bytes','prefill_capacity','low_kv_enabled']




# Independently proven exact868 production inputs; custom diagnostic files are additional.
GOLD_SOURCE_HASHES={'Makefile': '2292a6f4976d4e6adab1034cfda0710a0cccbbb632bf98b5dce76d1e2b6dbcc7',
 'include/myds.h': 'd73af31a68e379d1651239791d80538fd7191b4cf9072bfd98a84d834a64f6c5',
 'include/myds_align.h': '27db712ed10840fa27a2703dcd14c390f31cd0f666abb51db04e6f21d2a62476',
 'include/myds_configfile.h': 'b467e58f982c6dbeee12c14c55912716bc69d18e17e7596efeef4e7f291d27a3',
 'include/myds_cpu.h': '31d872795cc2ef1713f1a51e43d16c02cd3889b03ee8cc4ab440dc41e87fe7d0',
 'include/myds_dspark.h': '4e39abe8cbb8744a0f38a0975e6ce993c3f5f3f6945dfcc76029fd14a44d460f',
 'include/myds_engram.h': '57ee56c41399c41e255f9c232b1369579d9efd6bb771db15f6272a4d8717fb21',
 'include/myds_gguf.h': '8c8cd1d4de303725d8483c7085a433b84df0b48818a6ebf175dba8d27555cd7a',
 'include/myds_internal.h': 'b810de8a2c3e8a9ed81711728994e71bdca9695da33f2eb4552de67192607ac1',
 'include/myds_kvdsk.h': 'fc425edf81333e2308fa2a1cdb259aef7fe3ddbd609174ee4c3da04cae3a461c',
 'include/myds_log.h': '96da7e04b593b5a17ac4dfa6ba0fc694fb11928a167e3f7690e74228a71faa9e',
 'include/myds_lowkv_policy.h': 'def5dc1dec946fc06af0fa04338be57f42baad28e2ee65567d73f3682e282a6c',
 'include/myds_metal.h': 'ecb5191c4f5de1038d242e97e10732a3411102fc779c05e7ca2886ccc20e9ab1',
 'include/myds_moe_cpu.h': 'b4087827a191de2e4ae03d3a9682e58e1f5d92b941c2cbbebaf0c1f1a9dfddc4',
 'include/myds_pld.h': 'd5900389894ee34ab3189bd08a2eb38f284c0b6dae4060600fed71c73c4353b5',
 'include/myds_server.h': '2e5fa57fead8a0c92258683da2ae983e935425f1aee491492d4e29ff183cec40',
 'include/myds_server_internal.h': '133ab3b9c7cf13a2b2258e688ed3861eebaa73db25833b2d33780b8443ff73b8',
 'include/myds_snapshot_slots.h': '14616746226747af04c5cb65940d16af880f25888e6af6093fe6dfa5620a8f31',
 'include/myds_tokenizer.h': '292cc9c3ce212cdeb08d9beb4f92a84660b248e8c677c04790b93229e26b451c',
 'include/myds_v41.h': '6389d3aeaf3c12f35721df5653f03b526cac2796b85346e0d59efedf2a392bec',
 'include/myds_v41_decode.h': '616fffa71022b9342db70d3df54ffab93037c35e58a3441c42e577873e217348',
 'include/myds_v41_gpu.h': '17ab8f4c00662b0661af5ea0e0fb11b7abc4119df13dab77d098e71c046ff373',
 'include/myds_v41_moe.h': 'ac7fdc9cdeedd057452c277833fd8917d8d6ca00f68e7f93048cea71e2728aed',
 'include/myds_visfwd.h': 'e188c8e488591be257ec1170331b78f01b6db4132ded5a7938ef1c564d998d13',
 'include/myds_vispre.h': 'addb28dd5286289001717abe81b7995fa6599f5a69e65dd558019c12ce7711be',
 'include/myds_visw.h': 'd8425832a8d1425c7e0cfa13fc9c2ecca2e94e8fbc46e881ec03791617b0affa',
 'include/myds_vjob.h': '7710ea311702951206f183d762d12b89b867cf72b21ffcd98420565818b449b3',
 'metal/argsort.metal': '7bcdcb48366f71cb9adf6fdb27429b614eeab24821947a6710e2a65b3f88cc17',
 'metal/bin.metal': '66f7c42264b1cfb7b1e42380c94093f963dc1f0792f43e79b5588892f882d5b6',
 'metal/concat.metal': '04fb0fc2a3b51015444e330290b979a56e4cac82aee523afa2b42e361ab23da7',
 'metal/cpy.metal': 'df8cabe88c33076c16a54cfb88a00133b68ea5cbfc1742ac79608c84d4456cd4',
 'metal/dense.metal': 'efdc3e24cf13b6d8651f4dd4a728a453f2a4d1b2295cef41135516ea628bed55',
 'metal/dsv41.metal': '7a15599dc6d329715c610b581b2babcce46580187707b8d87565e795b41ed1a2',
 'metal/dsv4_hc.metal': '7f5e44ce6108285740cca97a4c81e8c271d0631ed784f459baa63b3d01260bfc',
 'metal/dsv4_kv.metal': '87334bd001a306dcdf8e344f7f47938d67d6887889a3da7282cba0d25b36d172',
 'metal/dsv4_misc.metal': '14dfce00954e374f1da677a174ae92c0c7bacbd46a5266a7665d736833252531',
 'metal/dsv4_rope.metal': 'a6e86dcf24d35a0014c92b7542c7c2ba8b3815c853f5d636da94cb1df1c05ccd',
 'metal/flash_attn.metal': '503fc1ec7fb94130f4e44c1a1147bf182d11c9993cc0688e8ed2f4993df7bdc6',
 'metal/generate_mxfp4_half_lut.py': 'a20b9b79d993b16b3bcb4a9658c86618ce4cd33739838fe6aa47bf947f42194e',
 'metal/get_rows.metal': 'bd3b352a6ffb138ae5e391db97b211f8386d417365eac2caa3730f55a075aa3a',
 'metal/glu.metal': '6d226a91566d625b3a4b141ecadcb4d796bdc02de19211d891e215c530c812ca',
 'metal/moe.metal': '78cb927e7884254832f3c37e97b1c4952c1a1863555616693f3b35fb787f5776',
 'metal/myds_chunk.metal': '2cdfbd0c96b47cfa81c26ca6b2194874f42bd766b8c02efacd86f15d2808bda1',
 'metal/myds_iq2_v41.metal': 'cbd703565fb9715c7a611e8e5ce15a81c9289cc7fbeedd3ca285ae8f5ba26aa5',
 'metal/myds_q4_v41.metal': '6487878f399cd24b3e3c357cc161a10944a558465f2d0eb680251418a4ab22d0',
 'metal/myds_v41_moe.metal': 'd731dfbe42b6ce94aabd70a5bc4472b4168b2913d4ce82c0d84982a2fd75c0e4',
 'metal/norm.metal': 'fd89688ae3ac8a473f731da0051121239bff2c6767674d091832f1e5cfd698c8',
 'metal/probe_attention_register.metal': '0bfd2bd9b52cf5bb1f54e12a7bca5721be0a92ca0dabd70b3023b8ef2248b13b',
 'metal/probe_dequant_cost.metal': '6ece56ba63f6f1d409a7be14f5897639bf203fd9c89672fb978126222596fbab',
 'metal/probe_q2_down.metal': '1883cbb1710ebdf270220f0f8e100452085f41c2d261f9758872a7b849bde791',
 'metal/probe_q4_sumy.metal': '4934774b49944258c9a729d41c1baf3a0c2abfe7b7d531c9994928d65cc775fa',
 'metal/probe_v41_route.metal': '92c715cd2436ccfb39125d8bac91a1e6a6983889e66a8aa6d4b84ec6aa1f6436',
 'metal/pureread.metal': '3c740ef1ad065e617c42e82c8b5f7f3c02a1011d5977b9d88aa62ee445a8117a',
 'metal/repeat.metal': '2c6142b47a2007d246f5eb6e7395047627bb6dfa02c649311d16620da35007cd',
 'metal/set_rows.metal': '25f597bf5f5fb5a3791efc68e3182539727609f09c45750ef9376ed430d7a029',
 'metal/softmax.metal': '6eaa6ce78c4b221b3c7fe45647d6eb42f165201992a243d77d04cf2668fb7664',
 'metal/sum_rows.metal': '7e1f3ad3f945b1f6f59aa2d8ee5bf464052d989a632c33ea95ce50d3c4fbf0fc',
 'metal/topk_threshold.metal': '8fb93312ccf29f934668954a202244df60e5e7cfd62680c7c74137161a369fc0',
 'metal/unary.metal': '4607dbea21501230a193f8a333b1bd543d566f405568a094d6faf2c17cb0c5cf',
 'metal/visenc.metal': 'dced00348fbf19e3d1d6c9e4bb39cd457114819bdd576468d3a1dbf22d807165',
 'src/main.c': 'e376ed9ef8b89d449c743d6af5d2d60221ace33fa41bf16fbd3ad77cbd50d942',
 'src/myds_configfile.c': 'a2dfc690288cb56e37afd4572d8c8739cba3f995faa59a78a1482eb131e67f56',
 'src/myds_cpu.c': 'd88b99249fc1a4b40c3f5630b4e7b780d4ff13eb671996401cded90525ece7fe',
 'src/myds_decode_candidates.inc': 'a11bc77045a1e01447e2c2fa67f0c5e743929b7f7db899f380e8322d7db6a286',
 'src/myds_dspark.c': '6acea67bea57390cf670695acfb00e4c39f4c2651bdfcaf0c32a3358187f6547',
 'src/myds_engine.c': '47caa3321c221d3bf7b786fa6d7fee29bafac25f402f4217f0eeefb17f6f31a9',
 'src/myds_engram.c': '54ef3cad362c9b0c49c03be6590bc6b2ca90aee5707fc3391e5acf75881ab885',
 'src/myds_engram_cpu.c': '11d143161f9f0a989b88abe666e733f67cf9b2fbe632e6733eebdab5e090cc20',
 'src/myds_gguf.c': '63c5ca489d8f9d1b87bd36c36c22f09f47498f904925aaa1b9ca982ad0a2ae6c',
 'src/myds_iq2_tables.h': 'fd92b447b6c38445129862f080162e6dcdb6caa1d4f76646f482d515a028cd86',
 'src/myds_kvdsk.c': '2221cd4ae07634fcc011a78c6b0185defe62ac282b233438461012723207ed56',
 'src/myds_log.c': 'f6eff9e828fc882e3d5f0ed7ccd033c0bcd91fefb345fe8634ef8512179082b5',
 'src/myds_metal.m': 'a36c6eb122f415e924037b402a3c78934dbc86f05538469adc604412b6a75a31',
 'src/myds_moe_cpu.c': '81902164f9a60d79aac8ce7f609ed4ad35263b6b5b0f72d063931b2c906cc937',
 'src/myds_server.c': '5aa609e8a0845c313444d062e93c38d04520c5f21bb7fe0e2ea552481b365a5a',
 'src/myds_tokenizer.c': '808c432023f5fa2534a4464fbbc7816bb54d86ac44921ba9b7ec2d45e5fcf144',
 'src/myds_tokenizer_unicode.h': 'bfb6f1a5149a813f18212f6216ce10468b962f7e7bd3e88329ef823de1854808',
 'src/myds_v41.c': '1a4614a71f3b425fc2181f7892bdd6a2118ea79e97183b58ed0e05488758601a',
 'src/myds_v41_decode.c': '9f9d89f84e83e54cda3d718aa748e3311af06fd8a7f1002918cae0de959829ae',
 'src/myds_v41_gpu.c': 'a4f34a543e8836212aacebe0a05bbdf4f5fbc40719e4d8efea24067a557e6f10',
 'src/myds_v41_moe.c': 'bfed3b37525bf0d7eeb2146a172a3072ef3ab44e1fd42631fa3f8c616ef5b9e2',
 'src/myds_v41_shared3_gu_shadow.inc': 'b33bb37bb19f18ee870e1d1eef6a6bbd4d649c7c02a4d8894e20277f25f6d713',
 'src/myds_v41_shared3_routed_shadow.inc': '9fb29d7128bb62f4b8d3945b2c54851c6b60de665036f98a52d23d274d76726a',
 'src/myds_visfwd.c': 'b9014c5f3248d524800c3cd6e67907e5dc858a27806a922aa0c6d71879800a95',
 'src/myds_vispre.c': 'bbd6e6caf1c5728c89420c080334917018fcbf2f7a0ca54d6ed404cee7426a0e',
 'src/myds_visw.c': '3f3b138ff904376c804c71e04f4ca2317b706394950442202bdd8c0604cf8089',
 'src/myds_vjob.c': '5acb00588396128b34e922d595b286d007e66bb363837d6bb886b62a13c50807',
 'third_party/iris/LICENSE': '663e51edcd9a3a8f858018ce6924ad2895a24c269c26c39ad75325bcd0f7e5bc',
 'third_party/iris/jpeg.h': '9a1d13b3fc3dfbe3af76ab35b650bfe9a54b418b254bf4bd9c7ed958b0c98403',
 'third_party/iris/png.h': 'b87dda8580af164c443e76ca073d77f1340e1a7584b73a8449a0f5bf14cd5018',
 'tools/build_metallib.py': 'd18dd1fd37cfac24fbd87db9d6ae08df3a1738e313aa627a327f0cb3e72365ce'}

def need(value,message):
    if not value: raise ValueError(message)


def load(path):
    def object_pairs(pairs):
        result={}
        for key,value in pairs:
            need(key not in result,'Duplicate JSON key: '+key);result[key]=value
        return result
    return json.loads(path.read_text(),object_pairs_hook=object_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def records(path):
    result=[]
    for line in path.read_text().splitlines():
        if line.strip():
            result.append(json.loads(line,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))
    return result


def sha(path,limit=None):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while limit is None or limit:
            block=stream.read(min(1048576,limit) if limit is not None else 1048576)
            if not block:break
            value.update(block)
            if limit is not None:limit-=len(block)
    return value.hexdigest()


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def close(a,b,tolerance=1e-8):return finite(a) and finite(b) and abs(a-b)<=tolerance


def actual_environment(actual,expected):
    need({k:v for k,v in actual.items() if v is not None}==expected,
         'Actual complete MYDS environment differs from frozen production plus explicit flag/sinks')


class ByteHashes:
    def __init__(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='myds-fullmodel-cpu-fnv-')
        root=Path(self.temporary.name);source=root/'fnv.c';library=root/('fnv.dylib' if sys.platform=='darwin' else 'fnv.so')
        source.write_text('#include <stdint.h>\n#include <stddef.h>\n'
          'uint64_t hash_span(uint64_t h,const unsigned char *p,size_t n){'
          'for(size_t i=0;i<n;++i){h^=p[i];h*=UINT64_C(1099511628211);}return h;}\n')
        command=['cc','-O3','-shared','-fPIC',str(source),'-o',str(library)]
        done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        need(done.returncode==0,'CPU byte hashing helper compilation failed: '+done.stderr)
        self.library=ctypes.CDLL(str(library));self.function=self.library.hash_span
        self.function.argtypes=[ctypes.c_uint64,ctypes.c_char_p,ctypes.c_size_t]
        self.function.restype=ctypes.c_uint64;self.cache={}

    def file(self,path):
        st=path.stat();key=(str(path.resolve()),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in self.cache:
            digest=hashlib.sha256();value=14695981039346656037
            with path.open('rb') as stream:
                while True:
                    block=stream.read(1048576)
                    if not block:break
                    digest.update(block);value=self.function(value,block,len(block))
            self.cache[key]=dict(bytes=st.st_size,sha256=digest.hexdigest(),fnv1a64_bytes=format(value,'016x'))
        return self.cache[key]

    def bytes(self,data):return format(self.function(14695981039346656037,data,len(data)),'016x')
    def close(self):self.temporary.cleanup()


def bounded(directory,name):
    need(type(name) is str and name and Path(name).name==name,'Nonflat raw payload path')
    path=(directory/name).resolve(strict=True)
    need(path.parent==directory.resolve() and path.is_file(),'Raw payload escaped its run directory')
    return path


def file_difference(left,right,f32=False):
    first=None;different=0;total=0;float_differences=0
    with left.open('rb') as a,right.open('rb') as b:
        while True:
            x,y=a.read(1048576),b.read(1048576)
            if not x and not y:break
            if x!=y:
                for offset,(u,v) in enumerate(zip(x,y)):
                    if u!=v:
                        if first is None:first=total+offset
                        different+=1
                different+=abs(len(x)-len(y))
                if first is None:first=total+min(len(x),len(y))
                if f32:
                    need(len(x)%4==len(y)%4==0,'Misaligned raw logits')
                    float_differences+=sum(u!=v for u,v in zip(struct.iter_unpack('<I',x),struct.iter_unpack('<I',y)))
                    float_differences+=abs(len(x)-len(y))//4
            total+=max(len(x),len(y))
    return dict(equal=different==0,different_bytes=different,first_different_byte=first,
                f32_bit_mismatches=float_differences if f32 else None)


def state_payload(directory,sample,hashes):
    manifest=load(directory/'manifest.json');metadata=manifest['metadata']
    need(manifest['schema_version']==1 and manifest['scope']=='fresh_context_full_prefill_final_state' and
         manifest['run']==sample['run'] and manifest['complete'] is True,'Incomplete raw state manifest')
    need(metadata['fresh_context'] is True and metadata['byte_order']=='little' and
         metadata['layers']==40 and metadata['window_width']==512 and metadata['window_order']=='chronological_active_rows' and
         metadata['compressed_row_bytes']==1024 and metadata['index_row_bytes']==256 and
         metadata['scalar_order']==SCALAR_ORDER,'Raw state metadata contract differs')
    scalars=struct.unpack('<15Q',bounded(directory,'context-scalars.u64le').read_bytes())
    need(scalars[2:5]==(327680,32,42) and scalars[8] in (0,1) and scalars[9]==1 and scalars[14] in (0,1),
         'Raw context configuration/valid state differs')
    next_pos=scalars[10]
    need(0<next_pos<=327680 and metadata['ctx_pos']==scalars[0] and metadata['next_position']==next_pos and
         metadata['window_rows']==min(next_pos,128),'Raw active state extent differs')
    tail=list(struct.unpack('<3I',bounded(directory,'engram-tail.u32le').read_bytes()))
    need(metadata['rng_state']==format(scalars[1],'016x') and metadata['engram_position']==scalars[11] and
         metadata['engram_tail']==tail,'Raw RNG/Engram metadata differs')
    expected={'context-scalars.u64le':120,'engram-tail.u32le':12,'snapshot-positions.u32le':352,
              'prefix-tokens.i32le':scalars[6]*4,'anchor-tokens.i32le':scalars[7]*4,
              'tokens.i32le':128,'output.bin':sample['output_bytes']}
    for layer in range(40):expected[f'layer-{layer:02d}-window-active.f32']=min(next_pos,128)*2048
    for owner in range(4):
        if owner<3:
            for kind in ('kv','score'):expected[f'owner-{owner}-previous-{kind}.f32']=2048
        count=next_pos//(1 if owner==3 else 2)
        expected[f'owner-{owner}-compressed-active.bf16']=count*1024
        expected[f'owner-{owner}-index-active.bf16']=count*256
    positions=bounded(directory,'snapshot-positions.u32le').read_bytes()
    need(len(positions)==352,'44 published/staged snapshot pairs required')
    published=[]
    for slot,(position,staged) in enumerate(struct.iter_unpack('<II',positions)):
        need(position<=next_pos and staged<=next_pos and not(position and staged),'Invalid published/staged snapshot pair')
        if position:
            published.append(dict(slot=slot,position=position))
            expected[f'snapshot-{slot:02d}-engram-tail.u32le']=12
            for layer in range(40):expected[f'snapshot-{slot:02d}-layer-{layer:02d}-window-active.f32']=min(position,128)*2048
            for owner in range(3):
                for kind in ('kv','score'):expected[f'snapshot-{slot:02d}-owner-{owner}-previous-{kind}.f32']=2048
    items=manifest['files'];names=[item['path'] for item in items]
    need(len(set(names))==len(names)==manifest['file_count'] and set(names)==set(expected),
         'Exact complete live/published state file coverage required')
    need(set(p.name for p in directory.iterdir())==set(expected)|{'manifest.json'},'Unrecorded raw state files')
    files={}
    for item in items:
        name=item['path'];path=bounded(directory,name);actual=hashes.file(path)
        need(actual['bytes']==item['bytes']==expected[name] and actual['fnv1a64_bytes']==item['fnv1a64_bytes'],
             'Raw state extent/FNV differs: '+str(path))
        files[name]=actual
    token_bytes=bounded(directory,'tokens.i32le').read_bytes()
    need(len(sample['token_ids'])==32 and token_bytes==struct.pack('<32i',*sample['token_ids']),
         'Complete raw token ID bytes differ')
    need(files['output.bin']['fnv1a64_bytes']==sample['output_bytes_fnv1a64'],'Raw original text bytes differ from callback FNV')
    if 'token_ids_fnv1a64' in sample:need(hashes.bytes(token_bytes)==sample['token_ids_fnv1a64'],'Native token FNV differs')
    need(len(manifest['limitations'])==2 and any('Historical published snapshot Engram tails use an added pure read-only getter.' in x for x in manifest['limitations']) and
         any('allocator padding' in x for x in manifest['limitations']),'Export boundary/limitations missing')
    return dict(metadata=metadata,files=files,published_snapshots=published,limitations=manifest['limitations'])



def workspace_stats(value,setting,prompt_tokens,gold=False):
    need(type(value) is dict and value['mode']==setting and value['capacity']==2048,'Workspace mode/capacity differs')
    B=2048*min(8,(prompt_tokens-1+2047)//2048)
    need(value['moe_batch']==B,'Actual macroblock-sized MoE capacity differs')
    fields=('legacy_gate_bytes','legacy_up_bytes','legacy_mid_bytes')
    if not setting:
        need(all(value[k]==B*6*2304*4 for k in fields) and value['scratch_owner_bytes']==3*B*6*2304*4,
             'Native scratch must contain three complete B-sized F32 buffers')
    if gold:
        need(setting==0 and value.get('provider_counters_available') is False,'Gold must use original source with only pure diagnostic getters')
        return value
    need(value['failures']==0 and value['requests']==value['mm_encoded']+value['mv_encoded'] and
         value['mm_encoded']>0,'Complete actual MoE host encoding counters required')
    if setting:
        arena=value['arena'];need(all(value[k]==0 for k in fields),'Compact path retained legacy F32 pools')
        need(arena['mm_capacity']==B and arena['mm_mid_bytes']==B*6*2304*2 and
             arena['mm_encoded']>0 and arena['failures']==0 and
             arena['requests']==arena['mm_encoded']+arena['mv_encoded'] and
             arena['requests']==value['requests'] and arena['mm_encoded']==value['mm_encoded'] and
             arena['mv_encoded']==value['mv_encoded'],'Actual compact arena/provider differs')
        F=arena['f32_capacity'];need(type(F) is int and 0<=F<=B,'Fallback capacity outside arena')
        need(all(arena[k]==F*6*2304*4 for k in ('f32_gate_bytes','f32_up_bytes','f32_mid_bytes')),
             'Full fallback pool extent differs')
        total=sum(arena[k] for k in ('mm_mid_bytes','f32_gate_bytes','f32_up_bytes','f32_mid_bytes','retired_bytes'))
        need(arena['allocated_bytes']==value['scratch_owner_bytes']==total and
             arena['peak_allocated_bytes']>=total and value['workspace_owner_bytes']>=total,
             'Logical requested owner byte accounting differs')
    return value


def verify_sample(sample,directory,logits,setting,prompt_tokens,hashes,gold=False):
    need(sample['generation_ok'] is True and sample['sample_valid'] is True and sample['fresh_context'] is True and
         sample['output_tokens']==32 and sample['prompt_tokens']==prompt_tokens and sample['cached_tokens']==0 and
         sample['stop_reason']=='length' and sample['warmup']==(sample['run']==0),'Incomplete fresh-context sample')
    for key in ('decode_s','prefill_s','wall_s','first_callback_s','last_callback_s','decode_ms_per_token'):
        need(finite(sample[key]) and sample[key]>0,'Invalid raw timing: '+key)
    need(sample['decode_intervals']==31 and sample['prefill_tokens']==prompt_tokens-1 and
         close(sample['decode_s'],sample['last_callback_s']-sample['first_callback_s'],3e-9) and
         close(sample['decode_ms_per_token'],sample['decode_s']*1000/31) and
         sample['last_callback_s']<=sample['wall_s']+1e-8,'Callback/prefill timing arithmetic differs')
    intervals=sample['callback_intervals_ms']
    need(len(intervals)==31 and all(finite(v) and v>=0 for v in intervals) and
         close(sum(intervals),sample['decode_s']*1000,1e-5),'Complete callback intervals disagree with decode interval')
    raw=logits.read_bytes();actual=hashes.file(logits)
    need(len(raw)==129280*4 and sample['final_logits']['elements']==129280 and sample['final_logits']['finite'] is True and
         all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw)) and
         actual['fnv1a64_bytes']==sample['final_logits']['fnv1a64_bytes'],'Full finite raw vocabulary logits/FNV differ')
    candidate=workspace_stats(sample['moe_workspace'],setting,prompt_tokens,gold)
    providers={}
    for key in ('gpu_attention_packed_register','gpu_attention_rope_fuse'):
        value=sample[key]
        need(value['requests']==value['encoded']>0 and value['failures']==0 and
             value['decode_requests']==value['decode_encoded']>0 and value['decode_failures']==0,
             'Actual production provider failed: '+key);providers[key]=value
    state=state_payload(directory,sample,hashes)
    return dict(run=sample['run'],warmup=sample['warmup'],decode_s=sample['decode_s'],prefill_s=sample['prefill_s'],
        wall_s=sample['wall_s'],callback_interval_sum_ms=sum(intervals),logits=actual,state=state,
        candidate=candidate,production_providers=providers,original_text_path=str(directory/'output.bin'),
        original_logits_path=str(logits),state_path=str(directory))



def memory_review(data,trace_path,repeats):
    raw=records(trace_path);need(raw,'Missing raw process/device memory trace')
    for row in raw:
        need(row['kind']=='memory_sample' and finite(row['monotonic_s']) and row['run'] in range(repeats+1) and
             row['phase'] in (0,1,2,3),'Invalid memory sample identity')
        v=row['value'];need(v['task_valid'] is True and v['device_valid'] is True and
            all(type(v[k]) is int and v[k]>=0 for k in ('rss_bytes','phys_footprint_bytes','device_allocated_bytes')),
            'Invalid raw memory measurement')
    need(all(a['monotonic_s']<=z['monotonic_s'] for a,z in zip(raw,raw[1:])),'Nonmonotonic memory trace')
    reports=[r for r in data if r.get('kind')=='memory_phase'];result=[]
    for run in range(repeats+1):
        phases=[r for r in reports if r['run']==run]
        need([(r['phase'],r['label']) for r in phases]==[(1,'context_create'),(2,'generation'),(3,'raw_export')],
             'Memory phase coverage/order differs')
        for report in phases:
            rows=[r for r in raw if r['run']==run and r['phase']==report['phase']]
            need(report['sampling_ok'] is True and report['interval_ms']==10 and
                 report['observed_samples']==len(rows)>0,'Incomplete raw memory phase observations')
            for kind,key in [('phys_footprint','phys_footprint_bytes'),('rss','rss_bytes'),('device_allocated','device_allocated_bytes')]:
                need(report['observed_peak_'+kind+'_bytes']==max(r['value'][key] for r in rows),'Raw memory phase peak differs')
        generation=phases[1];generation_rows=[r for r in raw if r['run']==run and r['phase']==2]
        need(len(generation_rows)>1,'Insufficient actual generation memory observations')
        gaps=[z['monotonic_s']-a['monotonic_s'] for a,z in zip(generation_rows,generation_rows[1:])]
        result.append(dict(generation,raw_generation_samples=len(generation_rows),
            observed_max_sample_gap_ms=max(gaps)*1000 if gaps else 0))
    return dict(trace_sha256=sha(trace_path),raw_samples=len(raw),generation=result,
        limitation='10ms target interval plus synchronous boundary samples; reported peaks are sampled observations, not exact physical maxima.')



def arm(root,step,plan,environment,hashes):
    label=step['name'];phase,workload,name=label.split('-');setting=int(name in ('on','B'))
    repeats=1 if phase=='correctness' else 3;data=records(root/(label+'.stdout'))
    metadata=[r for r in data if r.get('kind')=='metadata'];loaded=[r for r in data if r.get('kind')=='load']
    samples=[r for r in data if r.get('kind')=='sample']
    need(len(metadata)==len(loaded)==1 and all(r.get('kind') in ('metadata','load','sample','memory_phase') for r in data),
         'Unexpected actual harness records')
    meta,library=metadata[0],loaded[0];directory=root/label
    env=load(directory/'environment.json');expected=dict(environment)
    expected.update({FLAG:str(setting),**{'MYDS_BENCH_'+k+'_DIR':str(directory/k.lower()) for k in ('LOGITS','STATE','MEMORY')}})
    need(env==expected,'Requested per-arm environment differs');actual_environment(meta['environment'],expected)
    need(all(meta.get(k)==v for k,v in dict(protocol_version=4,repeats=repeats,warmup_runs=1,max_new_tokens=32,
        max_seq_len=327680,temperature=0,top_p=1,top_k=0,min_p=0,repetition_penalty=1,seed=42,
        clock='CLOCK_MONOTONIC',transport='direct_byte_callback',decode_basis='first_to_last_callback/(tokens-1)',
        prompt_renderer='parse_chat_prompt').items()),'Actual harness protocol differs')
    argv=step['argv'];need(len(argv)==6 and argv[-3:]==[str(repeats),'32','327680'],'Actual harness argv differs')
    stage=Path(argv[0]).parent.parent
    need(Path(argv[0]).name=='bench_moe_workspace_fullmodel' and Path(meta['request_file'])==Path(argv[2]) and
         Path(meta['model'])==Path(argv[1]),'Model/request/harness argv binding differs')
    # Actual source path is retained as evidence; an offline replay need not retain the stage itself.
    need(Path(library['actual_metallib_path']).resolve()==(stage/'bin/myds.metallib').resolve() and
         library['fresh_context_per_sample'] is True and library['cache_configuration_preserved'] is True and
         library['moe_workspace_mode']==setting,'Actual provider/library identity differs')
    need([r['run'] for r in samples]==list(range(len(samples))) and 0<len(samples)<=repeats+1,'Actual sample sequence differs')
    reviewed=[verify_sample(sample,directory/'state'/f"run-{sample['run']}",
        directory/'logits'/f"run-{sample['run']}.f32",setting,plan['fullmodel']['requests'][workload]['expected_prompt_tokens'],hashes,name=='gold') for sample in samples]
    memory=memory_review(data,directory/'memory/samples.jsonl',repeats)
    return dict(label=label,phase=phase,workload=workload,name=name,setting=setting,stage=str(stage),
                returncode=step['returncode'],complete=len(samples)==repeats+1,metadata=meta,load=library,samples=reviewed,memory=memory)


def compare_sample(gold,candidate):
    differences=[]
    if gold['state']['metadata']!=candidate['state']['metadata']:differences.append(dict(kind='state_metadata'))
    for name in sorted(set(gold['state']['files'])|set(candidate['state']['files'])):
        a,b=Path(gold['state_path'])/name,Path(candidate['state_path'])/name
        if not a.is_file() or not b.is_file():differences.append(dict(kind='state_file_coverage',path=name));continue
        delta=file_difference(a,b)
        if not delta['equal']:differences.append(dict(kind='state_original_bytes',path=name,**delta))
    delta=file_difference(Path(gold['original_logits_path']),Path(candidate['original_logits_path']),True)
    if not delta['equal']:differences.append(dict(kind='full_vocabulary_f32_bits',**delta))
    return differences



def protocol(plan):
    need(plan['schema_version']==2 and plan['source_baseline']=='868d94e9e3168aeadac478e75d096d992b6bc92e' and
         plan['flag']==FLAG and plan['default_off'] is True and plan['production_adoption'] is False,'Plan scope differs')
    f=plan['fullmodel'];c=f['correctness'];p=f['performance'];a=f['adoption']
    need((f['max_seq_len'],f['max_new_tokens'],f['temperature'],f['seed'])==(327680,32,0,42) and
         f['fresh_context_per_sample'] is True and c['order']==['gold','off','on'] and
         c['warmup']==c['formal_samples']==1,'Correctness protocol differs')
    need(p['order']==['A1','B','A2'] and p['flags']==dict(A1=0,B=1,A2=0) and
         (p['warmup'],p['formal_samples'],p['repeat_limit'],p['memory_sampling_interval_ms'])==(1,3,1,10) and
         p['maximum_absolute_prefill_and_decode_control_drift_percent']==2 and
         p['maximum_conservative_time_regression_percent']==1 and p['physical_memory_control_drift_limit_bytes']==268435456 and
         a['minimum_conservative_peak_footprint_reduction_bytes']==268435456 and
         a['minimum_conservative_prefill_time_saved_percent_strict']==2 and a['max_footprint_regression_bytes']==134217728,
         'Frozen latency/memory/adoption thresholds differ')
    need(set(f['requests'])=={'pf7k','pf16k'} and
         [(f['requests'][k]['expected_prompt_tokens'],f['requests'][k]['expected_prefill_rows']) for k in ('pf7k','pf16k')]==[(6917,6916),(15187,15186)],'Real workload extents differ')



def performance_group(workload,group):
    arms=('A1','B','A2')
    def med(key):return [statistics.median(s[key] for s in a['samples'][1:]) for a in group]
    pf,dec=med('prefill_s'),med('decode_s')
    def peak(key):return [statistics.median(s['observed_peak_'+key+'_bytes'] for s in a['memory']['generation'][1:]) for a in group]
    phys,device,rss=peak('phys_footprint'),peak('device_allocated'),peak('rss')
    pfdrift=(pf[2]/pf[0]-1)*100;ddrift=(dec[2]/dec[0]-1)*100
    pfsave=(1-pf[1]/min(pf[0],pf[2]))*100;dsave=(1-dec[1]/min(dec[0],dec[2]))*100
    physsave=min(phys[0],phys[2])-phys[1];devsave=min(device[0],device[2])-device[1]
    timepass=abs(pfdrift)<=2 and abs(ddrift)<=2 and pfsave>=-1 and dsave>=-1
    mempass=abs(phys[2]-phys[0])<=268435456 and physsave>=-134217728 and devsave>0
    gain=physsave>=268435456 or (pfsave>2 and physsave>=0)
    return dict(workload=workload,prefill_seconds=dict(zip(arms,pf)),decode_seconds=dict(zip(arms,dec)),
        observed_peak_phys_footprint_bytes=dict(zip(arms,phys)),observed_peak_device_allocated_bytes=dict(zip(arms,device)),
        observed_peak_rss_bytes=dict(zip(arms,rss)),prefill_control_drift_percent=pfdrift,decode_control_drift_percent=ddrift,
        conservative_prefill_time_saved_percent=pfsave,conservative_decode_time_saved_percent=dsave,
        conservative_peak_phys_reduction_bytes=physsave,conservative_peak_device_reduction_bytes=devsave,
        latency_gate_pass=timepass,memory_gate_pass=mempass,measured_gain_pass=gain,gate_pass=timepass and mempass and gain)


def archive_bindings(root,source_archives,artifact_archives,arms):
    bindings=load(root/'stage-inputs.json');stages=sorted({Path(v['stage']) for v in arms},key=str)
    need(len(stages)==2,'Exactly independent gold and candidate stages required')
    all_archives=[];index={}
    for kind,archives in (('source',source_archives),('artifacts',artifact_archives)):
        for path in archives:
            count=0
            with tarfile.open(path,'r:*') as archive:
                for member in archive:
                    if member.isdir():continue
                    need(not member.issym() and not member.isdev() and '..' not in Path(member.name).parts,'Unsafe archive entry')
                    stream=archive.extractfile(member);need(stream is not None,'Unreadable archived member')
                    digest=hashlib.sha256();size=0
                    while True:
                        block=stream.read(1048576)
                        if not block:break
                        digest.update(block);size+=len(block)
                    need(member.islnk() or size==member.size,'Archived member extent differs')
                    index.setdefault((kind,digest.hexdigest()),[]).append(member.name);count+=1
            all_archives.append(dict(kind=kind,path=str(path),sha256=sha(path),members_read=count))
    archived=[];external=[]
    for name,digest in bindings.items():
        path=Path(name);matches=[stage for stage in stages if stage in path.parents]
        if matches:
            need(len(matches)==1,'Ambiguous stage source path');relative=str(path.relative_to(matches[0]));kind='artifacts' if relative.startswith('bin/') else 'source'
            candidates=index.get((kind,digest),[])
            need(any(n==relative or n.endswith('/'+relative) for n in candidates),'Frozen source/artifact not archived: '+name)
            archived.append(dict(path=name,sha256=digest,kind=kind))
        else:
            # Frozen approval, leaf reports, and plan are evidence, not executable
            # source. Keep their original paths visible rather than pretending
            # they were members of a source archive.
            need(path.is_file() and sha(path)==digest,'Frozen external evidence differs: '+name)
            external.append(dict(path=name,sha256=digest))
    for record in arms:
        stage=Path(record['stage'])
        required=['Makefile','tools/bench_moe_workspace_fullmodel.c','tools/moe_workspace_fullmodel.mk','tools/moe_workspace_fullmodel_getters.inc','tools/moe_workspace_memory.inc','src/myds_v41.c','src/myds_metal.m','src/myds_v41_gpu.c',
                  'bin/bench_moe_workspace_fullmodel','bin/myds.metallib','bin/myds']
        if record['name']!='gold':required+=['tools/run_moe_workspace_fullmodel_window.py','tools/run_moe_workspace_window.py','tools/run_iq2_signed4_window.py']
        for name in required:
            need(str(stage/name) in bindings,'Critical executed source/artifact is not frozen: '+str(stage/name))
        if record['name']=='gold':
            need(bindings[str(stage/'bin/myds')]==PRODUCTION_BINARY and bindings[str(stage/'bin/myds.metallib')]==PRODUCTION_LIBRARY,
                 'Independent gold does not reproduce exact production artifacts')
            need(all(bindings.get(str(stage/path))==digest for path,digest in GOLD_SOURCE_HASHES.items()),
                 'Gold diagnostic harness does not use all91 exact production source inputs')
    return dict(archives=all_archives,archived_stage_inputs=archived,external_evidence_inputs=external,
                all_frozen_stage_inputs_bound=True)


def window_identity(root):
    before,restored,summary=[load(root/name) for name in ('before.json','restored.json','summary.json')]
    need(all(k in before and before[k]==restored[k] for k in IDENTITY_KEYS) and before['health']==restored['health']=={'status':'ok'},
         'Original production identity, config, nice, argv, environment or health differs after restore')
    need(before['binary_sha256']==PRODUCTION_BINARY and before['metallib_sha256']==PRODUCTION_LIBRARY,
         'Actual original production artifact identity differs')
    need(summary['restoration']['restored'] is True and summary['restoration']['pid']==restored['pid'] and
         summary['restoration']['exact_argv_myds_artifacts_config'] is True and summary['production_adoption'] is False and
         summary['fullmodel_run'] is True,
         'Production restoration/adoption summary differs')
    need(sha(root/'production-artifacts/myds')==PRODUCTION_BINARY and
         sha(root/'production-artifacts/myds.metallib')==PRODUCTION_LIBRARY and
         sha(root/'original-myds.conf')==before['config_sha256'] and
         load(root/'effective-env.json')==before['effective_myds_env'],'Original production recovery payload differs')
    events=records(root/'events.jsonl')
    need(events and events[-1]['kind']=='fullmodel_decision_complete' and events[-1]['status']==summary['status'] and
         not any(r['kind'] in ('monitor_failure','restoration_failure') for r in events),'Competition/restoration failure event')
    env=before['effective_myds_env'];need(len(env)==51,'Actual51 production environment required')
    need(env.get('MYDS_METAL_CONCURRENT')=='1' and env.get('MYDS_PREFIX_CACHE')=='1' and env.get('MYDS_PLD')=='1',
         'Production concurrent/cache/PLD mode differs')
    need(not any(k.startswith(('MYDS_TRACE','MYDS_SKIP_','MYDS_PROFILE','MYDS_KPROF','MYDS_GAP_PROF','MYDS_CAPTURE_')) for k in env),
         'Diagnostic/skip selectors present in original production environment')
    return before,restored,summary



def approval_inputs(root):
    bindings=load(root/'stage-inputs.json');reports=[];plans=[]
    for name,digest in bindings.items():
        if not name.endswith('.json'):continue
        path=Path(name)
        if not path.is_file():continue
        value=load(path)
        if type(value) is not dict:continue
        for key in ('component_gate_pass','recovery_gate_pass'):
            if key in value and value.get('evidence_valid') is True:
                need(value[key] is True and sha(path)==digest,'Frozen prerequisite failed or changed')
                reports.append(dict(key=key,path=str(path),sha256=digest))
        if value.get('experiment')=='Provider-specific prefill MoE scratch arena':
            need(sha(path)==digest,'Frozen current or historical plan SHA differs')
            plans.append((path,digest,value))
    need({r['key'] for r in reports}=={'component_gate_pass','recovery_gate_pass'},'Both independent approved prerequisite reports required')
    # History remains frozen evidence. Only an exact current protocol body may
    # identify the executed plan, and duplicate matching paths stay ambiguous.
    protocol_body=load(root/'protocol.json')
    matches=[p for p in plans if p[2]==protocol_body]
    need(len(matches)==1,'Unique content-matching frozen execution plan required')
    chosen=matches[0]
    need(chosen[1]==load(root/'summary.json')['plan_sha256'],'Selected execution plan differs from controller frozen SHA')
    return dict(reports=reports,plan_path=str(chosen[0]),plan_sha256=chosen[1],
        retained_historical_plans=[dict(path=str(p),sha256=h) for p,h,v in plans if p!=chosen[0]],
        limitation='Every current/historical candidate plan SHA is checked; exactly one matches the current protocol and controller frozen plan SHA. Prerequisite report bytes/pass fields are frozen; component/failure readers are not re-run here.')


def review(root,model,source_archives,artifact_archives):
    plan=load(root/'protocol.json');protocol(plan);before,restored,summary=window_identity(root)
    prerequisites=approval_inputs(root)
    need(summary['model']['bytes']==model.stat().st_size and summary['model_header_sha256']==sha(model,1048576),
         'Actual model extent/prefix identity differs')
    expected=[f'correctness-{workload}-{name}' for workload in ('pf7k','pf16k') for name in ('gold','off','on')]
    expected +=[f'performance-{workload}-{name}' for workload in ('pf7k','pf16k') for name in ('A1','B','A2')]
    steps=load(root/'steps.json');names=[s['name'] for s in steps]
    need(names==expected[:len(names)] and 0<len(names)<=12,'Strict gold/off/on before A1/B/A2 sequence differs')
    hashes=ByteHashes()
    try:
        arms=[arm(root,step,plan,before['effective_myds_env'],hashes) for step in steps]
        need(all(Path(a['metadata']['model']).resolve()==model.resolve() for a in arms),'Actual harness model path differs')
        bindings=load(root/'stage-inputs.json')
        requests={}
        for workload in ('pf7k','pf16k'):
            paths={Path(a['metadata']['request_file']) for a in arms if a['workload']==workload}
            if not paths:continue
            need(len(paths)==1,'Different actual requests within workload');path=paths.pop()
            need(bindings.get(str(path))==plan['fullmodel']['requests'][workload]['sha256'] and path.is_file() and
                 sha(path)==plan['fullmodel']['requests'][workload]['sha256'],'Frozen workload request SHA differs')
            body=path.read_bytes();requests[workload]=dict(bytes=len(body),sha256=sha(path),fnv1a64=hashes.bytes(body))
            for a in arms:
                if a['workload']==workload:
                    need(a['metadata']['request_bytes']==len(body) and a['metadata']['request_fnv1a64']==requests[workload]['fnv1a64'],
                         'Actual harness parsed different request bytes')
        comparisons=[];oracles={};oracle_loads={};oracle_metadata={}
        for a in arms:
            if a['name']=='gold':
                oracles[a['workload']]=a['samples'][0];oracle_loads[a['workload']]=a['load'];oracle_metadata[a['workload']]=a['metadata']
            need(a['workload'] in oracles,'Missing earlier independent production oracle')
            need(all(a['metadata'][k]==oracle_metadata[a['workload']][k] for k in
                     ('prompt_fnv1a64','prompt_bytes','request_fnv1a64','request_bytes','dspark_model')),
                 'Actual rendered prompt or auxiliary model differs from independent gold')
            modes={k:v for k,v in oracle_loads[a['workload']].items() if k.endswith('_mode') and k!='moe_workspace_mode'}
            need(all(a['load'].get(k)==v for k,v in modes.items()),'Other production provider modes differ from independent gold')
            for sample in a['samples']:
                differences=compare_sample(oracles[a['workload']],sample)
                comparisons.append(dict(arm=a['label'],run=sample['run'],full_original_bytes_equal=not differences,differences=differences))
        correct_arms=[a for a in arms if a['phase']=='correctness']
        correctness=(len(correct_arms)==6 and all(a['complete'] and a['returncode']==0 for a in correct_arms) and
            all(c['full_original_bytes_equal'] for c in comparisons if c['arm'].startswith('correctness-')))
        performance_arms=[a for a in arms if a['phase']=='performance']
        need(correctness or not performance_arms,'Performance was executed before all independent correctness gates passed')
        recomputed=[]
        if performance_arms:
            need(len(performance_arms)==6 and all(a['complete'] and a['returncode']==0 for a in performance_arms),
                 'Incomplete performance arm coverage')
            need(all(c['full_original_bytes_equal'] for c in comparisons),'Performance raw bytes differ from independent correctness gold')
            for workload in ('pf7k','pf16k'):
                group=[next(a for a in performance_arms if a['workload']==workload and a['name']==name) for name in ('A1','B','A2')]
                item=performance_group(workload,group)
                recomputed.append(item)
                published=[p for p in summary['performance'] if p['workload']==workload];need(len(published)==1,'Missing workload summary')
                for key,value in item.items():
                    if key=='workload' or type(value) is bool:need(published[0][key]==value,'Published performance decision differs')
                    elif type(value) is dict:need(all(close(published[0][key][n],v) for n,v in value.items()),'Published median differs')
                    else:need(close(published[0][key],value),'Published performance arithmetic differs')
        gate=bool(recomputed) and all(x['gate_pass'] for x in recomputed)
        if correctness:
            need(summary['correctness_pass'] is True,'Correctness summary differs')
        if recomputed:
            need(summary['performance_gate_pass'] is gate and summary['status']==('fullmodel_passed_review_required' if gate else 'fullmodel_rejected_gain_or_controls'),
                 'Runner fullmodel decision differs from independently recomputed gates')
        else:need(summary['status']=='failed','A partial/correctness-failed window cannot claim fullmodel performance')
        archive=archive_bindings(root,source_archives,artifact_archives,arms)
        return dict(schema_version=1,review_kind='independent_CPU_fullmodel_original_bytes_and_frozen_gates',
            reader_sha256=sha(Path(__file__)),root=str(root),protocol_sha256=sha(root/'protocol.json'),
            gpu_work_submitted_by_reader=False,production_adoption=False,artifact_review_passed=True,
            fullmodel_correctness_pass=correctness,fullmodel_performance_measured=bool(recomputed),
            fullmodel_performance_gate_pass=gate,decision='eligible_for_production_adoption_review' if gate else
                ('retain_default_off_fullmodel_performance_gate_failed' if recomputed else 'retain_default_off_correctness_gate_failed_or_incomplete'),
            original_pid=before['pid'],restored_pid=restored['pid'],restoration_exact_including_nice=True,
            requests=requests,arms=arms,full_original_byte_comparisons=comparisons,
            raw_samples_verified=sum(len(a['samples']) for a in arms),
            state_payload_files_verified=sum(len(s['state']['files']) for a in arms for s in a['samples']),
            recomputed_performance=recomputed,prerequisite_frozen_component_approval=prerequisites,**archive,
            limits=['Fresh-context native harness excludes HTTP queue/transport and context creation. Memory phases include context creation separately; generation peaks are sampled at a target 10ms interval.',
                    'Active live state and published snapshot KV/compressor/index/Engram tails are exported. Persistent disk-cache payload, allocator padding, inactive capacity and scratch data are outside state comparison.',
                    'A passing raw review does not itself enable the candidate or modify production.'])
    finally:hashes.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','model','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-archive',type=Path,action='append',required=True)
    parser.add_argument('--artifacts-archive',type=Path,action='append',required=True)
    a=parser.parse_args()
    try:result=review(a.root.resolve(),a.model.resolve(),a.source_archive,a.artifacts_archive)
    except Exception as exc:
        result=dict(schema_version=1,artifact_review_passed=False,production_adoption=False,
                    gpu_work_submitted_by_reader=False,error=repr(exc),reader_sha256=sha(Path(__file__)))
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('artifact_review_passed','decision','error') if k in result},sort_keys=True))
    return 0 if result['artifact_review_passed'] else 1


if __name__=='__main__':sys.exit(main())
