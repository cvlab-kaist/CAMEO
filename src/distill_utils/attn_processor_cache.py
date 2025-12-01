# just for coding
import torch

from my_diffusers.models.attention_processor import AttnProcessor2_0

# unet layers: total 16
# down_blocks(6): 0.0, 0.1, 1.0, 1.1, 2.0, 2.1
# mid_block(1): 0.0
# up_blocks(9): 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2

# sorted attn processor  = ['down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor']
# Attn Head Num: [5,5,10,10,20,20] [20] [20,20,20, 10,10,10, 5,5,5]

SDXL_ATTN_DIM = [320, 320, 640, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320, 320]
SDXL_ATTN_HEAD_NUM = [5,5,10,10,20,20, 20, 20,20,20,10,10,10,5,5,5]


def print_attn_cache_setting(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        print(f"Layer {i}: {name}, cache_attn: {unet.attn_processors[name].cache_attn}, cache_qk: {unet.attn_processors[name].cache_qk}")

def set_attn_cache( unet,  cache_layers = [] , print_=False, cache_qk=False):
    ''' cache_layers: list of str(int) layer index to cache attention (0~15) 
    '''
    # 10/24: added cache_qk option, cache_layers format: int -> str(int) 
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        if (i in cache_layers) or (str(i) in cache_layers): # string format
            # print(f"Cache attn layer: {i}={name}")
            if cache_qk:
                unet.attn_processors[name].cache_qk = True 
            else:
                unet.attn_processors[name].cache_attn = True
    if print_:
        print_attn_cache_setting(unet)

 ##  1019
def set_qk_cache(unet, cache_layers = [], print_=False):
    '''Enable caching of query/key for selected layers without changing existing behavior.'''
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        if i in cache_layers:
            unet.attn_processors[name].cache_qk = True  
        else:
            unet.attn_processors[name].cache_qk = False  
    if print_:
        for i, name in enumerate(attn_proc_names):
            print(f"Layer {i}: {name}, cache_qk: {getattr(unet.attn_processors[name], 'cache_qk', False)}")
            
 ##  1019
def unset_qk_cache(unet, print_=False):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        setattr(unet.attn_processors[name], 'cache_qk', False)  
    if print_:
        for i, name in enumerate(attn_proc_names):
            print(f"Layer {i}: {name}, cache_qk: {getattr(unet.attn_processors[name], 'cache_qk', False)}")

def unset_attn_cache(unet, print_=False):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        unet.attn_processors[name].cache_attn = False
        unet.attn_processors[name].cache_qk = False
    if print_:
        print_attn_cache_setting(unet)

def pop_cached_attn(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    block_attn_cache = {}
    for i, name in enumerate(attn_proc_names):
        if unet.attn_processors[name].cache_attn:
            cache_list = unet.attn_processors[name].cache['attn']
            assert len(cache_list) == 1, f"Expected single cached attention tensor, got {len(cache_list)}"
            block_attn_cache[str(i)] = cache_list.pop()
        elif unet.attn_processors[name].cache_qk:
            q_cache_list = unet.attn_processors[name].cache['q']
            k_cache_list = unet.attn_processors[name].cache['k']
            assert len(q_cache_list) == 1, f"Expected single cached q tensor, got {len(q_cache_list)}"
            assert len(k_cache_list) == 1, f"Expected single cached k tensor, got {len(k_cache_list)}"
            block_attn_cache[str(i)] = {
                'q': q_cache_list.pop(),
                'k': k_cache_list.pop()
            }
    return block_attn_cache

 ##  1019
def pop_cached_qk(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    block_qk_cache = {}
    for i, name in enumerate(attn_proc_names):
        proc = unet.attn_processors[name]
        if getattr(proc, 'cache_qk', False):  # ##  1019
            q_list = proc.cache.get('q', [])
            k_list = proc.cache.get('k', [])
            # expect at most one cached q/k per layer to mirror attn cache behavior
            if len(q_list) == 1 and len(k_list) == 1:
                block_qk_cache[str(i)] = (q_list.pop(), k_list.pop())
            elif len(q_list) == 0 and len(k_list) == 0:
                continue
            else:
                raise AssertionError(f"Expected single cached q/k tensor per layer, got q:{len(q_list)} k:{len(k_list)}")
    return block_qk_cache

 ##  1019
def clear_qk_cache(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        proc = unet.attn_processors[name]
        proc.cache.get('q', []).clear()
        proc.cache.get('k', []).clear()
    torch.cuda.empty_cache()

def clear_attn_cache(unet, print_=False):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        unet.attn_processors[name].cache['attn'].clear()
        unet.attn_processors[name].cache['q'].clear()
        unet.attn_processors[name].cache['k'].clear()
    torch.cuda.empty_cache()
    if print_:
        print_attn_cache_setting(unet)  
# Feat
def print_feat_cache_setting(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        print(f"Layer {i}: {name}, cache_feat: {unet.attn_processors[name].cache_feat}")

def set_feat_cache( unet,  cache_layers = [], print_=False):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        if (str(i) in cache_layers) or (i in cache_layers):
            # print(f"Cache feat layer: {name}")
            unet.attn_processors[name].cache_feat = True
        else:
            unet.attn_processors[name].cache_feat = False
    if print_:
        print_feat_cache_setting(unet)

def unset_feat_cache(unet, print_=False):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        unet.attn_processors[name].cache_feat = False
    if print_:
        print_feat_cache_setting(unet)

def pop_cached_feat(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    block_feat_cache = {}
    for i, name in enumerate(attn_proc_names):
        if unet.attn_processors[name].cache_feat:
            cache_list = unet.attn_processors[name].cache['feat']
            assert len(cache_list) == 1, f"Expected single cached feature tensor, got {len(cache_list)}"
            block_feat_cache[str(i)] = cache_list.pop()
    return block_feat_cache

def clear_feat_cache(unet):
    attn_proc_names = sorted(list(unet.attn_processors.keys()))
    for i, name in enumerate(attn_proc_names):
        unet.attn_processors[name].cache['feat'].clear()
    torch.cuda.empty_cache()