"""Convert attention/FFN projection Einsums (constant weight, possibly behind a
Cast) -> Reshape+MatMul so int8/int4 quantization covers them. Handles 3D and 2D:
  btd,dnh->btnh : MatMul(x, W[d,n*h]) -> Reshape [.,.,n,h]
  btnh,dnh->btd : Reshape x[.,.,n*h] -> MatMul(x, W'[n*h,d])
  btd,de->bte   : MatMul(x, W)
  bte,de->btd   : MatMul(x, W.T)
(bhqd,shd->bhqs = attention scores, both activations -> left as Einsum)"""
import onnx, numpy as np
from onnx import helper, numpy_helper

def rewrite_einsum_to_matmul(model):
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    def const_of(name):
        if name in inits: return numpy_helper.to_array(inits[name])
        p = prod.get(name)
        if p is not None and p.op_type == "Cast" and p.input[0] in inits:
            return numpy_helper.to_array(inits[p.input[0]])
        return None
    new, ctr = [], [0]
    U = lambda s: (ctr.__setitem__(0, ctr[0]+1) or f"{s}_mm{ctr[0]}")
    nconv = 0
    for node in g.node:
        if node.op_type != "Einsum": new.append(node); continue
        eq = "".join(a.s.decode() for a in node.attribute if a.name=="equation").replace(" ","")
        W = const_of(node.input[1])
        if eq not in ("btd,dnh->btnh","btnh,dnh->btd","btd,de->bte","bte,de->btd") or W is None:
            new.append(node); continue
        x, out = node.input[0], node.output[0]
        def addW(arr):
            nm=U("w"); g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(arr), nm)); return nm
        def addS(arr):
            nm=U("s"); g.initializer.append(numpy_helper.from_array(np.array(arr,np.int64), nm)); return nm
        if eq == "btd,dnh->btnh":
            d,n,h = W.shape; wn=addW(W.reshape(d,n*h)); mm=U("t")
            new.append(helper.make_node("MatMul",[x,wn],[mm]))
            new.append(helper.make_node("Reshape",[mm,addS([0,0,n,h])],[out]))
        elif eq == "btnh,dnh->btd":
            d,n,h = W.shape; wn=addW(np.transpose(W,(1,2,0)).reshape(n*h,d)); xr=U("x")
            new.append(helper.make_node("Reshape",[x,addS([0,0,n*h])],[xr]))
            new.append(helper.make_node("MatMul",[xr,wn],[out]))
        elif eq == "btd,de->bte":
            new.append(helper.make_node("MatMul",[x,addW(W)],[out]))
        else:  # bte,de->btd
            new.append(helper.make_node("MatMul",[x,addW(W.T)],[out]))
        nconv += 1
    del g.node[:]; g.node.extend(new)
    used=set(i for nd in g.node for i in nd.input)
    keep=[i for i in g.initializer if i.name in used]
    del g.initializer[:]; g.initializer.extend(keep)
    return model, nconv
