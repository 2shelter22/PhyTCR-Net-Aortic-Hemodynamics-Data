"""Full-cycle PhyTCR-Net and direct neural prediction model definitions."""
import torch
from torch import nn

def mlp(in_features, out_features, width, dropout):
    return nn.Sequential(nn.Linear(in_features,width),nn.SiLU(),nn.Dropout(dropout),nn.Linear(width,out_features))

class SpatialBlock(nn.Module):
    def __init__(self,width,message_passing,dropout):
        super().__init__()
        self.message_passing=message_passing
        if message_passing:
            self.message=mlp(4*width+1,width,width,dropout)
        self.node_update=mlp((3 if message_passing else 2)*width,width,width,dropout)
        self.edge_update=mlp((4 if message_passing else 2)*width,width,width,dropout)
        self.global_update=mlp(5*width,width,width,dropout)
        self.node_norm,self.edge_norm,self.global_norm=[nn.LayerNorm(width) for _ in range(3)]

    def forward(self,h,edge,g,index):
        s,t=index
        ge=g.expand(len(edge),-1)
        if self.message_passing:
            source,target=torch.cat((s,t)),torch.cat((t,s))
            direction=torch.cat((h.new_ones((len(s),1)),-h.new_ones((len(s),1))))
            message=self.message(torch.cat((h[source],h[target],torch.cat((edge,edge)),
                g.expand(len(source),-1),direction),dim=1))
            aggregate=torch.zeros_like(h).index_add(0,target,message)
            degree=torch.bincount(target,minlength=len(h)).clamp_min(1).to(h.dtype)[:,None]
            node_input=torch.cat((h,aggregate/degree,g.expand(len(h),-1)),dim=1)
            edge_input=torch.cat((edge,h[s],h[t],ge),dim=1)
        else:
            node_input=torch.cat((h,g.expand(len(h),-1)),dim=1)
            edge_input=torch.cat((edge,ge),dim=1)
        h=self.node_norm(h+self.node_update(node_input))
        edge=self.edge_norm(edge+self.edge_update(edge_input))
        context=torch.cat((g,h.mean(0,keepdim=True),h.amax(0,keepdim=True),
            edge.mean(0,keepdim=True),edge.amax(0,keepdim=True)),dim=1)
        g=self.global_norm(g+self.global_update(context))
        return h,edge,g

class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation,dropout):
        super().__init__()
        self.conv=nn.Conv1d(channels,channels,5,dilation=dilation,padding=2*dilation,padding_mode='circular')
        self.norm=nn.GroupNorm(1,channels)
        self.act=nn.SiLU()
        self.drop=nn.Dropout(dropout)

    def forward(self,x):
        return x+self.drop(self.act(self.norm(self.conv(x))))

class MeanWaveHead(nn.Module):
    def __init__(self,width,time_inputs,dropout,spatial=False,channels=32):
        super().__init__()
        self.mean=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1,bias=not spatial))
        self.condition=nn.Linear(width,channels)
        self.forcing=nn.Conv1d(time_inputs,channels,1)
        self.blocks=nn.Sequential(*(TemporalBlock(channels,d,dropout) for d in (1,2,4,8)))
        
        self.wave=nn.Conv1d(channels,1,1,bias=False)
        nn.init.normal_(self.mean[-1].weight,std=.001)
        if self.mean[-1].bias is not None:
            nn.init.zeros_(self.mean[-1].bias)
        nn.init.normal_(self.wave.weight,std=.001)

    def forward(self,h,time_features):
        state=torch.nn.functional.silu(self.forcing(time_features.transpose(1,2))+self.condition(h)[:,:,None])
        wave=self.wave(self.blocks(state)).squeeze(1)
        wave=wave-wave.mean(1,keepdim=True)
        return self.mean(h)+wave

class FullCycleGraphModel(nn.Module):
    def __init__(self,schema,use_rom=True,residual=True,message_passing=True,width=128,depth=8,dropout=.05):
        super().__init__()
        if residual and not use_rom:
            raise ValueError('Residual arm requires the fixed physical baseline')
        self.config=dict(schema=schema,use_rom=use_rom,residual=residual,message_passing=message_passing,
            width=width,depth=depth,dropout=dropout)
        self.use_rom,self.residual=use_rom,residual
        node_inputs=schema['node_common']+(schema['node_rom'] if use_rom else 0)
        edge_inputs=schema['edge_common']+(schema['edge_rom'] if use_rom else 0)
        ptime=schema['node_common_time']+(schema['node_rom_time'] if use_rom else 0)
        qtime=schema['edge_common_time']+(schema['edge_rom_time'] if use_rom else 0)
        self.node_encoder=mlp(node_inputs,width,width,dropout)
        self.edge_encoder=mlp(edge_inputs,width,width,dropout)
        self.global_encoder=mlp(schema['global_common'],width,width,dropout)
        self.spatial=nn.ModuleList([SpatialBlock(width,message_passing,dropout) for _ in range(depth)])
        self.p_reference=MeanWaveHead(width,ptime,dropout)
        self.p_spatial=MeanWaveHead(width,ptime,dropout,spatial=True)
        self.q_observation=MeanWaveHead(width,qtime,dropout)

    def forward(self,data):
        node=data['node_common']
        edge=data['edge_common']
        pt,qt=data['node_common_time'],data['edge_common_time']
        if self.use_rom:
            node=torch.cat((node,data['node_rom']),dim=1)
            edge=torch.cat((edge,data['edge_rom']),dim=1)
            pt=torch.cat((pt,data['node_rom_time']),dim=2)
            qt=torch.cat((qt,data['edge_rom_time']),dim=2)
        h=self.node_encoder(node)
        edge_h=self.edge_encoder(edge)
        g=self.global_encoder(data['global_common'][None,:])
        for block in self.spatial:
            h,edge_h,g=block(h,edge_h,g,data['edge_index'])
        reference=self.p_reference(g,pt.mean(0,keepdim=True))
        spatial=self.p_spatial(h,pt)
        spatial=spatial-spatial.mean(0,keepdim=True)
        pressure=reference+spatial
        flow=self.q_observation(edge_h,qt)
        correction_pressure=pressure.transpose(0,1)
        correction_flow=flow.transpose(0,1)
        if self.residual:
            pressure=pressure+data['baseline_P'].transpose(0,1)
            flow=flow+data['baseline_Q'].transpose(0,1)
        s,t=data['edge_index']
        return dict(pressure_norm=pressure.transpose(0,1),flow_norm=flow.transpose(0,1),
            pressure_correction=correction_pressure,flow_correction=correction_flow,
            pressure_reference=reference.squeeze(0),pressure_spatial=spatial.transpose(0,1),
            pressure_drop_correction=(spatial[s]-spatial[t]).transpose(0,1))

def loss_terms(prediction,target,mass_weight=.01):

    terms={}
    for name,key,truthkey,maskkey in (
        ('p','pressure_norm','P','P_mask'),('q','flow_norm','Q','Q_mask')):
        error=prediction[key][:,target[maskkey]]-target[truthkey][:,target[maskkey]]
        mean=error.mean(0)
        terms[name+'_mean']=mean.square().mean()
        terms[name+'_wave']=(error-mean).square().mean()
    if len(target['local_edges']):

        local=prediction['flow_norm'][:,target['local_edges']]-target['local_target_norm']
        terms['local_mass']=local.square().mean()
    else:
        terms['local_mass']=prediction['flow_norm'].new_zeros(())
    terms['supervised']=.5*sum(terms[k] for k in ('p_mean','p_wave','q_mean','q_wave'))
    terms['total']=terms['supervised']+mass_weight*terms['local_mass']
    return terms

def parameter_counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
