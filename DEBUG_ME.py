import torch

inp = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_inp.pt')
conv = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_conv.pt')
out = torch.nn.functional.conv2d(inp, conv, bias=None, stride=1, padding=1, dilation=1, groups=1)

d = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_out.pt')
print((out-d).abs().max())
tensor(0.0044, device='cuda:0', grad_fn=<MaxBackward1>)

inp = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_inp.pt').to('cpu')
conv = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_conv.pt').to('cpu')
out = torch.nn.functional.conv2d(inp, conv, bias=None, stride=1, padding=1, dilation=1, groups=1)

d = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_out.pt')
print((out-d).abs().max())
tensor(0., grad_fn=<MaxBackward1>)

out = torch.nn.functional.conv2d(inp, conv, bias=None, stride=1, padding=1, dilation=1, groups=1)
torch.save(out, '/home/thomwolf/Documents/Github/ACT/tensor_out_lerobot.pt')

=====

inp = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_inp.pt')
conv = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_conv.pt')
out = torch.nn.functional.conv2d(inp, conv, bias=None, stride=1, padding=1, dilation=1, groups=1)

torch.save(out, '/home/thomwolf/Documents/Github/ACT/tensor_out_lerobot.pt')

act = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_out_act.pt')
ler = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_out_lerobot.pt')
print((act-ler).abs().max())

====

inp = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_inp.pt')
conv = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_conv.pt')
out = torch.nn.functional.conv2d(inp, conv, bias=None, stride=1, padding=1, dilation=1, groups=1)

torch.save(out, '/home/thomwolf/Documents/Github/ACT/tensor_out_act.pt')

act = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_out_act.pt')
ler = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_out_lerobot.pt')
print((act-ler).abs().max())

===

torch.save(features, '/home/thomwolf/Documents/Github/ACT/tensor_features_act.pt')
torch.save(cam_features, '/home/thomwolf/Documents/Github/ACT/tensor_features_ler.pt')

torch.save(output, '/home/thomwolf/Documents/Github/ACT/tensor_features_act.pt')
torch.save(src2, '/home/thomwolf/Documents/Github/ACT/tensor_features_act.pt')
torch.save(src, '/home/thomwolf/Documents/Github/ACT/tensor_features_act.pt')
torch.save(x, '/home/thomwolf/Documents/Github/ACT/tensor_features_ler.pt')

act = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_features_act.pt')
ler = torch.load('/home/thomwolf/Documents/Github/ACT/tensor_features_ler.pt')
print((act-ler).abs().max())

cam_features[0]

===

torch.save(self.self_attn.q_proj_weight, '/home/thomwolf/Documents/Github/ACT/tensor_q_proj_weight_act.pt')
torch.save(self.self_attn.k_proj_weight, '/home/thomwolf/Documents/Github/ACT/tensor_k_proj_weight_act.pt')
torch.save(self.self_attn.v_proj_weight, '/home/thomwolf/Documents/Github/ACT/tensor_v_proj_weight_act.pt')

torch.save(self.self_attn.q_proj_weight, '/home/thomwolf/Documents/Github/ACT/tensor_q_proj_weight_ler.pt')
torch.save(self.self_attn.k_proj_weight, '/home/thomwolf/Documents/Github/ACT/tensor_k_proj_weight_ler.pt')
torch.save(self.self_attn.v_proj_weight, '/home/thomwolf/Documents/Github/ACT/tensor_v_proj_weight_ler.pt')

===
torch.save(q, '/home/thomwolf/Documents/Github/ACT/tensor_q_act.pt')
torch.save(k, '/home/thomwolf/Documents/Github/ACT/tensor_k_act.pt')
torch.save(src, '/home/thomwolf/Documents/Github/ACT/tensor_values_act.pt')

torch.save(self.self_attn.in_proj_weight, '/home/thomwolf/Documents/Github/ACT/tensor_in_proj_weight_act.pt')
torch.save(self.self_attn.in_proj_bias, '/home/thomwolf/Documents/Github/ACT/tensor_in_proj_bias_act.pt')
torch.save(self.self_attn.out_proj.weight, '/home/thomwolf/Documents/Github/ACT/tensor_out_proj.weight_act.pt')
torch.save(self.self_attn.out_proj.bias, '/home/thomwolf/Documents/Github/ACT/tensor_out_proj.bias_act.pt')


torch.save(q, '/home/thomwolf/Documents/Github/ACT/tensor_q_ler.pt')
torch.save(k, '/home/thomwolf/Documents/Github/ACT/tensor_k_ler.pt')
torch.save(x, '/home/thomwolf/Documents/Github/ACT/tensor_values_ler.pt')

torch.save(self.self_attn.in_proj_weight, '/home/thomwolf/Documents/Github/ACT/tensor_in_proj_weight_ler.pt')
torch.save(self.self_attn.in_proj_bias, '/home/thomwolf/Documents/Github/ACT/tensor_in_proj_bias_ler.pt')
torch.save(self.self_attn.out_proj.weight, '/home/thomwolf/Documents/Github/ACT/tensor_out_proj.weight_ler.pt')
torch.save(self.self_attn.out_proj.bias, '/home/thomwolf/Documents/Github/ACT/tensor_out_proj.bias_ler.pt')

weights = ['in_proj_weight', 'in_proj_bias', 'out_proj.weight',
           'out_proj.bias', 'q', 'k',
           'values']  #, 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
for i in weights:
    act = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_{i}_act.pt')
    ler = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_{i}_ler.pt')
    print(i)
    print((act-ler).abs().max())

----

act_ler = 'act'

query = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_q_{act_ler}.pt')
key = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_k_{act_ler}.pt')
value = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_values_{act_ler}.pt')
in_proj_weight = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_in_proj_weight_{act_ler}.pt')
in_proj_bias = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_in_proj_bias_{act_ler}.pt')
out_proj_weight = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_out_proj.weight_{act_ler}.pt')
out_proj_bias = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_out_proj.bias_{act_ler}.pt')

torch.use_deterministic_algorithms(True)
attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
    query, key, value, 512, 8,
    in_proj_weight, in_proj_bias,
    None, None, False,
    0.1, out_proj_weight, out_proj_bias,
    training=False,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False)

torch.save(attn_output, f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_{act_ler}.pt')
print(act_ler)

act_ler = 'ler'

query = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_q_{act_ler}.pt')
key = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_k_{act_ler}.pt')
value = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_values_{act_ler}.pt')
in_proj_weight = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_in_proj_weight_{act_ler}.pt')
in_proj_bias = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_in_proj_bias_{act_ler}.pt')
out_proj_weight = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_out_proj.weight_{act_ler}.pt')
out_proj_bias = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_out_proj.bias_{act_ler}.pt')

torch.use_deterministic_algorithms(True)
attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
    query, key, value, 512, 8,
    in_proj_weight, in_proj_bias,
    None, None, False,
    0.1, out_proj_weight, out_proj_bias,
    training=False,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False)

torch.save(attn_output, f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_{act_ler}.pt')
print(act_ler)

weights = ['in_proj_weight', 'in_proj_bias', 'out_proj.weight',
           'out_proj.bias', 'q', 'k',
           'values']  #, 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
for i in weights:
    act = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_{i}_act.pt')
    ler = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_{i}_ler.pt')
    print(i)
    print((act-ler).abs().max())

act = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_act.pt')
ler = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_ler.pt')
print((act-ler).abs().max())


---

attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
    query.to('cpu'), key.to('cpu'), value.to('cpu'), 512, 8,
    in_proj_weight.to('cpu'), in_proj_bias.to('cpu'),
    None, None, False,
    0.1, out_proj_weight.to('cpu'), out_proj_bias.to('cpu'),
    training=False,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False)

torch.save(attn_output, f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_cpu.pt')

act = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_act.pt')
cpu = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_cpu.pt')
print((act.to('cpu')-cpu).abs().max())

ler = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_ler.pt')
cpu = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_cpu.pt')
print((ler.to('cpu')-cpu).abs().max())

cpu2 = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_cpu_2.pt')
cpu = torch.load(f'/home/thomwolf/Documents/Github/ACT/tensor_attn_output_cpu.pt')
print((ler.to('cpu')-cpu).abs().max())
