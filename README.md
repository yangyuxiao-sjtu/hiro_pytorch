try to make some changes,but have lots of problems  
用  
"""python3 main.py --train""" 进行训练  
目前会报错RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:  
尝试过设置 inplace=False in nn.ReLU and nn.LeakyReLU(https://github.com/NVlabs/FUNIT/issues/23), 但是并不管用  
修改思路：  
原本代码在policy_with_noise中默认to_numpy=True并导致梯度不被记录，因此考虑在原来结构上单独增加tensor_n_sg和tensor_sg来记录to_numpy=False的结果并把tensor_sg转为
numpy的结果重新赋给sg,n_sg来使得新加内容尽可能和原本代码解耦，为此，还需要在hiro_utils中ReplayBuffer增加tensor_sg和tensor_n_sg的buffer  
在subgoal_transition中增加传入sg是tensor的情况判断，因为这种情况下也需要记录梯度  
用low_con_train代替原来的self.low_con.train 因为此时底层策略的train也会修改上层的参数  
用low_con__train代替原来的self.low_con._train，并且在计算底层actor_loss的时候用tenser_sg而不是不带梯度记录的sg进行计算以此来达到反向传播到上层actor的效果  
