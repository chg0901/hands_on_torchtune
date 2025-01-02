# hands_on_torchtune
torchtune实践

## Huggingface
**Huggingface Model Path**: https://huggingface.co/chg0901/llama3_2_3B_Instruct_alphaca_epoch_1/tree/main

## ZhiHu

**知乎文章**: [torchtune lora微调上手体验](https://zhuanlan.zhihu.com/p/15661723645)

## Problems To Solve

中间遇到两个没解决的问题, 感兴趣的可以翻一下, 详情请查看知乎

- https://github.com/pytorch/torchtune/issues/2224 ： 不能加载微调完模型, 官方链接里说可以直接加载
  
  ![a30a47d38763da0e42ffb01c907257bb](https://github.com/user-attachments/assets/82820909-1876-45c5-a22c-10e2d7309dd3)
  
  ![image](https://github.com/user-attachments/assets/75528ed5-59b8-417b-a2ca-2ed1d1249f71)

  

- https://github.com/pytorch/torchtune/issues/2218 ： 不能用它的length_squence pack pipeline
  
  ![2b97eea7c01b13a23071a414098f13bc](https://github.com/user-attachments/assets/9e065445-528f-4050-aed3-3887ff2bd31b)

  ![image](https://github.com/user-attachments/assets/0ad6099e-c627-49a2-b10b-34d2deab6a38)
