import torch

from goodDataML.learning.horizontal.torch_model.testnet_models import Net1, Net2

if __name__ == '__main__':
    net1 = Net1()
    net2 = Net2()

    net1_path = './model1.pt'
    net2_path = './model2.pt'

    torch.save(net1, net1_path)
    torch.save(net2, net2_path)

    model = torch.load(net1_path)
    print(model)